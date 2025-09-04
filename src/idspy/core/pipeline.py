import types
from enum import Enum
from typing import Sequence, Optional, Iterable, List, Any, Callable, Dict, Tuple, Union, Mapping

from .state import State
from .step import Step, FitAwareStep
from ..events.bus import EventBus
from ..events.events import Event


class PipelineEvent(str, Enum):
    """Pipeline lifecycle and step events."""
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    BEFORE_STEP = "before_step"  # args: (step, state, index=int)
    AFTER_STEP = "after_step"  # args: (step, state, index=int)
    ON_ERROR = "on_error"  # args: (state, exc, step=..., index=int)


EventKey = Union[str, PipelineEvent]
HookFunc = Callable[..., None]


def _normalize_event_key(event: EventKey) -> str:
    return event.value if isinstance(event, PipelineEvent) else str(event)


def hook(event: EventKey, *, priority: int = 0) -> Callable[[HookFunc], HookFunc]:
    """Decorator to mark a method as a hook for `event` with priority (lower runs first)."""

    def deco(fn: HookFunc) -> HookFunc:
        hooks = getattr(fn, "__hooks__", [])
        hooks.append((_normalize_event_key(event), priority))
        setattr(fn, "__hooks__", hooks)
        return fn

    return deco


class Pipeline(Step):
    """Step that runs a sequence of sub-steps with hook support."""

    def __init__(
            self,
            steps: Sequence[Step],
            name: Optional[str] = None,
            requires: Optional[Iterable[str]] = None,
            provides: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(name=name, requires=requires, provides=provides)
        self.steps: List[Step] = list(steps)
        self._hooks: Dict[str, List[Tuple[int, HookFunc]]] = {}
        self._bind_decorated_hooks()

    # --- hooks registration ---

    def _bind_decorated_hooks(self) -> None:
        for cls in self.__class__.mro():
            for _, fn in cls.__dict__.items():
                if isinstance(fn, types.FunctionType) and hasattr(fn, "__hooks__"):
                    bound = fn.__get__(self, self.__class__)
                    for ev, prio in getattr(fn, "__hooks__", []):
                        self._hooks.setdefault(ev, []).append((prio, bound))
        for ev in self._hooks:
            self._hooks[ev].sort(key=lambda t: t[0])

    def add_hook(self, event: EventKey, func: HookFunc, *, priority: int = 0) -> None:
        """Register a runtime hook."""
        key = _normalize_event_key(event)
        self._hooks.setdefault(key, []).append((priority, func))
        self._hooks[key].sort(key=lambda t: t[0])

    def remove_hook(self, event: EventKey, func: HookFunc) -> None:
        """Remove a runtime hook (no-op if missing)."""
        key = _normalize_event_key(event)
        lst = self._hooks.get(key)
        if not lst:
            return
        self._hooks[key] = [pair for pair in lst if pair[1] is not func]
        if not self._hooks[key]:
            del self._hooks[key]

    def _fire(self, event: EventKey, *args: Any, **kwargs: Any) -> None:
        """Invoke hooks for `event` in priority order."""
        key = _normalize_event_key(event)
        for _, fn in list(self._hooks.get(key, ())):
            fn(*args, **kwargs)

    # --- Step API ---

    def run(self, state: State) -> None:
        """Execute sub-steps sequentially and emit events."""
        self._fire(PipelineEvent.PIPELINE_START, state)
        try:
            for idx, step in enumerate(self.steps):
                self._fire(PipelineEvent.BEFORE_STEP, step, state, index=idx)
                try:
                    step(state)  # __call__: respects Step validations
                except Exception as e:
                    self._fire(PipelineEvent.ON_ERROR, state, e, step=step, index=idx)
                    raise
                else:
                    self._fire(PipelineEvent.AFTER_STEP, step, state, index=idx)
        finally:
            self._fire(PipelineEvent.PIPELINE_END, state)

    # --- convenience ---

    def add_step(self, step: Step) -> None:
        """Append a sub-step."""
        self.steps.append(step)

    def extend(self, steps: Sequence[Step]) -> None:
        """Append multiple sub-steps."""
        self.steps.extend(steps)

    def __repr__(self) -> str:
        names = [s.name for s in self.steps]
        return f"{self.__class__.__name__}(steps={names!r}, requires={len(self.requires)}, provides={len(self.provides)})"


class ObservablePipeline(Pipeline):
    """Pipeline that publishes lifecycle events to an EventBus."""

    def __init__(self, *args: Any, bus: Optional[EventBus] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bus: EventBus = bus or EventBus()

    @staticmethod
    def _ctx(state: State) -> Mapping[str, Any]:
        return state.readonly()

    def _label(self, step: Optional[Step]) -> str:
        return f"{self.name}.{step.name}" if step is not None else self.name

    @hook(PipelineEvent.PIPELINE_START)
    def start(self, state: State) -> None:
        self.bus.publish(Event(PipelineEvent.PIPELINE_START, self.name, state=self._ctx(state)))

    @hook(PipelineEvent.PIPELINE_END)
    def end(self, state: State) -> None:
        self.bus.publish(Event(PipelineEvent.PIPELINE_END, self.name, state=self._ctx(state)))

    @hook(PipelineEvent.BEFORE_STEP)
    def before_step(self, step: Step, state: State, *, index: int) -> None:
        self.bus.publish(
            Event(
                PipelineEvent.BEFORE_STEP,
                self._label(step),
                payload={
                    "index": index,
                    "requires": list(step.requires),
                    "provides": list(step.provides),
                },
                state=self._ctx(state),
            )
        )

    @hook(PipelineEvent.AFTER_STEP)
    def after_step(self, step: Step, state: State, *, index: int) -> None:
        self.bus.publish(
            Event(
                PipelineEvent.AFTER_STEP,
                self._label(step),
                payload={
                    "index": index,
                    "requires": list(step.requires),
                    "provides": list(step.provides),
                },
                state=self._ctx(state),
            )
        )

    @hook(PipelineEvent.ON_ERROR)
    def on_error(self, state: State, exc: Exception, *, step: Step, index: int) -> None:
        self.bus.publish(
            Event(
                PipelineEvent.ON_ERROR,
                self._label(step),
                payload={"index": index, "error": repr(exc)},
                state=self._ctx(state),
            )
        )


class FitAwarePipeline(Pipeline):
    """Pipeline that fits FitAwareStep sub-steps before running."""

    def __init__(
            self,
            *args: Any,
            refit: bool = False,
            **kwargs: Any,
    ) -> None:
        """
        refit: if False, fit only steps not yet fitted; if True, always fit.
        fit_priority: hook priority; lower runs earlier.
        """
        super().__init__(*args, **kwargs)
        self.refit = refit
        self._is_fitted = False

    @hook(PipelineEvent.PIPELINE_START, priority=1)
    def _fit_on_start(self, state: State) -> None:
        for step in self.steps:
            if isinstance(step, FitAwareStep):
                if self.refit or not step.is_fitted:
                    step.fit(state)
        self._is_fitted = True

    @property
    def is_fitted(self) -> bool:
        """Whether the pipeline has executed its fit hook at least once."""
        return self._is_fitted

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"{base}, fitted={self._is_fitted}, refit={self.refit})"


class FitAwareObservablePipeline(FitAwarePipeline, ObservablePipeline):
    """FitAwarePipeline that also publishes events."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # cooperative MRO: both parents call super().__init__
        super().__init__(*args, **kwargs)
