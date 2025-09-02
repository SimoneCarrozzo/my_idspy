import types
from enum import Enum
from typing import Sequence, Optional, Iterable, List, Any, Callable, Dict, Tuple, Union

from .state import State
from .step import Step, FitAwareStep
from ..events.bus import EventBus
from ..events.events import Event


class PipelineEvent(str, Enum):
    """Typed identifiers for pipeline lifecycle and step events."""
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_ERROR = "on_error"


EventKey = Union[str, PipelineEvent]
HookFunc = Callable[..., None]


def _normalize_event_key(event: EventKey) -> str:
    return event.value if isinstance(event, PipelineEvent) else str(event)


def hook(event: EventKey, *, priority: int = 0) -> Callable[[HookFunc], HookFunc]:
    """
    Decorator: mark a bound method as a hook handler for `event` with an integer priority.

    Lower priority executes earlier. Multiple events can be attached to the same function
    by stacking decorators.
    """

    def deco(fn: HookFunc) -> HookFunc:
        hooks = getattr(fn, "__hooks__", [])
        hooks.append((_normalize_event_key(event), priority))
        setattr(fn, "__hooks__", hooks)
        return fn

    return deco


class Pipeline(Step):
    """
    A Step that runs a sequence of sub-steps with lightweight decorator-based hooks.

    Hooks can be declared via @hook(...) on methods of this instance or registered
    dynamically at runtime with `add_hook`.
    """

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

    def _bind_decorated_hooks(self) -> None:
        for cls in self.__class__.mro():
            for _, fn in cls.__dict__.items():
                if isinstance(fn, types.FunctionType) and hasattr(fn, "__hooks__"):
                    bound = fn.__get__(self, self.__class__)
                    for event, prio in getattr(fn, "__hooks__", []):
                        self._hooks.setdefault(event, []).append((prio, bound))

        for ev in self._hooks:
            # lower priority runs first
            self._hooks[ev].sort(key=lambda t: t[0])

    def add_hook(self, event: EventKey, func: HookFunc, *, priority: int = 0) -> None:
        """Register a runtime hook for an `event`."""
        key = _normalize_event_key(event)
        self._hooks.setdefault(key, []).append((priority, func))
        self._hooks[key].sort(key=lambda t: t[0])

    def remove_hook(self, event: EventKey, func: HookFunc) -> None:
        """Remove a previously registered runtime hook for an `event` (no-op if missing)."""
        key = _normalize_event_key(event)
        lst = self._hooks.get(key)
        if not lst:
            return
        self._hooks[key] = [pair for pair in lst if pair[1] is not func]
        if not self._hooks[key]:
            del self._hooks[key]

    def _fire(self, event: EventKey, *args: Any, **kwargs: Any) -> None:
        """Invoke all hooks for `event` in priority order."""
        key = _normalize_event_key(event)
        for _, fn in list(self._hooks.get(key, ())):
            fn(*args, **kwargs)

    def run(self, state: State) -> None:
        """Execute all steps sequentially, emitting lifecycle events."""
        self._fire(PipelineEvent.PIPELINE_START, state)
        try:
            for idx, step in enumerate(self.steps):
                self._fire(PipelineEvent.BEFORE_STEP, step, state, index=idx)
                try:
                    step.execute(state)
                except Exception as e:
                    self._fire(PipelineEvent.ON_ERROR, state, e, step=step, index=idx)
                    raise
                else:
                    self._fire(PipelineEvent.AFTER_STEP, step, state, index=idx)
        finally:
            self._fire(PipelineEvent.PIPELINE_END, state)

    def add_step(self, step: Step) -> None:
        """Append a sub-step to the pipeline."""
        self.steps.append(step)

    def extend(self, steps: Sequence[Step]) -> None:
        """Append multiple sub-steps to the pipeline."""
        self.steps.extend(steps)

    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        return (
            f"{self.__class__.__name__}"
            f"(steps={step_names!r}, requires={self.requires!r}, provides={self.provides!r})"
        )


class ObservablePipeline(Pipeline):
    """Publishes pipeline lifecycle events to an EventBus."""

    def __init__(self, *args: Any, bus: Optional[EventBus] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bus: EventBus = bus or EventBus()

    @staticmethod
    def _ctx(state: State):
        return state.get_view()

    def _label(self, step: Optional[Step]) -> str:
        return f"{self.name}.{step.name}" if step is not None else self.name

    @hook(PipelineEvent.PIPELINE_START)
    def start(self, state: State) -> None:
        self.bus.publish(Event(PipelineEvent.PIPELINE_START, self.name, context=self._ctx(state)))

    @hook(PipelineEvent.PIPELINE_END)
    def end(self, state: State) -> None:
        self.bus.publish(Event(PipelineEvent.PIPELINE_END, self.name, context=self._ctx(state)))

    @hook(PipelineEvent.BEFORE_STEP, priority=10)
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
                context=self._ctx(state),
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
                context=self._ctx(state),
            )
        )

    @hook(PipelineEvent.ON_ERROR)
    def on_error(
            self,
            state: State,
            e: Exception,
            step: Optional[Step],
            index: Optional[int],
    ) -> None:
        self.bus.publish(
            Event(
                PipelineEvent.ON_ERROR,
                self._label(step),
                payload={"index": index, "error": repr(e)},
                context=self._ctx(state),
            )
        )


class FitAwarePipeline(Pipeline):
    """Pipeline that ensures any FitAwareStep sub-steps is fitted before running."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._is_fitted = False

    @hook(PipelineEvent.PIPELINE_START, priority=1)
    def start(self, state: State) -> None:
        for step in self.steps:
            if isinstance(step, FitAwareStep):
                step.fit(state)
        self._is_fitted = True


class FitAwareObservablePipeline(FitAwarePipeline, ObservablePipeline):
    """A FitAwarePipeline that also publishes events."""
    pass
