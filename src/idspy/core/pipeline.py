from collections import defaultdict
from enum import Enum
from typing import (
    Sequence,
    Optional,
    Iterable,
    List,
    Any,
    Callable,
    Dict,
    Tuple,
    Union,
    Mapping,
)

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


class HookRegistry:
    """Efficient hook registry with lazy sorting and caching."""

    def __init__(self):
        self._hooks: Dict[str, List[Tuple[int, HookFunc]]] = defaultdict(list)
        self._sorted_cache: Dict[str, List[HookFunc]] = {}

    def add(self, event: str, func: HookFunc, priority: int = 0) -> None:
        """Add a hook with priority."""
        self._hooks[event].append((priority, func))
        # Invalidate cache for this event
        self._sorted_cache.pop(event, None)

    def remove(self, event: str, func: HookFunc) -> None:
        """Remove a hook."""
        if event in self._hooks:
            self._hooks[event] = [
                (p, f) for p, f in self._hooks[event] if f is not func
            ]
            if not self._hooks[event]:
                del self._hooks[event]
            self._sorted_cache.pop(event, None)

    def get_hooks(self, event: str) -> List[HookFunc]:
        """Get sorted hooks for event (cached)."""
        if event not in self._sorted_cache:
            hooks = self._hooks.get(event, [])
            # Sort by priority (lower first) and cache result
            self._sorted_cache[event] = [
                func for _, func in sorted(hooks, key=lambda x: x[0])
            ]
        return self._sorted_cache[event]


def _normalize_event_key(event: EventKey) -> str:
    return event.value if isinstance(event, PipelineEvent) else str(event)


def hook(event: EventKey, *, priority: int = 0) -> Callable[[HookFunc], HookFunc]:
    """Decorator to mark a method as a hook for `event` with priority (lower runs first)."""

    def decorator(func: HookFunc) -> HookFunc:
        # Store hook metadata directly on function
        if not hasattr(func, "_hook_events"):
            func._hook_events = []
        func._hook_events.append((_normalize_event_key(event), priority))
        return func

    return decorator


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
        self._hook_registry = HookRegistry()
        self._register_decorated_hooks()

    # --- hooks registration ---

    def _register_decorated_hooks(self) -> None:
        """Efficiently register all decorated hooks from class hierarchy."""
        # Use a set to avoid duplicate registration
        seen_methods = set()

        for cls in reversed(self.__class__.mro()):  # Start from base classes
            for name, method in cls.__dict__.items():
                if (
                    name not in seen_methods
                    and callable(method)
                    and hasattr(method, "_hook_events")
                ):

                    bound_method = getattr(self, name)
                    for event, priority in method._hook_events:
                        self._hook_registry.add(event, bound_method, priority)
                    seen_methods.add(name)

    def add_hook(self, event: EventKey, func: HookFunc, *, priority: int = 0) -> None:
        """Register a runtime hook."""
        key = _normalize_event_key(event)
        self._hook_registry.add(key, func, priority)

    def remove_hook(self, event: EventKey, func: HookFunc) -> None:
        """Remove a runtime hook."""
        key = _normalize_event_key(event)
        self._hook_registry.remove(key, func)

    def _fire(self, event: EventKey, *args: Any, **kwargs: Any) -> None:
        """Invoke hooks for `event` in priority order."""
        key = _normalize_event_key(event)
        for hook_func in self._hook_registry.get_hooks(key):
            hook_func(*args, **kwargs)

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

    def __init__(
        self, *args: Any, bus: Optional[EventBus] = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bus: EventBus = bus or EventBus()

    @staticmethod
    def _ctx(state: State) -> Mapping[str, Any]:
        return state.readonly()

    def _label(self, step: Optional[Step]) -> str:
        return f"{self.name}.{step.name}" if step is not None else self.name

    @hook(PipelineEvent.PIPELINE_START)
    def start(self, state: State) -> None:
        self.bus.publish(
            Event(PipelineEvent.PIPELINE_START, self.name, state=self._ctx(state))
        )

    @hook(PipelineEvent.PIPELINE_END)
    def end(self, state: State) -> None:
        self.bus.publish(
            Event(PipelineEvent.PIPELINE_END, self.name, state=self._ctx(state))
        )

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
