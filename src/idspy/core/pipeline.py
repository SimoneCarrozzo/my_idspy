from enum import Enum
from typing import (
    Sequence,
    Optional,
    List,
    Any,
    Callable,
    Dict,
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
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_ERROR = "on_error"


EventKey = Union[str, PipelineEvent]
HookFunc = Callable[..., None]


class Pipeline(Step):
    """Step that runs a sequence of sub-steps with hook support."""

    @staticmethod
    def hook(event: PipelineEvent, priority: int = 0):
        """Decorator to register a method as a hook for an event."""

        def decorator(func: HookFunc) -> HookFunc:
            if not hasattr(func, "_pipeline_hooks"):
                func._pipeline_hooks = []
            func._pipeline_hooks.append((event.value, priority))
            return func

        return decorator

    def __init__(
        self,
        steps: Sequence[Step],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.steps: List[Step] = list(steps)

        # Get hooks from decorators
        self._hooks = {}
        for name in dir(self):
            method = getattr(self, name)
            if callable(method) and hasattr(method, "_pipeline_hooks"):
                for event, priority in method._pipeline_hooks:
                    if event not in self._hooks:
                        self._hooks[event] = []
                    self._hooks[event].append((priority, method))

        # Sort hooks by priority
        for event in self._hooks:
            self._hooks[event].sort(key=lambda x: x[0])

    def _fire(self, event: PipelineEvent, *args: Any, **kwargs: Any) -> None:
        """Call all hooks for an event in priority order."""
        hook_list = self._hooks.get(event.value, [])
        for priority, hook_func in hook_list:
            hook_func(*args, **kwargs)

    def run(self, **inputs) -> Optional[Dict[str, Any]]:
        """Execute sub-steps sequentially and emit events."""
        # Pipeline doesn't return outputs directly - sub-steps mutate state through __call__
        return None

    def __call__(self, state: State, **kwargs) -> None:
        """Execute sub-steps sequentially and emit events."""
        self._fire(PipelineEvent.PIPELINE_START, state)
        try:
            for idx, step in enumerate(self.steps):
                self._fire(PipelineEvent.BEFORE_STEP, step, state, idx)
                try:
                    step(state, **kwargs)
                except Exception as e:
                    self._fire(PipelineEvent.ON_ERROR, state, e, step, idx)
                    raise
                else:
                    self._fire(PipelineEvent.AFTER_STEP, step, state, idx)
        finally:
            self._fire(PipelineEvent.PIPELINE_END, state)

    def add_step(self, step: Step) -> None:
        """Append a sub-step."""
        self.steps.append(step)

    def extend(self, steps: Sequence[Step]) -> None:
        """Append multiple sub-steps."""
        self.steps.extend(steps)

    def __repr__(self) -> str:
        names = [s.name for s in self.steps]
        return f"{self.__class__.__name__}(steps={names!r}, precon={len(self.requires)}, postcon={len(self.provides)})"


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

    @Pipeline.hook(PipelineEvent.PIPELINE_START)
    def start(self, state: State) -> None:
        self.bus.publish(
            Event(PipelineEvent.PIPELINE_START, self.name, state=self._ctx(state))
        )

    @Pipeline.hook(PipelineEvent.PIPELINE_END)
    def end(self, state: State) -> None:
        self.bus.publish(
            Event(PipelineEvent.PIPELINE_END, self.name, state=self._ctx(state))
        )

    @Pipeline.hook(PipelineEvent.BEFORE_STEP)
    def before_step(self, step: Step, state: State, index: int) -> None:
        self.bus.publish(
            Event(
                PipelineEvent.BEFORE_STEP,
                self._label(step),
                constraints={
                    "index": index,
                    "requires": list(step.requires.keys()),
                    "provides": list(step.provides.keys()),
                },
                state=self._ctx(state),
            )
        )

    @Pipeline.hook(PipelineEvent.AFTER_STEP)
    def after_step(self, step: Step, state: State, index: int) -> None:
        self.bus.publish(
            Event(
                PipelineEvent.AFTER_STEP,
                self._label(step),
                constraints={
                    "index": index,
                    "requires": list(step.requires.keys()),
                    "provides": list(step.provides.keys()),
                },
                state=self._ctx(state),
            )
        )

    @Pipeline.hook(PipelineEvent.ON_ERROR)
    def on_error(self, state: State, exc: Exception, step: Step, index: int) -> None:
        self.bus.publish(
            Event(
                PipelineEvent.ON_ERROR,
                self._label(step),
                constraints={"index": index, "error": repr(exc)},
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
        self.refit = refit
        self._is_fitted = False
        super().__init__(*args, **kwargs)

    @Pipeline.hook(PipelineEvent.PIPELINE_START, priority=1)
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
