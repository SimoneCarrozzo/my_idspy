from typing import Sequence, Optional, Iterable, List, Any

from .state import State
from .step import Step, FittedStep
from ..events.bus import EventBus
from ..events.events import Event


class Pipeline(Step):
    """
    A Step that runs a sequence of sub-steps with lifecycle hooks.
    """

    def __init__(
            self,
            steps: Sequence[Step],
            name: Optional[str] = None,
            requires: Optional[Iterable[str]] = None,
            produces: Optional[Iterable[str]] = None,
            **_: Any,
    ):
        super().__init__(name=name, requires=requires, produces=produces)
        self.steps: List[Step] = list(steps)

    def _run(self, state: State) -> None:
        self.on_pipeline_start(state)
        try:
            for idx, step in enumerate(self.steps):
                self.before_step(step, state, index=idx)
                try:
                    step.run(state)
                except Exception as e:
                    self.on_error(state, e, step=step, index=idx)
                    raise
                else:
                    self.after_step(step, state, index=idx)
        finally:
            self.on_pipeline_end(state)

    # Hooks (no-op by default)
    def on_pipeline_start(self, state: State) -> None:
        ...

    def on_pipeline_end(self, state: State) -> None:
        ...

    def before_step(self, step: Step, state: State, *, index: int) -> None:
        ...

    def after_step(self, step: Step, state: State, *, index: int) -> None:
        ...

    def on_error(
            self,
            state: State,
            e: Exception,
            step: Optional[Step],
            index: Optional[int],
    ) -> None:
        ...

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def extend(self, steps: Sequence[Step]) -> None:
        self.steps.extend(steps)

    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"steps={step_names!r}, requires={self.requires!r}, provides={self.provides!r})"
        )


class InstrumentationMixin:
    """
    Mixin that publishes pipeline lifecycle events to an EventBus.
    Put this BEFORE Pipeline/FittedPipeline in the bases so its hooks take effect.
    """

    def __init__(self, *args: Any, bus: Optional[EventBus] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.bus: EventBus = bus or EventBus()

    # Hook implementations
    def on_pipeline_start(self, state: State) -> None:
        self.bus.publish(
            Event.pipeline_start(getattr(self, 'name', 'pipeline'), state.get_view())
        )

    def on_pipeline_end(self, state: State) -> None:
        self.bus.publish(
            Event.pipeline_end(getattr(self, 'name', 'pipeline'), state.get_view())
        )

    def before_step(self, step: Step, state: State, *, index: int) -> None:
        self.bus.publish(
            Event.step_start(
                f"{getattr(self, 'name', 'pipeline')}.{step.name}",
                state.get_view(),
                index=index,
                requires=list(step.requires),
                provides=list(step.provides),
            )
        )

    def after_step(self, step: Step, state: State, *, index: int) -> None:
        self.bus.publish(
            Event.step_end(
                f"{getattr(self, 'name', 'pipeline')}.{step.name}",
                state.get_view(),
                index=index,
                requires=list(step.requires),
                provides=list(step.provides),
            )
        )

    def on_error(
            self,
            state: State,
            e: Exception,
            step: Optional[Step],
            index: Optional[int],
    ) -> None:
        step_label = (
            f"{getattr(self, 'name', 'pipeline')}.{step.name}"
            if step is not None else getattr(self, 'name', 'pipeline')
        )
        self.bus.publish(
            Event.step_error(
                step_label,
                state.get_view(),
                index=index,
                error=repr(e),
            )
        )


class InstrumentedPipeline(InstrumentationMixin, Pipeline):
    """
    A regular Pipeline with instrumentation.
    Order matters: mixin goes first so its hooks override Pipeline's.
    """
    pass


class FittedPipeline(Pipeline):
    """
    Pipeline that requires its fitted sub-steps to be trained before running.
    """

    def __init__(
            self,
            steps: Sequence[Step],
            name: Optional[str] = None,
            requires: Optional[Iterable[str]] = None,
            produces: Optional[Iterable[str]] = None,
            **kwargs: Any,
    ):
        super().__init__(steps, name=name, requires=requires, produces=produces, **kwargs)
        self._is_fitted: bool = False

    def fit(self, state: State) -> None:
        for step in self.steps:
            if isinstance(step, FittedStep):
                step.fit(state)
        self._is_fitted = True

    def _run(self, state: State) -> None:
        if not self._is_fitted:
            self.fit(state)
        super()._run(state)


class FittedInstrumentedPipeline(InstrumentationMixin, FittedPipeline):
    """
    A FittedPipeline that also publishes events.
    """
    pass
