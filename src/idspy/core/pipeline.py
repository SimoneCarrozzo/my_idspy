from enum import Enum
from types import MappingProxyType
from typing import (
    Sequence,
    Optional,
    List,
    Any,
    Callable,
    Union,
    Mapping,
)

from .state import State
from .step import Step, FitAwareStep
from ..events.bus import EventBus
from ..events.events import Event


class PipelineEvent(str, Enum):     #ogni pipeline ha un ciclo di vita con eventi specifici
    """Pipeline lifecycle and step events."""

    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    BEFORE_STEP = "before_step"  # args: (step, state, index=int)   --| questi due servono per il 
    AFTER_STEP = "after_step"  # args: (step, state, index=int)     --| tracciamento e debug degli step
    ON_ERROR = "on_error"  # args: (state, exc, step=..., index=int)


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

    def run(self, state: State, **kwargs) -> None:
        """Execute sub-steps sequentially and emit events."""
        self._fire(PipelineEvent.PIPELINE_START, state) #avvia la pipeline
        try:    #per ogni step, viene notificato il passo iniziale/precedente 
            for idx, step in enumerate(self.steps):
                self._fire(
                    PipelineEvent.BEFORE_STEP,
                    step,
                    state.view(self.in_scope, strict=False),
                    idx,
                )
                try:
                    step(state, **kwargs)
                except Exception as e:
                    self._fire(PipelineEvent.ON_ERROR, state, e, step, idx)
                    raise
                else:
                    self._fire(
                        PipelineEvent.AFTER_STEP,
                        step,
                        state.view(self.out_scope, strict=True),
                        idx,
                    )
        finally:
            self._fire(PipelineEvent.PIPELINE_END, state) #termina la pipeline

    def __call__(self, state: State, **kwargs) -> None:
        self.run(state, **kwargs)

    def add_step(self, step: Step) -> None:
        """Append a sub-step."""
        self.steps.append(step)

    def extend(self, steps: Sequence[Step]) -> None:
        """Append multiple sub-steps."""
        self.steps.extend(steps)

    def __repr__(self) -> str:  #rappr testuale della pipeline per debugging, cioè il nome pipeline, i nomi degli step, il num di chiavi richieste e fornite
        names = [s.name for s in self.steps]
        return f"{self.__class__.__name__}(steps={names!r}, precon={len(self.requires)}, postcon={len(self.provides)})"

#questa pipeline, oltre che gestire gli hook come la Pipeline base, pubblica eventi su un EventBus, cioè un sistema di messaggistica
#che permette a diverse parti di un'applicazione di comunicare tra loro in modo asincrono.
#In questo modo, altre componenti dell'applicazione possono iscriversi a questi eventi e reagire di conseguenza.
#infatti, gli eventi vengongo pubblicati tramite @hook, che associa i metodi della pipeline agli eventi del ciclo di vita. 
#Inoltre vi è uno statichmetode che converte lo stato in una mappa di sola lettura, affinché gli ascoltatori degli eventi possano accedere allo stato senza modificarlo.
class ObservablePipeline(Pipeline):
    """Pipeline that publishes lifecycle events to an EventBus."""

    def __init__(
        self, steps: Sequence[Step], bus: Optional[EventBus] = None, **kwargs: Any
    ) -> None:
        super().__init__(steps, **kwargs)
        self.bus: EventBus = bus or EventBus()

    @staticmethod
    def _payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return MappingProxyType(payload)

    def _label(self, step: Optional[Step]) -> str:
        return f"{self.name}.{step.name}" if step is not None else self.name

    @Pipeline.hook(PipelineEvent.PIPELINE_START)
    def start(self, state: State) -> None:
        payload = self._payload(state.as_dict())
        self.bus.publish(
            Event(
                PipelineEvent.PIPELINE_START,
                self.name,
                payload=payload,
            )
        )

    @Pipeline.hook(PipelineEvent.PIPELINE_END)
    def end(self, state: State) -> None:
        payload = self._payload(state.as_dict())
        self.bus.publish(
            Event(
                PipelineEvent.PIPELINE_END,
                self.name,
                payload=payload,
            )
        )

    @Pipeline.hook(PipelineEvent.BEFORE_STEP)
    def before_step(self, step: Step, state: State, index: int) -> None:
        payload = state.as_dict()
        payload["step_index"] = index

        self.bus.publish(
            Event(
                PipelineEvent.BEFORE_STEP,
                self._label(step),
                payload=self._payload(payload),
            )
        )

    @Pipeline.hook(PipelineEvent.AFTER_STEP)
    def after_step(self, step: Step, state: State, index: int) -> None:
        payload = state.as_dict()
        payload["step_index"] = index

        self.bus.publish(
            Event(
                PipelineEvent.AFTER_STEP,
                self._label(step),
                payload=self._payload(payload),
            )
        )

    @Pipeline.hook(PipelineEvent.ON_ERROR)
    def on_error(self, state: State, exc: Exception, step: Step, index: int) -> None:
        payload = state.as_dict()
        payload["step_index"] = index
        payload["error"] = str(exc)

        self.bus.publish(
            Event(
                PipelineEvent.ON_ERROR,
                self._label(step),
                payload=self._payload(payload),
            )
        )

#tale classe estende la Pipeline base per includere funzionalità specifiche per il fitting di modelli.
#In particolare, gestisce gli step che sono istanze di FitAwareStep, che richiedono un processo di fitting prima di essere eseguiti.
#La classe introduce un parametro refit che determina se gli step devono essere rifittati ogni volta 
# che la pipeline viene eseguita o solo se non sono già stati fittati.
#Inoltre, tiene traccia dello stato di fitting della pipeline stessa tramite la proprietà is_fitted.
class FitAwarePipeline(Pipeline):
    """Pipeline that fits FitAwareStep sub-steps before running."""

    def __init__(
        self,
        steps: Sequence[Step],
        refit: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        refit: if False, fit only steps not yet fitted; if True, always fit.
        fit_priority: hook priority; lower runs earlier.
        """
        self.refit = refit
        self._is_fitted = False
        super().__init__(steps, **kwargs)

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

    def __init__(
        self,
        steps: Sequence[Step],
        bus: Optional[EventBus] = None,
        refit: bool = False,
        **kwargs: Any,
    ) -> None:
        # cooperative MRO: both parents call super().__init__
        super().__init__(steps, bus=bus, refit=refit, **kwargs)
