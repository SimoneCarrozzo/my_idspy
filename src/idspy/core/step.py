from abc import ABC, abstractmethod
# modificato da
# from typing import Iterable, Optional, Set, Any, override, Callable
# A:
from typing import Iterable, Optional, Set, Any, Callable
# aggiunto per me
from typing_extensions import override

from .state import State, StatePredicate

#tale classe rappresenta un singolo passo / trasformazione in una pipeline di elaborazione dati.
#Ogni passo può avere requisiti (requires) e fornire risultati (provides) che vengono gestiti tramite uno stato condiviso (State).
#La classe supporta anche funzionalità avanzate come l'esecuzione condizionale, il fitting di modelli e l'esecuzione ripetuta di passi.
class Step(ABC):
    """Abstract pipeline step."""

    def __init__(
        self,
        name: Optional[str] = None,
        requires: Optional[Iterable[str]] = None,
        provides: Optional[Iterable[str]] = None,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self.requires: Set[str] = set(requires or [])
        self.provides: Set[str] = set(provides or [])

    def check(self, state: State, constraints: Iterable[str]) -> None:
        """Raise if constraints keys are missing."""
        missing = [k for k in constraints if k not in state]
        if missing:
            raise KeyError(f"{self.name}: missing {missing}")

    @abstractmethod
    def run(self, state: State) -> None:
        """Mutate state and/or call services."""
        ...
    #tale metodo esegue un controllo sugli input richiesti (requires) prima di eseguire il passo
    #e un controllo sugli output forniti (provides) dopo l'esecuzione così da garantire che il passo rispetti i suoi contratti.
    def __call__(self, state: State) -> None:
        """Validate inputs, run, validate outputs."""
        self.check(state, self.requires)
        self.run(state)
        self.check(state, self.provides)

    def execute(self, state: State) -> None:
        """Alias for __call__."""
        self(state)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, requires={len(self.requires)}, provides={len(self.provides)}"
        )


class ConditionalStep(Step, ABC):
    """Step that runs only if a condition holds."""

    @abstractmethod
    def should_run(self, state: State) -> bool:
        """Return True to run, False to skip."""
        ...

    def on_skip(self, state: State) -> None:
        """Called if the step is skipped."""
        pass

    @override
    def __call__(self, state: State) -> None:
        if not self.should_run(state):
            self.on_skip(state)
            return
        super().__call__(state)

#tale classe estende Step per rappresentare un passo che richiede un processo di fitting prima di essere eseguito.
#Introduce un metodo astratto fit_impl che deve essere implementato dalle sottoclassi per definire il comportamento di fitting specifico.
#La classe tiene traccia dello stato di fitting tramite la proprietà is_fitted e garantisce che il passo non possa essere eseguito se non è stato fittato.
#(per fitting s'intende il processo di addestramento o adattamento di un modello o di una trasformazione ai dati contenuti nello stato)
class FitAwareStep(Step, ABC):
    """Step that must be fitted before running."""

    def __init__(
        self,
        name: Optional[str] = None,
        requires: Optional[Iterable[str]] = None,
        provides: Optional[Iterable[str]] = None,
    ) -> None:
        self._is_fitted: bool = False

        super().__init__(
            name=name,
            requires=requires,
            provides=provides,
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the step is fitted."""
        return self._is_fitted

    @abstractmethod
    def fit_impl(self, state: State) -> None:
        """Subclass hook: fit/precompute resources."""
        ...

    def fit(self, state: State) -> None:
        """Validate inputs and fit."""
        self.check(state, self.requires)
        self.fit_impl(state)
        self._is_fitted = True

    def __call__(self, state: State) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name!r} is not fitted.")
        super().__call__(state)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"{base}, fitted={self._is_fitted})"


class Repeat(Step):
    """Repeat a step multiple times or until a predicate returns True (stop)."""

    def __init__(
        self,
        step: Step,
        count: int,
        predicate: Optional[StatePredicate] = None,
        name: Optional[str] = None,
    ) -> None:

        if count <= 0:
            raise ValueError("count must be > 0")

        self.step = step
        self.count = count
        self.predicate = predicate

        super().__init__(
            name=name or f"repeated({step.name})",
            requires=set(step.requires),
            provides=set(step.provides),
        )

    def run(self, state: State) -> None:
        for _ in range(self.count):
            self.step(state)

            if self.predicate and self.predicate(state):
                break

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"repeated({base}, count={self.count}, predicate={self.predicate})"


class ContextualStep(Step, ABC):
    """Step that runs within a context manager."""

    def __init__(
        self,
        step: Step,
        target: str = "context",
        name: Optional[str] = None,
    ) -> None:
        self.step = step
        self.target = target

        super().__init__(
            name=name or f"contextual({step.name})",
            requires=set(step.requires),
            provides=set(step.provides + [self.target]),
        )

    @abstractmethod
    def context(self, state: State) -> Any:
        """Return a context manager."""
        ...

    def run(self, state: State) -> None:
        with self.context(state) as ctx:
            state[self.target] = ctx
            self.step(state)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"contextual({base})"
