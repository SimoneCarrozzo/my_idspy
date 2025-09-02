from abc import ABC, abstractmethod
from typing import Iterable, Optional, Set, override  # Python 3.12+: override in typing

from .state import State


class Step(ABC):
    """
    Abstract base class for all steps.

    A step is an independent routine that can read/modify the state and call services.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            requires: Optional[Iterable[str]] = None,
            provides: Optional[Iterable[str]] = None,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self.requires: Set[str] = set(requires or [])
        self.provides: Set[str] = set(provides or [])

    def check_requires(self, state: State) -> None:
        """Raise if any required keys are missing from the state."""
        missing = [k for k in self.requires if k not in state]
        if missing:
            raise KeyError(f"{self.name}: missing dependencies in State: {missing}")

    @abstractmethod
    def run(self, state: State) -> None:
        """Core of the step: mutate state and/or call services."""
        ...

    def execute(self, state: State) -> None:
        """Validate requirements, then call :meth:`run`."""
        self.check_requires(state)
        self.run(state)

    def __call__(self, state: State) -> None:
        """Alias for :meth:`execute`."""
        self.execute(state)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, requires={self.requires!r}, provides={self.provides!r})"
        )


class ConditionalStep(Step):
    """A step that can be skipped dynamically based on the current state."""

    @abstractmethod
    def should_run(self, state: State) -> bool:
        """Return True to execute, False to skip."""
        ...

    @override
    def execute(self, state: State) -> None:
        if self.should_run(state):
            super().execute(state)


class FitAwareStep(Step):
    """
    A step that must be 'fit' before it can be run (e.g., train a model,
    precompute resources, warm caches, etc.).
    """

    def __init__(
            self,
            name: Optional[str] = None,
            requires: Optional[Iterable[str]] = None,
            provides: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(name=name, requires=requires, provides=provides)
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the step has been successfully fitted."""
        return self._is_fitted

    @abstractmethod
    def fit_core(self, state: State) -> None:
        """Subclass hook: do the work necessary to make this step runnable."""
        ...

    def fit(self, state: State) -> None:
        """
        Fit the step.

        We validate requirements too, since fitting may depend on the same inputs as running.
        """
        self.check_requires(state)
        self.fit_core(state)
        self._is_fitted = True

    @override
    def execute(self, state: State) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"Step {self.name!r} has not been fitted yet.")
        super().execute(state)
