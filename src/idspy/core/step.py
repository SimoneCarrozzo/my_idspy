from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Iterable, Set

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
            produces: Optional[Iterable[str]] = None,
    ):
        self.name: str = name or self.__class__.__name__
        self.requires: Set[str] = set(requires or [])
        self.provides: Set[str] = set(produces or [])

    def check_requires(self, state: State) -> None:
        """Raise if any required keys are missing from the state."""
        missing = [k for k in self.requires if k not in state]
        if missing:
            raise KeyError(f"{self.name}: missing dependencies in State: {missing}")

    @abstractmethod
    def _run(self, state: State) -> None:
        """Core of the step: mutate state and/or call services."""
        ...

    def run(self, state: State) -> None:
        """Validate requirements, then execute the step."""
        self.check_requires(state)
        self._run(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, requires={self.requires!r}, provides={self.provides!r})"


class ConditionalStep(Step, ABC):
    """A step that can be skipped dynamically based on the current state."""

    @abstractmethod
    def should_run(self, state: State) -> bool:
        """Return True to execute, False to skip."""
        ...

    def run(self, state: State) -> None:
        if self.should_run(state):
            # Call Step.run, not self.run, to avoid recursion.
            super().run(state)


class FittedStep(Step, ABC):
    """
    A step that must be 'fit' before it can be run (e.g., train a model,
    precompute resources, warm caches, etc.).
    """

    def __init__(
            self,
            name: Optional[str] = None,
            requires: Optional[Iterable[str]] = None,
            produces: Optional[Iterable[str]] = None,
    ):
        super().__init__(name=name, requires=requires, produces=produces)
        self._is_fitted: bool = False

    @abstractmethod
    def _fit(self, state: State) -> None:
        """Do the work necessary to make this step runnable."""
        ...

    def fit(self, state: State) -> None:
        """
        Fit the step. We validate requirements too, since fitting may
        depend on the same inputs as running.
        """
        self.check_requires(state)
        self._fit(state)
        self._is_fitted = True

    def run(self, state: State) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"Step {self.name!r} has not been fitted yet.")

        super().run(state)
