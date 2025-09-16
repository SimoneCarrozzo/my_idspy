from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Any, Tuple, override
from dataclasses import dataclass

from .state import State, StatePredicate


@dataclass
class TypeConstraint:
    """Constraint that a state key must have a specific type."""

    key: str
    expected_type: type

    def check(self, state: State) -> bool:
        """Check if the constraint is satisfied."""
        return state.check_type(self.key, self.expected_type)

    def get_value(self, state: State) -> Any:
        """Get the value, ensuring it has the correct type."""
        return state.get_typed(self.key, self.expected_type)


Constraints = Dict[str, TypeConstraint]


class Step(ABC):
    """Abstract pipeline step."""

    def __init__(
        self,
        name: Optional[str] = None,
        precon: Optional[Constraints] = None,
        postcon: Optional[Constraints] = None,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self.precon = precon or {}
        self.postcon = postcon or {}

    def check_constraints(self, state: State, constraints: Constraints) -> None:
        """Raise if constraints are not satisfied."""
        missing = [
            name
            for name, constraint in constraints.items()
            if not constraint.check(state)
        ]

        if missing:
            raise KeyError(f"{self.name}: constraints not satisfied: {missing}")

    def get_inputs(self, state: State) -> Dict[str, Any]:
        """Extract and validate required inputs from state."""
        inputs = {}
        for name, constraint in self.precon.items():
            inputs[name] = constraint.get_value(state)
        return inputs

    @abstractmethod
    def run(self, state: State, **kwargs) -> None:
        """Mutate state and/or call services."""
        ...

    def __call__(self, state: State, **kwargs) -> None:
        """Check preconditions, run, validate postconditions."""
        self.check_constraints(state, self.precon)
        self.run(state, **kwargs)
        self.check_constraints(state, self.postcon)

    def execute(self, state: State, **kwargs) -> None:
        """Alias for __call__."""
        self(state, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, precon={len(self.precon)}, postcon={len(self.postcon)})"
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
    def __call__(self, state: State, **kwargs) -> None:
        if not self.should_run(state):
            self.on_skip(state)
            return
        super().__call__(state, **kwargs)


class FitAwareStep(Step, ABC):
    """Step that must be fitted before running."""

    def __init__(
        self,
        name: Optional[str] = None,
        precon: Optional[Constraints] = None,
        postcon: Optional[Constraints] = None,
    ) -> None:
        self._is_fitted: bool = False

        super().__init__(
            name=name,
            precon=precon,
            postcon=postcon,
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
        self.check_constraints(state, self.precon)
        self.fit_impl(state)
        self._is_fitted = True

    def __call__(self, state: State, **kwargs) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name!r} is not fitted.")
        super().__call__(state, **kwargs)

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
            precon=step.precon,
            postcon=step.postcon,
        )

    def run(self, state: State, **kwargs) -> None:
        for _ in range(self.count):
            self.step(state, **kwargs)

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
        name: Optional[str] = None,
    ) -> None:
        self.step = step

        super().__init__(
            name=name or f"contextual({step.name})",
            precon=step.precon,
            postcon=step.postcon,
        )

    @abstractmethod
    def context(self, state: State) -> Any:
        """Return a context manager."""
        ...

    def run(self, state: State, **kwargs) -> None:
        with self.context(state) as ctx:
            # Pass context directly as kwarg to the inner step
            kwargs["context"] = ctx
            self.step(state, **kwargs)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"contextual({base})"
