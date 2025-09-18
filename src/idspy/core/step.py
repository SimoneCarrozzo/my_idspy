from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from .state import State, StatePredicate


class Step(ABC):
    """Abstract pipeline step with decorator-based configuration."""

    @staticmethod
    def requires(**inputs: type):
        """Decorator to specify input requirements."""

        def decorator(cls):
            cls._required_inputs = inputs
            return cls

        return decorator

    @staticmethod
    def provides(**outputs: type):
        """Decorator to specify output guarantees."""

        def decorator(cls):
            cls._provided_outputs = outputs
            return cls

        return decorator

    def __init__(
        self, name: Optional[str] = None, scope_prefix: Optional[str] = None
    ) -> None:
        self.name = name or self.__class__.__name__
        self.scope_prefix = scope_prefix

        # Get requirements from decorators
        self.requires = getattr(self.__class__, "_required_inputs", {})
        self.provides = getattr(self.__class__, "_provided_outputs", {})

    def get_inputs(self, state: State) -> Dict[str, Any]:
        """Extract and validate required inputs from state."""
        state = state.scope(self.scope_prefix) if self.scope_prefix else state
        inputs = {}
        for name, expected_type in self.requires.items():
            # Use same name for state key by default
            key = name

            value = state.get_typed(key, expected_type)
            inputs[name] = value
        return inputs

    def validate_outputs(self, outputs: Dict[str, Any]) -> None:
        """Validate that outputs match declared types."""
        if not outputs:
            return

        for name, expected_type in self.provides.items():
            if name not in outputs:
                raise KeyError(f"{self.name}: missing required output '{name}'")

            value = outputs[name]
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{self.name}: expected {expected_type.__name__} for output '{name}', got {type(value).__name__}"
                )

    def update_state(self, state: State, outputs: Dict[str, Any]) -> None:
        """Update state with validated outputs."""
        if outputs:
            self.validate_outputs(outputs)

            state = state.scope(self.scope_prefix) if self.scope_prefix else state
            state.update(outputs)

    @abstractmethod
    def run(self, **inputs) -> Optional[Dict[str, Any]]:
        """Process inputs and return outputs. No direct state access."""
        ...

    def __call__(self, state: State, **kwargs) -> None:
        """Extract inputs, run step, update state with outputs."""
        # Get validated inputs
        inputs = self.get_inputs(state)
        inputs.update(kwargs)

        outputs = self.run(**inputs)

        # Update state with outputs
        self.update_state(state, outputs or {})

    def execute(self, state: State, **kwargs) -> None:
        """Alias for __call__."""
        self(state, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, precon={len(self.requires)}, postcon={len(self.provides)})"
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

    def __call__(self, state: State, **kwargs) -> None:
        if not self.should_run(state):
            self.on_skip(state)
            return
        super().__call__(state, **kwargs)


class FitAwareStep(Step, ABC):
    """Step that must be fitted before running."""

    def __init__(
        self, name: Optional[str] = None, scope_prefix: Optional[str] = None
    ) -> None:
        self._is_fitted: bool = False
        super().__init__(name=name, scope_prefix=scope_prefix)

    @property
    def is_fitted(self) -> bool:
        """Whether the step is fitted."""
        return self._is_fitted

    @abstractmethod
    def fit_impl(self, **inputs) -> None:
        """Subclass hook: fit/precompute resources using inputs."""
        ...

    def fit(self, state: State) -> None:
        """Validate inputs and fit."""
        inputs = self.get_inputs(state)
        self.fit_impl(**inputs)
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

    @staticmethod
    def _count(value: int):
        """Decorator to specify repeat count."""

        def decorator(cls):
            cls._repeat_count = value
            return cls

        return decorator

    @staticmethod
    def _predicate(predicate_fn: StatePredicate):
        """Decorator to specify stop predicate."""

        def decorator(cls):
            cls._repeat_predicate = predicate_fn
            return cls

        return decorator

    def __init__(
        self,
        step: Step,
        count: Optional[int] = None,
        predicate: Optional[StatePredicate] = None,
        name: Optional[str] = None,
    ) -> None:
        # Use decorator values as defaults if not provided
        final_count = (
            count if count is not None else getattr(self.__class__, "_repeat_count", 1)
        )
        final_predicate = (
            predicate
            if predicate is not None
            else getattr(self.__class__, "_repeat_predicate", None)
        )

        if final_count <= 0:
            raise ValueError("count must be > 0")

        self.step = step
        self._count = final_count
        self._predicate = final_predicate

        super().__init__(name=name or f"repeated({step.name})")

        # Inherit constraints from wrapped step
        self.requires = step.requires
        self.provides = step.provides

    def run(self, **inputs) -> Optional[Dict[str, Any]]:
        """This shouldn't be called directly - we override __call__ instead."""
        return None

    def __call__(self, state: State, **kwargs) -> None:
        """Override to handle repetition logic."""
        for _ in range(self._count):
            self.step(state, **kwargs)

            if self._predicate and self._predicate(state):
                break

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"repeated({base}, count={self._count}, predicate={self._predicate})"


class ContextualStep(Step, ABC):
    """Step that runs within a context manager."""

    def __init__(
        self, step: Step, name: Optional[str] = None, scope_prefix: Optional[str] = None
    ) -> None:
        self.step = step
        super().__init__(
            name=name or f"contextual({step.name})", scope_prefix=scope_prefix
        )

        # Inherit constraints from wrapped step
        self.requires = step.requires
        self.provides = step.provides

    @abstractmethod
    def context(self, state: State) -> Any:
        """Return a context manager."""
        ...

    def run(self, **inputs) -> Optional[Dict[str, Any]]:
        """This shouldn't be called directly - we override __call__ instead."""
        return None

    def __call__(self, state: State, **kwargs) -> None:
        """Override to handle context management."""
        with self.context(state) as ctx:
            kwargs["context"] = ctx
            self.step(state, **kwargs)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"contextual({base})"
