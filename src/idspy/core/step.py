import inspect
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, Optional, Type

from .state import State, StatePredicate


class Step(ABC):
    """Abstract base class for pipeline steps with decorator-based state management.

    Rules:
    - The state can be scoped to a prefix (e.g., step name) to avoid key collisions.
    - Requires are set to strict=False when scoped, allowing fallback to global keys.
    - Provides are set to strict=True when scoped, limiting access to only the scoped keys.
    - Abstract methods must be implemented by subclasses and must be decorated for automatic state handling.
    """

    def __init__(self, name: Optional[str] = None, scope: Optional[str] = None) -> None:
        """Initialize a pipeline step. The scope is used to isolate state keys."""
        self.name = name or self.__class__.__name__
        self.scope = scope

    @classmethod
    def requires(cls, **requirements: Type[Any]):
        """Automatically injects required parameters from state or kwargs.
        Rules:
        - Parameter resolution order: explicit kwargs > scoped state > method defaults.
        - The state can be scoped to a prefix (e.g., step name) to avoid key collisions.
        - Inputs are set to strict=False when scoped, allowing fallback to global keys.
        """

        def decorator(func):
            sig = inspect.signature(func)

            @wraps(func)
            def wrapper(self, state: State, **kwargs):
                view = state.view(self.scope, strict=False) if self.scope else state

                filled: Dict[str, Any] = {}

                for key, typ in requirements.items():
                    if key in kwargs:
                        # explicit value passed by caller
                        val = kwargs.pop(key)
                        filled[key] = val
                        continue

                    # try to read from scoped state
                    try:
                        val = view.get(key, typ)
                        filled[key] = val
                        continue
                    except KeyError:
                        pass  # look for default in signature
                    except TypeError as e:
                        # type in state doesn't conform
                        raise

                    # default in method signature, if present
                    param = sig.parameters.get(key)
                    if param is not None and param.default is not inspect._empty:
                        default_val = param.default
                        filled[key] = default_val
                    else:
                        raise KeyError(
                            f"Missing required state/key '{key}' for step '{self.name}'"
                        )

                kwargs.update(filled)
                return func(self, state, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def provides(cls, **outputs: Type[Any]):
        """Automatically validates and saves method outputs to state with type checking.

        Rules:
        - The decorated method must return a dict containing all declared outputs.
        - The state can be scoped to a prefix (e.g., step name) to avoid key collisions.
        - Each output key must be unique within the step's scope.
        - Outputs are set to strict=True when scoped, limiting access to only the scoped keys.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(self, state: State, **kwargs):
                result = func(self, state, **kwargs)

                if result is None:
                    return None

                if not isinstance(result, dict):
                    raise TypeError(
                        f"Step '{self.name}' must return a dict (or None) when using "
                        f"@Step.provides, got {type(result).__name__}"
                    )

                view = state.view(self.scope, strict=True) if self.scope else state

                # validate and persist each declared key
                for key, typ in outputs.items():
                    if key not in result:
                        raise KeyError(
                            f"Returned dict from step '{self.name}' is missing key '{key}'"
                        )
                    value = result[key]
                    view.set(key, value, typ)

                return result

            return wrapper

        return decorator

    @classmethod
    def repeat(
        cls, count: Optional[int] = 1000, predicate: Optional[StatePredicate] = None
    ):
        """Repeat the decorated method multiple times or until a predicate returns True.

        Args:
            count: Number of times to repeat (default: 1000)
            predicate: Optional predicate function to check for early stopping

        Rules:
        - The method will be repeated up to 'count' times
        - If predicate is provided and returns True, repetition stops early
        - Both count and predicate can be None for single execution
        - Each decorator below is re-applied at each iteration
        """

        def decorator(func):
            @wraps(func)
            def wrapper(self, state: State, **kwargs):
                final_count = count if count is not None else 1

                if final_count <= 0:
                    raise ValueError("count must be > 0")

                for _ in range(final_count):
                    result = func(self, state, **kwargs)

                    if predicate and predicate(state):
                        break

                return result

            return wrapper

        return decorator

    @abstractmethod
    def run(self, state: State, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the step's main logic.

        This method should be implemented by subclasses and can be decorated
        with @Step.requires and @Step.provides for automatic parameter/output handling.

        Args:
            state: Current pipeline state
            **kwargs: Additional parameters

        Returns:
            Optional dictionary of outputs to be saved to state
        """
        ...

    def __call__(self, state: State, **kwargs) -> None:
        """Execute the step by calling its run method."""
        self.run(state, **kwargs)

    def __repr__(self) -> str:
        """Return string representation of the step."""
        return f"{self.__class__.__name__}(name={self.name!r}, scope={self.scope!r})"


class ConditionalStep(Step, ABC):
    """Step that runs only if a condition holds."""

    @abstractmethod
    def should_run(self, state: State, **kwargs) -> bool:
        """Return True to run, False to skip."""
        ...

    @abstractmethod
    def on_skip(self, state: State, **kwargs) -> None:
        """Called if the step is skipped."""
        pass

    def __call__(self, state: State, **kwargs) -> None:
        if not self.should_run(state, **kwargs):
            self.on_skip(state, **kwargs)
            return
        super().__call__(state, **kwargs)


class FitAwareStep(Step, ABC):
    """Step that must be fitted before running."""

    def __init__(self, name: Optional[str] = None, **kwargs) -> None:
        self._is_fitted: bool = False
        super().__init__(name=name, **kwargs)

    @property
    def is_fitted(self) -> bool:
        """Whether the step is fitted."""
        return self._is_fitted

    @abstractmethod
    def fit_impl(self, state: State, **kwargs) -> None:
        """Subclass hook: fit/precompute resources using inputs."""
        ...

    def fit(self, state: State, **kwargs) -> None:
        """Validate inputs and fit."""
        self.fit_impl(state, **kwargs)
        self._is_fitted = True

    def __call__(self, state: State, **kwargs) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name!r} is not fitted.")
        super().__call__(state, **kwargs)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"{base}, fitted={self._is_fitted})"


class ContextualStep(Step, ABC):
    """Step that runs within a context manager."""

    def __init__(self, step: Step, **kwargs) -> None:
        self.step = step
        super().__init__(name=f"contextual({step.name})", scope=step.scope, **kwargs)

    @abstractmethod
    def context(self, state: State) -> Any:
        """Return a context manager."""
        ...

    def run(self, state: State, **kwargs) -> None:
        """Execute the step within the context manager. Provides 'context' kwarg."""
        with self.context(state) as ctx:
            kwargs["context"] = ctx
            self.step(state, **kwargs)

    def __call__(self, state: State, **kwargs) -> None:
        """Override to handle context management."""
        self.run(state, **kwargs)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"contextual({base})"
