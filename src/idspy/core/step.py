from abc import ABC, abstractmethod
from typing import Iterable, Optional, Set, Any, override, Callable

from .state import State


class Step(ABC):
    """Abstract pipeline step."""

    def __init__(
            self,
            name: Optional[str] = None,
            requires: Optional[Iterable[str]] = None,
            provides: Optional[Iterable[str]] = None,
            validate_outputs: bool = True,
            require_change: bool = False,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self.requires: Set[str] = set(requires or [])
        self.provides: Set[str] = set(provides or [])
        self.validate_outputs = validate_outputs
        self.require_change = require_change

    def check_requires(self, state: State) -> None:
        """Raise if required keys are missing."""
        missing = [k for k in self.requires if k not in state]
        if missing:
            raise KeyError(f"{self.name}: missing {missing}")

    def _snapshot(self, state: State) -> dict[str, Any]:
        if not self.provides:
            return {}
        return {k: state[k] for k in self.provides if k in state}

    def check_provides(
            self,
            state: State,
            before: Optional[dict[str, Any]] = None,
    ) -> None:
        """Raise if declared outputs are not present (or unchanged if required)."""
        missing = [k for k in self.provides if k not in state]
        if missing:
            raise KeyError(f"{self.name}: missing provided keys {missing}")

        if self.require_change and before is not None:
            unchanged = []
            for k in self.provides:
                if k in before and state[k] is before[k]:
                    unchanged.append(k)
                elif k in before and state[k] == before[k]:
                    unchanged.append(k)
            if unchanged:
                raise ValueError(f"{self.name}: outputs not changed {unchanged}")

    @abstractmethod
    def run(self, state: State) -> None:
        """Mutate state and/or call services."""
        ...

    def __call__(self, state: State) -> None:
        """Validate inputs, run, validate outputs."""
        self.check_requires(state)
        before = self._snapshot(state) if (self.validate_outputs and self.require_change) else None
        self.run(state)
        if self.validate_outputs:
            self.check_provides(state, before)

    def execute(self, state: State) -> None:
        """Alias for __call__."""
        self(state)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, requires={len(self.requires)}, provides={len(self.provides)}, "
            f"validate_outputs={self.validate_outputs}, require_change={self.require_change})"
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


class FitAwareStep(Step, ABC):
    """Step that must be fitted before running."""

    def __init__(
            self,
            name: Optional[str] = None,
            requires: Optional[Iterable[str]] = None,
            provides: Optional[Iterable[str]] = None,
            validate_outputs: bool = True,
            require_change: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            requires=requires,
            provides=provides,
            validate_outputs=validate_outputs,
            require_change=require_change,
        )
        self._is_fitted: bool = False

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
        self.check_requires(state)
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
            *,
            count: Optional[int] = None,  # fixed iterations
            until: Optional[Callable[[State, int], bool]] = None,  # True => stop
            max_iters: Optional[int] = None,  # safety cap for until-only
            name: Optional[str] = None,
    ) -> None:
        if count is None and until is None:
            raise ValueError("either count or until must be provided")
        if count is not None and count <= 0:
            raise ValueError("count must be > 0")
        super().__init__(
            name=name or f"Repeat({step.name})",
            requires=set(step.requires),
            provides=set(step.provides),
        )
        self.step = step
        self.count = count
        self.until = until
        self.max_iters = max_iters or (10_000 if (count is None and until is not None) else None)

    def run(self, state: State) -> None:
        i = 0
        while True:
            self.step(state)
            i += 1

            # stop-on-predicate
            if self.until and self.until(state, i):
                break

            # stop-on-count
            if self.count is not None and i >= self.count:
                break

            # safety cap
            if self.max_iters is not None and i >= self.max_iters:
                raise RuntimeError(
                    f"{self.name}: exceeded max_iters={self.max_iters} without meeting stop condition"
                )

    def __repr__(self) -> str:
        mode = []
        if self.count is not None: mode.append(f"count={self.count}")
        if self.until is not None: mode.append("until")
        if self.max_iters is not None: mode.append(f"max_iters={self.max_iters}")
        return f"{self.__class__.__name__}(step={self.step.name!r}, " + ", ".join(mode) + ")"
