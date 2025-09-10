import pandas as pd

from ..utils import validate_instance
from ...core.state import State
from ...core.step import Step
from ...data.tab_accessor import reattach_meta


class DownsampleToMinority(Step):
    """Downsample each class to the size of the minority class."""

    def __init__(
        self,
        class_col: str,
        source: str = "data.root",
        target: str | None = None,
        name: str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.class_col = class_col
        self.source = source
        self.target = target or source
        self.random_state = random_state

        super().__init__(
            name=name or "downsample_to_minority",
            requires=[self.source],
            provides=[self.target],
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        # Early exits for edge cases
        if obj.empty or self.class_col not in obj.columns:
            state[self.target] = obj
            return

        counts = obj[self.class_col].value_counts(dropna=False)
        if counts.empty:
            state[self.target] = obj
            return

        minority = int(counts.min())
        if minority <= 0:
            sampled = obj.iloc[0:0]  # empty but keep schema
            state[self.target] = reattach_meta(obj, sampled)
            return

        # Efficient sampling: avoid full shuffle if not needed
        sampled = obj.groupby(
            self.class_col, dropna=False, group_keys=False, sort=False
        ).sample(n=minority, replace=False, random_state=self.random_state)

        state[self.target] = reattach_meta(obj, sampled)


class Downsample(Step):
    """Downsample rows globally or per class."""

    def __init__(
        self,
        frac: float,
        class_col: str | None = None,
        source: str = "data.root",
        target: str | None = None,
        name: str | None = None,
        random_state: int | None = None,
    ) -> None:
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"downsample: frac must be in (0, 1], got {frac}.")

        self.frac = frac
        self.class_col = class_col
        self.source = source
        self.target = target or source
        self.random_state = random_state

        super().__init__(
            name=name or "downsample",
            requires=[self.source],
            provides=[self.target],
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        if obj.empty:
            state[self.target] = obj
            return

        # Optimized sampling logic
        if self.class_col is not None and self.class_col in obj.columns:
            # Per-class sampling with efficient groupby
            sampled = obj.groupby(
                self.class_col, dropna=False, group_keys=False, sort=False
            ).sample(frac=self.frac, replace=False, random_state=self.random_state)
        else:
            # Global sampling (handles both None class_col and missing column cases)
            sampled = obj.sample(
                frac=self.frac, replace=False, random_state=self.random_state
            )

        state[self.target] = reattach_meta(obj, sampled)
