import pandas as pd

from ..helpers import validate_instance
from ...core.state import State
from ...core.step import Step
from ...data.tab_accessor import reattach_meta


class DownsampleToMinority(Step):
    """Downsample each class to the size of the minority class."""

    def __init__(
        self,
        class_column: str,
        dataframe_in: str = "data.root",
        dataframe_out: str | None = None,
        random_state: int | None = None,
        name: str | None = None,
    ) -> None:
        self.class_column = class_column
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in
        self.random_state = random_state

        super().__init__(
            name=name or "downsample_to_minority",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out],
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        # Early exits for edge cases
        if dataframe.empty or self.class_column not in dataframe.columns:
            state[self.dataframe_out] = dataframe
            return

        counts = dataframe[self.class_column].value_counts(dropna=False)
        if counts.empty:
            state[self.dataframe_out] = dataframe
            return

        minority = int(counts.min())
        if minority <= 0:
            sampled = dataframe.iloc[0:0]  # empty but keep schema
            state[self.dataframe_out] = reattach_meta(dataframe, sampled)
            return

        # Efficient sampling: avoid full shuffle if not needed
        sampled = dataframe.groupby(
            self.class_column, dropna=False, group_keys=False, sort=False
        ).sample(n=minority, replace=False, random_state=self.random_state)

        state[self.dataframe_out] = reattach_meta(dataframe, sampled)


class Downsample(Step):
    """Downsample rows globally or per class."""

    def __init__(
        self,
        frac: float,
        dataframe_in: str = "data.root",
        dataframe_out: str | None = None,
        class_column: str | None = None,
        random_state: int | None = None,
        name: str | None = None,
    ) -> None:
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"downsample: frac must be in (0, 1], got {frac}.")

        self.frac = frac
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in
        self.class_column = class_column
        self.random_state = random_state

        super().__init__(
            name=name or "downsample",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out],
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        if dataframe.empty:
            state[self.dataframe_out] = dataframe
            return

        # Optimized sampling logic
        if self.class_column is not None and self.class_column in dataframe.columns:
            # Per-class sampling with efficient groupby
            sampled = dataframe.groupby(
                self.class_column, dropna=False, group_keys=False, sort=False
            ).sample(frac=self.frac, replace=False, random_state=self.random_state)
        else:
            # Global sampling (handles both None class_column and missing column cases)
            sampled = dataframe.sample(
                frac=self.frac, replace=False, random_state=self.random_state
            )

        state[self.dataframe_out] = reattach_meta(dataframe, sampled)
