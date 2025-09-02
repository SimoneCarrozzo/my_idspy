from typing import Final

from ..utils import validate_instance
from ...core.state import State
from ...core.step import Step
from ...data.tabular_data import Data, DataView


class DownsampleToMinority(Step):
    """Downsample each class to the minority class size."""

    def __init__(
            self,
            target: str,
            input_key: str = "data.default",
            output_key: str | None = None,
            name: str | None = None,
            random_state: int | None = None,
    ) -> None:
        self.target: Final[str] = target
        self.input_key: Final[str] = input_key
        self.output_key: Final[str] = output_key or input_key
        self.random_state: Final[int | None] = random_state

        super().__init__(
            name=name or "downsample_to_minority",
            requires=[self.input_key],
            provides=[self.output_key],
        )

    def run(self, state: State) -> None:
        data: Data | DataView = state[self.input_key]
        validate_instance(data, (Data, DataView), self.name)
        df = data.df

        if self.target not in df.columns or df.empty:
            state[self.output_key] = data
            return

        counts = df[self.target].value_counts(dropna=False)
        if counts.empty:
            state[self.output_key] = data
            return

        minority = int(counts.min())
        shuffled = df.sample(frac=1, replace=False, random_state=self.random_state)

        sampled = (
            shuffled.groupby(self.target, dropna=False, group_keys=False, sort=False)
            .head(minority)
        )

        state[self.output_key] = data.view(sampled.index.tolist())


class Downsample(Step):
    """Downsample rows globally or per-class (if `target` is set)."""

    def __init__(
            self,
            frac: float,
            target: str | None = None,
            input_key: str = "data.default",
            output_key: str | None = None,
            name: str | None = None,
            random_state: int | None = None,
    ) -> None:
        if not (0 < frac <= 1):
            raise ValueError(f"downsample: frac must be in (0, 1], got {frac}.")

        self.frac: Final[float] = frac
        self.target: Final[str | None] = target
        self.input_key: Final[str] = input_key
        self.output_key: Final[str] = output_key or input_key
        self.random_state: Final[int | None] = random_state

        super().__init__(
            name=name or "downsample",
            requires=[self.input_key],
            provides=[self.output_key],
        )

    def run(self, state: State) -> None:
        data: Data | DataView = state[self.input_key]
        validate_instance(data, (Data, DataView), self.name)
        df = data.df

        if df.empty:
            state[self.output_key] = data
            return

        if self.target:
            if self.target not in df.columns:
                state[self.output_key] = data
                return

            sampled = (
                df.groupby(self.target, group_keys=False, sort=False)
                .sample(frac=self.frac, replace=False, random_state=self.random_state)
            )
            selected_idx = sampled.index
        else:
            selected_idx = df.sample(frac=self.frac, replace=False, random_state=self.random_state).index

        state[self.output_key] = data.view(selected_idx.tolist())
