from typing import Final

import numpy as np

from ..utils import validate_instance
from ...core.state import State
from ...core.step import Step
from ...data.tabular_data import Data, DataView, TabularData, TabularView


class DropNulls(Step):
    """Drop all rows that contain null values, including NaN and ±inf."""

    def __init__(
            self,
            input_key: str = "data.default",
            output_key: str | None = None,
            name: str | None = None,
    ) -> None:
        self.input_key: Final[str] = input_key
        self.output_key: Final[str] = output_key or input_key

        super().__init__(
            name=name or "drop_nulls",
            requires=[self.input_key],
            provides=[self.output_key],
        )

    def run(self, state: State) -> None:
        data: Data | DataView = state[self.input_key]
        validate_instance(data, (Data, DataView), self.name)

        df = data.df
        data.df = df[~df.isin([np.inf, -np.inf, np.nan]).any(axis=1)]
        state[self.output_key] = data


class Filter(Step):
    """Filter rows using a pandas query string."""

    def __init__(
            self,
            query: str,
            input_key: str = "data.default",
            output_key: str | None = None,
            name: str | None = None,
    ) -> None:
        self.query: Final[str] = query
        self.input_key: Final[str] = input_key
        self.output_key: Final[str] = output_key or input_key

        super().__init__(
            name=name or "filter",
            requires=[self.input_key],
            provides=[self.output_key],
        )

    def run(self, state: State) -> None:
        data: Data | DataView = state[self.input_key]
        validate_instance(data, (Data, DataView), self.name)

        state[self.output_key] = data.view_from_query(self.query)


class Log1p(Step):
    """Apply np.log1p to numeric columns."""

    def __init__(
            self,
            input_key: str = "data.default",
            output_key: str | None = None,
            name: str | None = None,
    ) -> None:
        self.input_key: Final[str] = input_key
        self.output_key: Final[str] = output_key or input_key

        super().__init__(
            name=name or "log1p",
            requires=[self.input_key],
            provides=[self.output_key],
        )

    def run(self, state: State) -> None:
        data: TabularData | TabularView = state[self.input_key]
        validate_instance(data, (TabularData, TabularView), self.name)

        num = data.numeric
        data.numeric = np.log1p(num)
        state[self.output_key] = data
