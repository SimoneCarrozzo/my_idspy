import numpy as np
import pandas as pd

from ..helpers import validate_instance
from ...core.state import State
from ...core.step import Step
from ...data.tab_accessor import reattach_meta


class DropNulls(Step):
    """Drop all rows that contain null values, including NaN and ±inf."""

    def __init__(
        self,
        dataframe_in: str = "data.root",
        dataframe_out: str | None = None,
        name: str | None = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in

        super().__init__(
            name=name or "drop_nulls",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out],
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()
        state[self.dataframe_out] = dataframe


class Filter(Step):
    """Filter rows using a pandas query string."""

    def __init__(
        self,
        query: str,
        dataframe_in: str = "data.root",
        dataframe_out: str | None = None,
        name: str | None = None,
    ) -> None:
        self.query = query
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in

        super().__init__(
            name=name or "filter",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out],
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        filtered = dataframe.query(self.query)
        state[self.dataframe_out] = reattach_meta(dataframe, filtered)


class Log1p(Step):
    """Apply np.log1p to numerical columns."""

    def __init__(
        self,
        dataframe_in: str = "data.root",
        dataframe_out: str | None = None,
        name: str | None = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in

        super().__init__(
            name=name or "log1p",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out],
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        dataframe.tab.numerical = np.log1p(dataframe.tab.numerical)
        state[self.dataframe_out] = dataframe
