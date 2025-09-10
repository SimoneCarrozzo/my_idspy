import numpy as np
import pandas as pd

from ..utils import validate_instance, validate_schema
from ...core.state import State
from ...core.step import Step
from ...data.tab_accessor import reattach_meta


class DropNulls(Step):
    """Drop all rows that contain null values, including NaN and ±inf."""

    def __init__(
        self,
        source: str = "data.root",
        target: str | None = None,
        name: str | None = None,
    ) -> None:
        self.source = source
        self.target = target or source

        super().__init__(
            name=name or "drop_nulls",
            requires=[self.source],
            provides=[self.target],
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        obj = obj.replace([np.inf, -np.inf], np.nan).dropna()
        state[self.target] = obj


class Filter(Step):
    """Filter rows using a pandas query string."""

    def __init__(
        self,
        query: str,
        source: str = "data.root",
        target: str | None = None,
        name: str | None = None,
    ) -> None:
        self.query = query
        self.source = source
        self.target = target or source

        super().__init__(
            name=name or "filter",
            requires=[self.source],
            provides=[self.target],
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        out = obj.query(self.query)
        state[self.target] = reattach_meta(obj, out)


class Log1p(Step):
    """Apply np.log1p to numerical columns."""

    def __init__(
        self,
        source: str = "data.root",
        target: str | None = None,
        name: str | None = None,
    ) -> None:
        self.source = source
        self.target = target or source

        super().__init__(
            name=name or "log1p",
            requires=[self.source],
            provides=[self.target],
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)
        validate_schema(obj, self.name)

        obj.tab.numerical = np.log1p(obj.tab.numerical)
        state[self.target] = obj
