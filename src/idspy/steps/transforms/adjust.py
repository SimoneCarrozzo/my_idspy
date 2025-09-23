from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...core.step import Step
from ...core.state import State
from ...data.tab_accessor import reattach_meta


class DropNulls(Step):
    """Drop all rows that contain null values, including NaN and ±inf."""

    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or "drop_nulls",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        root = root.replace([np.inf, -np.inf], np.nan).dropna()
        return {"root": root}


class Filter(Step):
    """Filter rows using a pandas query string."""

    def __init__(
        self,
        query: str,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.query = query

        super().__init__(
            name=name or "filter",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        filtered = root.query(self.query)
        return {"root": reattach_meta(root, filtered)}


class Log1p(Step):
    """Apply np.log1p to numerical columns."""

    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or "log1p",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:

        root.tab.numerical = np.log1p(root.tab.numerical)
        return {"root": root}
