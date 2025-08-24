from typing import Final

import numpy as np
import pandas as pd

from ..utils import validate_instance
from ...core.state import State
from ...core.step import FittedStep
from ...data.tabular_data import TabularData, TabularView


class StandardScale(FittedStep):
    """Standardize numeric columns using mean/std with overflow-safe scaling."""

    def __init__(
            self,
            input_key: str = "data",
            fit_key: str = "train",
            output_key: str | None = None,
            name: str | None = None,
    ) -> None:
        self.input_key: Final[str] = input_key
        self.fit_key: Final[str] = fit_key
        self.output_key: Final[str] = output_key or input_key

        self._scale: pd.Series | None = None
        self._means_s: pd.Series | None = None
        self._stds_s: pd.Series | None = None

        super().__init__(
            name=name or "standard_scale",
            requires=[self.input_key, self.fit_key],
            produces=[self.output_key],
        )

    def _fit(self, state: State) -> None:
        data: TabularData | TabularView = state[self.fit_key]
        validate_instance(data, (TabularData, TabularView), self.name)

        num = data.numeric.astype(np.float64, copy=False)

        scale = num.abs().max(axis=0).fillna(0.0)
        scale = scale.where(scale > 0.0, 1.0)
        num_s = num.divide(scale, axis="columns")

        self._scale = scale
        self._means_s = num_s.mean()
        self._stds_s = num_s.std(ddof=0).replace(0.0, 1.0)

    def _run(self, state: State) -> None:
        data: TabularData | TabularView = state[self.input_key]
        validate_instance(data, (TabularData, TabularView), self.name)

        num = data.numeric.astype(np.float64, copy=False).replace([np.inf, -np.inf], np.nan)

        # Align saved stats to current columns
        scale = self._scale.reindex(num.columns, fill_value=1.0)
        means_s = self._means_s.reindex(num.columns, fill_value=0.0)
        stds_s = self._stds_s.reindex(num.columns, fill_value=1.0)

        num_s = num.divide(scale, axis="columns")
        data.numeric = (num_s - means_s) / stds_s

        state[self.output_key] = data


class MinMaxScale(FittedStep):
    """Scale numeric columns to [0, 1] using min/max."""

    def __init__(
            self,
            input_key: str = "data",
            fit_key: str = "train",
            output_key: str | None = None,
            name: str | None = None,
    ) -> None:
        self.input_key: Final[str] = input_key
        self.fit_key: Final[str] = fit_key
        self.output_key: Final[str] = output_key or input_key

        self._min: pd.Series | None = None
        self._max: pd.Series | None = None

        super().__init__(
            name=name or "min_max_scale",
            requires=[self.input_key, self.fit_key],
            produces=[self.output_key],
        )

    def _fit(self, state: State) -> None:
        data: TabularData | TabularView = state[self.fit_key]
        validate_instance(data, (TabularData, TabularView), self.name)

        self._min = data.numeric.min()
        self._max = data.numeric.max()

    def _run(self, state: State) -> None:
        data: TabularData | TabularView = state[self.input_key]
        validate_instance(data, (TabularData, TabularView), self.name)

        col_min = self._min.reindex(data.numeric.columns, fill_value=0)  # type: ignore[union-attr]
        col_max = self._max.reindex(data.numeric.columns, fill_value=1)  # type: ignore[union-attr]
        den = (col_max - col_min).replace(0, 1)

        data.numeric = (data.numeric - col_min) / den
        state[self.output_key] = data
