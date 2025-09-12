from typing import Optional

import numpy as np
import pandas as pd

from ..helpers import validate_instance
from ...core.state import State
from ...core.step import FitAwareStep
from ...data.partition import PartitionName


class StandardScale(FitAwareStep):
    """Standardize numerical columns using mean/std with overflow-safe scaling."""

    def __init__(
        self,
        dataframe_in: str = "data.root",
        dataframe_out: str | None = None,
        name: str | None = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in

        self._scale: Optional[pd.Series] = None
        self._means_s: Optional[pd.Series] = None
        self._stds_s: Optional[pd.Series] = None

        super().__init__(
            name=name or "standard_scale",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out],
        )

    def fit_impl(self, state: State) -> None:
        """Fit scaling stats on train split (overflow-safe)."""
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        numerical_data = dataframe.tab.train.tab.numerical
        if numerical_data.shape[1] == 0:
            self._scale = pd.Series(dtype="float64")
            self._means_s = pd.Series(dtype="float64")
            self._stds_s = pd.Series(dtype="float64")
            return

        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )

        # Overflow-safe: compute scale and scaled values efficiently
        abs_max = numerical_data.abs().max(axis=0)
        self._scale = (
            abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0)
        )

        num_scaled = numerical_data / self._scale
        self._means_s = num_scaled.mean()
        self._stds_s = num_scaled.std(ddof=0).clip(lower=1e-10, upper=None)

    def run(self, state: State) -> None:
        """Apply standardization to numerical columns."""
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        numerical_data = dataframe.tab.numerical
        if numerical_data.shape[1] == 0:
            state[self.dataframe_out] = dataframe
            return

        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )

        # Reindex scaling parameters efficiently
        cols = numerical_data.columns
        scale = self._scale.reindex(cols, fill_value=1.0)
        means_s = self._means_s.reindex(cols, fill_value=0.0)
        stds_s = self._stds_s.reindex(cols, fill_value=1.0)

        dataframe.tab.numerical = (numerical_data / scale - means_s) / stds_s
        state[self.dataframe_out] = dataframe


class MinMaxScale(FitAwareStep):
    """Scale numerical columns to [0, 1] via min/max."""

    def __init__(
        self,
        dataframe_in: str = "data.root",
        dataframe_out: str | None = None,
        name: str | None = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in

        self._min: Optional[pd.Series] = None
        self._max: Optional[pd.Series] = None

        super().__init__(
            name=name or "min_max_scale",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out],
        )

    def fit_impl(self, state: State) -> None:
        """Fit min/max on train split."""
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        numerical_data = dataframe.tab.train.tab.numerical
        if numerical_data.shape[1] == 0:
            self._min = pd.Series(dtype="float64")
            self._max = pd.Series(dtype="float64")
            return

        # Convert and clean data, then compute min/max in one efficient operation
        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )
        self._min = numerical_data.min()
        self._max = numerical_data.max()

    def run(self, state: State) -> None:
        """Apply min-max scaling to numerical columns."""
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        numerical_data = dataframe.tab.numerical
        if numerical_data.shape[1] == 0:
            state[self.dataframe_out] = dataframe
            return

        # Convert and clean data in one step
        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )

        # Reindex scaling parameters and apply min-max scaling efficiently
        cols = numerical_data.columns
        col_min = self._min.reindex(cols, fill_value=0.0)
        col_max = self._max.reindex(cols, fill_value=1.0)
        den = (col_max - col_min).clip(lower=1e-10, upper=None)

        dataframe.tab.numerical = (numerical_data - col_min) / den
        state[self.dataframe_out] = dataframe
