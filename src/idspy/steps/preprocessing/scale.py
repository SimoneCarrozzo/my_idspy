from typing import Optional

import numpy as np
import pandas as pd

from ..utils import validate_instance, validate_schema_and_split
from ...core.state import State
from ...core.step import FitAwareStep
from ...data.split import SplitName


class StandardScale(FitAwareStep):
    """Standardize numerical columns using mean/std with overflow-safe scaling."""

    def __init__(
        self,
        source: str = "data.root",
        target: str | None = None,
        name: str | None = None,
    ) -> None:
        self.source = source
        self.target = target or source

        self._scale: Optional[pd.Series] = None
        self._means_s: Optional[pd.Series] = None
        self._stds_s: Optional[pd.Series] = None

        super().__init__(
            name=name or "standard_scale",
            requires=[self.source],
            provides=[self.target],
        )

    def fit_impl(self, state: State) -> None:
        """Fit scaling stats on train split (overflow-safe)."""
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)
        validate_schema_and_split(obj, self.source, [SplitName.TRAIN.value])

        num = obj.tab.train.tab.numerical
        if num.shape[1] == 0:
            self._scale = pd.Series(dtype="float64")
            self._means_s = pd.Series(dtype="float64")
            self._stds_s = pd.Series(dtype="float64")
            return

        num = num.astype(np.float64, copy=False).replace([np.inf, -np.inf], np.nan)

        # Overflow-safe: compute scale and scaled values efficiently
        abs_max = num.abs().max(axis=0)
        self._scale = (
            abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0)
        )

        num_scaled = num / self._scale
        self._means_s = num_scaled.mean()
        self._stds_s = num_scaled.std(ddof=0).clip(lower=1e-10, upper=None)

    def run(self, state: State) -> None:
        """Apply standardization to numerical columns."""
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        num = obj.tab.numerical
        if num.shape[1] == 0:
            state[self.target] = obj
            return

        num = num.astype(np.float64, copy=False).replace([np.inf, -np.inf], np.nan)

        # Reindex scaling parameters efficiently
        cols = num.columns
        scale = self._scale.reindex(cols, fill_value=1.0)
        means_s = self._means_s.reindex(cols, fill_value=0.0)
        stds_s = self._stds_s.reindex(cols, fill_value=1.0)

        obj.tab.numerical = (num / scale - means_s) / stds_s
        state[self.target] = obj


class MinMaxScale(FitAwareStep):
    """Scale numerical columns to [0, 1] via min/max."""

    def __init__(
        self,
        source: str = "data.root",
        target: str | None = None,
        name: str | None = None,
    ) -> None:
        self.source = source
        self.target = target or source

        self._min: Optional[pd.Series] = None
        self._max: Optional[pd.Series] = None

        super().__init__(
            name=name or "min_max_scale",
            requires=[self.source],
            provides=[self.target],
        )

    def fit_impl(self, state: State) -> None:
        """Fit min/max on train split."""
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)
        validate_schema_and_split(obj, self.source, [SplitName.TRAIN.value])

        num = obj.tab.train.tab.numerical
        if num.shape[1] == 0:
            self._min = pd.Series(dtype="float64")
            self._max = pd.Series(dtype="float64")
            return

        # Convert and clean data, then compute min/max in one efficient operation
        num = num.astype(np.float64, copy=False).replace([np.inf, -np.inf], np.nan)
        self._min = num.min()
        self._max = num.max()

    def run(self, state: State) -> None:
        """Apply min-max scaling to numerical columns."""
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        num = obj.tab.numerical
        if num.shape[1] == 0:
            state[self.target] = obj
            return

        # Convert and clean data in one step
        num = num.astype(np.float64, copy=False).replace([np.inf, -np.inf], np.nan)

        # Reindex scaling parameters and apply min-max scaling efficiently
        cols = num.columns
        col_min = self._min.reindex(cols, fill_value=0.0)
        col_max = self._max.reindex(cols, fill_value=1.0)
        den = (col_max - col_min).clip(lower=1e-10, upper=None)

        obj.tab.numerical = (num - col_min) / den
        state[self.target] = obj
