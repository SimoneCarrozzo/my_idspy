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

        self._scale: pd.Series | None = None
        self._means_s: pd.Series | None = None
        self._stds_s: pd.Series | None = None

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

        # overflow-safe:  mean/std over scaled values
        scale = num.abs().max(axis=0).fillna(0.0).where(lambda s: s > 0.0, 1.0)
        num_s = num.divide(scale, axis="columns")

        self._scale = scale
        self._means_s = num_s.mean()
        self._stds_s = num_s.std(ddof=0).replace(0.0, 1.0)

    def run(self, state: State) -> None:
        """Apply standardization to numerical columns."""
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        num = obj.tab.numerical
        if num.shape[1] == 0:
            state[self.target] = obj
            return

        num = num.astype(np.float64, copy=False).replace([np.inf, -np.inf], np.nan)

        cols = num.columns
        scale = self._scale.reindex(cols, fill_value=1.0)
        means_s = self._means_s.reindex(cols, fill_value=0.0)
        stds_s = self._stds_s.reindex(cols, fill_value=1.0)

        num_s = num.divide(scale, axis="columns")
        obj.tab.numerical = (num_s - means_s) / stds_s

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

        self._min: pd.Series | None = None
        self._max: pd.Series | None = None

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

        num = num.astype(np.float64, copy=False).replace([np.inf, -np.inf], np.nan)

        cols = num.columns
        col_min = self._min.reindex(cols, fill_value=0.0)
        col_max = self._max.reindex(cols, fill_value=1.0)
        den = (col_max - col_min).replace(0.0, 1.0)

        obj.tab.numerical = (num - col_min) / den
        state[self.target] = obj
