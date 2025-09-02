from typing import Final, Optional, Dict

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from ..utils import validate_instance
from ...core.state import State
from ...core.step import FitAwareStep
from ...data.tabular_data import TabularData, TabularView


class FrequencyMap(FitAwareStep):
    """Map categorical columns to frequency-rank codes."""

    def __init__(
            self,
            max_levels: Optional[int] = None,
            default: int = 0,
            input_key: str = "data.default",
            fit_key: str = "data.train",
            output_key: Optional[str] = None,
            name: Optional[str] = None,
    ) -> None:
        self.max_levels: Final[Optional[int]] = max_levels
        self.default: Final[int] = default
        self.input_key: Final[str] = input_key
        self.fit_key: Final[str] = fit_key
        self.output_key: Final[str] = output_key or input_key
        self.cat_types: Dict[str, CategoricalDtype] = {}

        super().__init__(
            name=name or "frequency_map",
            requires=[self.input_key, self.fit_key],
            provides=[self.output_key],
        )

    def fit_core(self, state: State) -> None:
        data: TabularData | TabularView = state[self.fit_key]
        validate_instance(data, (TabularData, TabularView), self.name)

        self.cat_types.clear()
        for col in data.schema.categorical:
            vc = data.df[col].value_counts(dropna=False)
            cats = (
                vc.index.tolist()
                if self.max_levels is None
                else vc.nlargest(self.max_levels).index.tolist()
            )
            self.cat_types[col] = CategoricalDtype(categories=cats, ordered=True)

    def run(self, state: State) -> None:
        data: TabularData | TabularView = state[self.input_key]
        validate_instance(data, (TabularData, TabularView), self.name)

        out = data.df.copy()
        for col in data.schema.categorical:
            # If a column wasn't present during fit, skip it.
            if col not in self.cat_types or col not in out.columns:
                continue

            s = out[col].astype(self.cat_types[col])
            codes = s.cat.codes.to_numpy()  # -1 for unknowns
            mapped = np.where(codes != -1, codes + 1, self.default).astype("int32")
            out[col] = mapped

        data.df = out
        state[self.output_key] = data


class TargetMap(FitAwareStep):
    """Encode target: binary with `benign_tag`, else ordinal categories."""

    def __init__(
            self,
            benign_tag: Optional[str] = None,
            target_out: Optional[str] = None,
            default: int = -1,
            input_key: str = "data.default",
            fit_key: str = "data.train",
            output_key: Optional[str] = None,
            name: Optional[str] = None,
    ) -> None:
        self.benign_tag: Final[Optional[str]] = benign_tag
        self.target_out: Final[Optional[str]] = target_out
        self.default: Final[int] = default
        self.input_key: Final[str] = input_key
        self.fit_key: Final[str] = fit_key
        self.output_key: Final[str] = output_key or input_key
        self.cat_types: Optional[CategoricalDtype] = None

        super().__init__(
            name=name or "target_map",
            requires=[self.input_key, self.fit_key],
            provides=[self.output_key],
        )

    def fit_core(self, state: State) -> None:
        data: TabularData | TabularView = state[self.fit_key]
        validate_instance(data, (TabularData, TabularView), self.name)

        if self.benign_tag is None:
            vc = data.target.value_counts(dropna=False)
            self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)

    def run(self, state: State) -> None:
        data: TabularData | TabularView = state[self.input_key]
        validate_instance(data, (TabularData, TabularView), self.name)

        if self.benign_tag is not None:
            # Binary: benign_tag -> 0, everything else -> 1
            target = data.target.replace({self.benign_tag: 0}).where(lambda x: x == 0, 1)
            target = target.astype("int32")
        else:
            s = data.target.astype(self.cat_types)
            codes = s.cat.codes.to_numpy()
            target = pd.Series(
                np.where(codes != -1, codes + 1, self.default).astype("int32"),
                index=s.index,
                name=data.schema.target,
            )

        # Column name
        out_name = self.target_out or f"{data.schema.target}_encoded"
        target.name = out_name

        data.set_df(target)
        state[self.output_key] = data
