from typing import Optional, Dict

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from ..helpers import validate_instance
from ...core.state import State
from ...core.step import FitAwareStep
from ...data.partition import PartitionName
from ...data.tab_accessor import reattach_meta


class FrequencyMap(FitAwareStep):
    """Map categorical columns to frequency-rank codes."""

    def __init__(
        self,
        dataframe_in: str = "data.root",
        dataframe_out: Optional[str] = None,
        max_levels: Optional[int] = None,
        default: int = 0,
        name: Optional[str] = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in
        self.max_levels = max_levels
        self.default = default
        self.cat_types: Dict[str, CategoricalDtype] = {}

        super().__init__(
            name=name or "frequency_map",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out, "mapping.categorical"],
        )

    def fit_impl(self, state: State) -> None:
        """Infer ordered categories by frequency from train split."""
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        train_df = dataframe.tab.train
        self.cat_types.clear()

        cat_cols = train_df.tab.categorical.columns
        for col in cat_cols:
            vc = train_df[col].value_counts(dropna=False)
            if vc.empty:
                continue

            if self.max_levels is None:
                cats = vc.index.tolist()
            else:
                cats = vc.head(self.max_levels).index.tolist()

            self.cat_types[col] = CategoricalDtype(categories=cats, ordered=True)

    def run(self, state: State) -> None:
        """Apply learned frequency mapping to categorical columns."""
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        out = dataframe.copy()

        # Early exit if no categorical mappings learned
        if not self.cat_types:
            state[self.dataframe_out] = reattach_meta(dataframe, out)
            state["mapping.categorical"] = self.cat_types
            return

        cat_cols = dataframe.tab.categorical.columns
        for col in cat_cols:
            if col not in self.cat_types or col not in out.columns:
                continue

            s = out[col].astype(self.cat_types[col])
            codes = s.cat.codes
            out[col] = np.where(codes != -1, codes + 1, self.default).astype("int32")

        state[self.dataframe_out] = reattach_meta(dataframe, out)
        state["mapping.categorical"] = self.cat_types


class LabelMap(FitAwareStep):
    """Encode `target`: binary with `benign_tag`, else ordinal categories."""

    def __init__(
        self,
        dataframe_in: str = "data.root",
        dataframe_out: Optional[str] = None,
        benign_tag: Optional[str] = None,
        default: int = -1,
        name: Optional[str] = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in
        self.benign_tag = benign_tag
        self.default = default
        self.cat_types: Optional[CategoricalDtype] = None

        super().__init__(
            name=name or "target_map",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out, "mapping.target"],
        )

    def fit_impl(self, state: State) -> None:
        """Learn ordered categories for the target col (if not binary)."""
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        # Early exit for binary case
        if self.benign_tag is not None:
            self.cat_types = None
            return

        train_df = dataframe.tab.train
        tgt_cols = train_df.tab.target.columns
        if len(tgt_cols) != 1:
            raise ValueError(
                f"Expected exactly 1 target column, found {len(tgt_cols)}: {tgt_cols}"
            )

        tgt_col = tgt_cols[0]
        vc = train_df[tgt_col].value_counts(dropna=False)
        self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)

    def run(self, state: State) -> None:
        """Apply target encoding (binary or ordinal)."""
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        tgt_cols = dataframe.tab.target.columns
        if len(tgt_cols) != 1:
            raise ValueError(
                f"Expected exactly 1 target column, found {len(tgt_cols)}: {tgt_cols}"
            )
        tgt_col = tgt_cols[0]
        prev = dataframe[tgt_col].copy()

        if self.benign_tag is not None:
            tgt = (prev == self.benign_tag).astype("int32")
            tgt = tgt.where(tgt == 0, 1)
        else:
            if self.cat_types is None:
                raise RuntimeError("LabelMap was not fitted with category types.")

            s = prev.astype(self.cat_types)
            codes = s.cat.codes
            tgt = pd.Series(
                np.where(codes != -1, codes + 1, self.default).astype("int32"),
                index=s.index,
                name=tgt_col,
            )

        dataframe[f"original_{tgt_col}"] = prev
        dataframe.tab.target = tgt
        state[self.dataframe_out] = dataframe
        state["mapping.target"] = self.cat_types
