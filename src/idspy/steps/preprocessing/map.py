from typing import Optional, Dict

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from ..utils import validate_instance, validate_schema_and_split
from ...core.state import State
from ...core.step import FitAwareStep
from ...data.split import SplitName
from ...data.tab_accessor import reattach_meta


class FrequencyMap(FitAwareStep):
    """Map categorical columns to frequency-rank codes."""

    def __init__(
            self,
            max_levels: Optional[int] = None,
            default: int = 0,
            source: str = "data.root",
            target: Optional[str] = None,
            name: Optional[str] = None,
    ) -> None:
        self.max_levels = max_levels
        self.default = default
        self.source = source
        self.target = target or source
        self.cat_types: Dict[str, CategoricalDtype] = {}

        super().__init__(
            name=name or "frequency_map",
            requires=[self.source],
            provides=[self.target, "mapping.categorical"],
        )

    def fit_impl(self, state: State) -> None:
        """Infer ordered categories by frequency from train split."""
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        validate_schema_and_split(obj, self.source, [SplitName.TRAIN.value])
        train_df = obj.tab.train
        self.cat_types.clear()

        for col in train_df.tab.categorical.columns:
            vc = train_df[col].value_counts(dropna=False)
            cats = (
                vc.index.tolist()
                if self.max_levels is None
                else vc.nlargest(self.max_levels).index.tolist()
            )
            self.cat_types[col] = CategoricalDtype(categories=cats, ordered=True)

    def run(self, state: State) -> None:
        """Apply learned frequency mapping to categorical columns."""
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        out = obj.copy()
        for col in obj.tab.categorical.columns:
            if col not in self.cat_types or col not in out.columns:
                continue
            s = out[col].astype(self.cat_types[col])
            codes = s.cat.codes.to_numpy()  # -1 for unknowns
            mapped = np.where(codes != -1, codes + 1, self.default).astype("int32")
            out[col] = mapped

        state[self.target] = reattach_meta(obj, out)
        state["mapping.categorical"] = self.cat_types


class LabelMap(FitAwareStep):
    """Encode `target`: binary with `benign_tag`, else ordinal categories."""

    def __init__(
            self,
            benign_tag: Optional[str] = None,
            default: int = -1,
            source: str = "data.root",
            target: Optional[str] = None,
            name: Optional[str] = None,
    ) -> None:
        self.benign_tag = benign_tag
        self.default = default
        self.source = source
        self.target = target or source
        self.cat_types: Optional[CategoricalDtype] = None

        super().__init__(
            name=name or "target_map",
            requires=[self.source],
            provides=[self.target, "mapping.target"],
        )

    def fit_impl(self, state: State) -> None:
        """Learn ordered categories for the target col (if not binary)."""
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        validate_schema_and_split(obj, self.source, [SplitName.TRAIN.value])
        train_df = obj.tab.train
        if self.benign_tag is not None:
            self.cat_types = None
            return

        tgt_cols = train_df.tab.target.columns
        if len(tgt_cols) != 1:
            raise ValueError(f"Expected exactly 1 target column, found {len(tgt_cols)}: {tgt_cols}")
        tgt_col = tgt_cols[0]

        vc = train_df[tgt_col].value_counts(dropna=False)
        self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)

    def run(self, state: State) -> None:
        """Apply target encoding (binary or ordinal)."""
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        tgt_cols = obj.tab.target.columns
        if len(tgt_cols) != 1:
            raise ValueError(f"Expected exactly 1 target column, found {len(tgt_cols)}: {tgt_cols}")
        tgt_col = tgt_cols[0]

        prev = obj[tgt_col].copy()

        if self.benign_tag is not None:
            tgt = prev.replace({self.benign_tag: 0}).where(lambda x: x == 0, 1).astype("int32")
        else:
            if self.cat_types is None:
                raise RuntimeError("LabelMap was not fitted with category types.")
            s = prev.astype(self.cat_types)

            codes = s.cat.codes.to_numpy()  # -1 for unknowns
            tgt = pd.Series(
                np.where(codes != -1, codes + 1, self.default).astype("int32"),
                index=s.index,
                name=tgt_col,
            )

        obj[f"original_{tgt_col}"] = prev
        obj.tab.target = tgt
        state[self.target] = obj
        state["mapping.target"] = self.cat_types
