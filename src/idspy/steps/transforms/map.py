from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from ...core.step import FitAwareStep, Step
from ...core.state import State


class FrequencyMap(FitAwareStep):
    """Map categorical columns to frequency-rank codes."""

    def __init__(
        self,
        max_levels: Optional[int] = None,
        default: int = 0,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.max_levels = max_levels
        self.default = default
        self.cat_types: Dict[str, CategoricalDtype] = {}

        super().__init__(
            name=name or "frequency_map",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        """Infer ordered categories by frequency from train split."""
        train_df = root.tab.train
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

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame, cat_mapping=dict)
    def run(self, state: State, root: pd.DataFrame) -> None:
        """Apply learned frequency mapping to categorical columns."""

        # Early exit if no categorical mappings learned
        if not self.cat_types:
            return {"root": root, "cat_mapping": {}}

        cat_cols = root.tab.categorical.columns
        for col in cat_cols:
            if col not in self.cat_types or col not in root.columns:
                continue

            s = root[col].astype(self.cat_types[col])
            codes = s.cat.codes
            root[col] = np.where(codes != -1, codes + 1, self.default).astype("int32")

        return {"root": root, "cat_mapping": self.cat_types}


class LabelMap(FitAwareStep):
    """Encode `target`: binary with `benign_tag`, else ordinal categories."""

    def __init__(
        self,
        benign_tag: Optional[str] = None,
        default: int = -1,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.benign_tag = benign_tag
        self.default = default
        self.cat_types: Optional[CategoricalDtype] = None

        super().__init__(
            name=name or "target_map",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        """Learn ordered categories for the target col (if not binary)."""

        # Early exit for binary case
        if self.benign_tag is not None:
            self.cat_types = None
            return

        train_df = root.tab.train
        tgt_col = train_df.tab.schema.target

        vc = train_df[tgt_col].value_counts(dropna=False)
        self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame, target_mapping=CategoricalDtype | None)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        tgt_col = root.tab.schema.target

        prev = root[tgt_col].copy()

        if self.benign_tag is not None:
            tgt = (prev == self.benign_tag).astype("int32")
            tgt = tgt.where(tgt == 0, 1)
        else:
            s = prev.astype(self.cat_types)
            codes = s.cat.codes
            tgt = pd.Series(
                np.where(codes != -1, codes + 1, self.default).astype("int32"),
                index=s.index,
                name=tgt_col,
            )

        root[f"original_{tgt_col}"] = prev
        root.tab.target = tgt
        return {"root": root, "target_mapping": self.cat_types}
