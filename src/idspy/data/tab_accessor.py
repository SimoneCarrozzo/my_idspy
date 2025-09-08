from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

from .schema import Schema, ColumnRole
from .split import Split, SplitName
from ..services.profiler import time_profiler


@register_dataframe_accessor("tab")
class TabAccessor:
    """Schema + split accessor."""

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj
        self._obj.attrs.setdefault("_schema", Schema())
        self._obj.attrs.setdefault("_splits", Split())

    def _attach_meta(self, out: pd.DataFrame) -> pd.DataFrame:
        """Attach schema (optionally pruned) and splits to `out`."""
        out.attrs["_schema"] = self.schema.clone_pruned(out.columns)
        out.attrs["_splits"] = self.splits.clone()
        return out

    @property
    def schema(self) -> Schema:
        return self._obj.attrs["_schema"]

    def set_schema(self, schema: Union["Schema", None] = None, **roles: List[str]) -> pd.DataFrame:
        """Replace schema with new role definitions (string keys allowed)."""
        if schema is not None:
            sch = schema
        else:
            sch = Schema(strict=self.schema.strict)
            for role_name, cols in roles.items():
                sch.add(cols, ColumnRole.from_name(role_name))

        self._obj.attrs["_schema"] = sch
        return self._obj

    def add_role(self, cols: Union[Iterable[str], str], role: Union[ColumnRole, str]) -> pd.DataFrame:
        """Add columns to a role."""
        self.schema.add(cols, role)
        return self._obj

    def update_role(self, cols: Union[Iterable[str], str], role: Union[ColumnRole, str]) -> pd.DataFrame:
        """Replace columns for a role and clean conflicts."""
        self.schema.update(cols, role)
        return self._obj

    def columns(self, role: Union[ColumnRole, str]) -> List[str]:
        self.schema.prune_missing(self._obj.columns)
        return self.schema.columns(role)

    @property
    def features(self) -> pd.DataFrame:
        """Feature view (excludes target)."""
        self.schema.prune_missing(self._obj.columns)
        cols = self.schema.feature_columns()
        out = self._obj.loc[:, cols]
        return self._attach_meta(out)

    @features.setter
    def features(self, updated: pd.DataFrame) -> None:
        self.schema.prune_missing(self._obj.columns)
        cols = self.schema.feature_columns()
        self._assign_block(updated, rows=None, cols=cols)

    @property
    def target(self) -> pd.DataFrame:
        """Target view."""
        self.schema.prune_missing(self._obj.columns)
        cols = self.schema.columns(ColumnRole.TARGET)
        if not cols:
            raise ValueError("No target is defined in the schema.")
        out = self._obj.loc[:, cols]
        return self._attach_meta(out)

    @target.setter
    def target(self, updated: pd.DataFrame) -> None:
        self.schema.prune_missing(self._obj.columns)
        cols = self.schema.columns(ColumnRole.TARGET)
        if not cols:
            raise ValueError("No target is defined in the schema.")
        self._assign_block(updated, rows=None, cols=cols)

    @property
    def numerical(self) -> pd.DataFrame:
        """Numerical columns view."""
        self.schema.prune_missing(self._obj.columns)
        cols = self.schema.columns(ColumnRole.NUMERICAL)
        out = self._obj.loc[:, cols]
        return self._attach_meta(out)

    @numerical.setter
    def numerical(self, updated: Union[pd.DataFrame, pd.Series]) -> None:
        self.schema.prune_missing(self._obj.columns)
        cols = self.schema.columns(ColumnRole.NUMERICAL)
        self._assign_block(updated, rows=None, cols=cols)

    @property
    def categorical(self) -> pd.DataFrame:
        """Categorical columns view."""
        self.schema.prune_missing(self._obj.columns)
        cols = self.schema.columns(ColumnRole.CATEGORICAL)
        out = self._obj.loc[:, cols]
        return self._attach_meta(out)

    @categorical.setter
    def categorical(self, updated: Union[pd.DataFrame, pd.Series]) -> None:
        self.schema.prune_missing(self._obj.columns)
        cols = self.schema.columns(ColumnRole.CATEGORICAL)
        self._assign_block(updated, rows=None, cols=cols)

    @property
    def splits(self) -> Split:
        return self._obj.attrs["_splits"]

    def set_splits_from_labels(self, mapping: Dict[str, Iterable]) -> pd.DataFrame:
        """Define splits from index labels."""
        sp = Split()
        sp.set_from_labels(mapping)
        self._obj.attrs["_splits"] = sp
        return self._obj

    def set_splits_from_positions(self, mapping: Dict[str, Iterable[int]]) -> pd.DataFrame:
        """Define splits from integer positions."""
        sp = Split()
        sp.set_from_positions(mapping, self._obj.index)
        self._obj.attrs["_splits"] = sp
        return self._obj

    def get_split(self, name: str) -> pd.DataFrame:
        """Return dataframe slice for a split."""
        idx = self.splits.indices_for(name, self._obj)
        out = self._obj.iloc[idx]
        return self._attach_meta(out)

    @property
    def train(self) -> pd.DataFrame:
        return self.get_split(SplitName.TRAIN.value)

    @train.setter
    def train(self, updated: pd.DataFrame) -> None:
        idx = self.splits.indices_for(SplitName.TRAIN.value, self._obj)
        self._assign_block(updated, rows=idx, cols=None)

    @property
    def val(self) -> pd.DataFrame:
        return self.get_split(SplitName.VAL.value)

    @val.setter
    def val(self, updated: pd.DataFrame) -> None:
        idx = self.splits.indices_for(SplitName.VAL.value, self._obj)
        self._assign_block(updated, rows=idx, cols=None)

    @property
    def test(self) -> pd.DataFrame:
        return self.get_split(SplitName.TEST.value)

    @test.setter
    def test(self, updated: pd.DataFrame) -> None:
        idx = self.splits.indices_for(SplitName.TEST.value, self._obj)
        self._assign_block(updated, rows=idx, cols=None)

    # @time_profiler
    def _assign_block(
            self,
            updated: Union[pd.DataFrame, pd.Series],
            rows: Optional[np.ndarray],
            cols: Optional[List[str]],
    ) -> None:
        """Assign updated values aligned by index and columns."""
        df = self._obj
        upd = updated.to_frame() if isinstance(updated, pd.Series) else updated

        target_index = df.index if rows is None else df.index[rows]
        target_cols = df.columns if cols is None else pd.Index(cols)

        common_index = target_index.intersection(upd.index)
        if not len(common_index):
            raise ValueError("No overlapping rows between destination and updated (index misaligned).")

        common_cols = target_cols.intersection(upd.columns)
        if not len(common_cols):
            raise ValueError("No overlapping columns between destination and updated.")

        if rows is None and common_index.equals(df.index):
            # full replacement, keep dtypes
            df[common_cols] = upd[common_cols]
        else:
            aligned = upd.reindex(index=common_index, columns=common_cols)
            df.loc[common_index, common_cols] = aligned


def reattach_meta(src: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Copy _schema (pruned) and _splits from src to out."""
    if not isinstance(src, pd.DataFrame) or not isinstance(out, pd.DataFrame):
        raise ValueError("Both arguments must be pandas DataFrames.")

    schema = src.attrs.get("_schema")
    splits = src.attrs.get("_splits")

    if schema is not None:
        out.attrs["_schema"] = schema.clone_pruned(out.columns)
    if splits is not None:
        out.attrs["_splits"] = splits

    return out
