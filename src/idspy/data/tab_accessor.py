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

        if "_schema" not in self._obj.attrs:
            self._obj.attrs["_schema"] = Schema()
        if "_splits" not in self._obj.attrs:
            self._obj.attrs["_splits"] = Split()

    def _get_columns_for_role(self, role: Union[ColumnRole, str]) -> List[str]:
        """Get columns for role."""
        # Prune schema to keep it in sync with current DataFrame columns
        self.schema.prune_missing(self._obj.columns)
        return self.schema.columns(role)

    def _get_view_for_role(self, role: Union[ColumnRole, str]) -> pd.DataFrame:
        """Get a view for a specific role with metadata attached."""
        cols = self._get_columns_for_role(role)
        out = self._obj[cols]
        return reattach_meta(self._obj, out)

    @property
    def schema(self) -> Schema:
        """Get the schema from DataFrame attrs."""
        return self._obj.attrs["_schema"]

    @property
    def splits(self) -> Split:
        """Get the splits from DataFrame attrs."""
        return self._obj.attrs["_splits"]

    def set_schema(
        self, schema: Union["Schema", None] = None, **roles: List[str]
    ) -> pd.DataFrame:
        """Replace schema with new role definitions (string keys allowed)."""
        if schema is not None:
            sch = schema
        else:
            sch = Schema(strict=self.schema.strict)
            for role_name, cols in roles.items():
                sch.add(cols, ColumnRole.from_name(role_name))

        self._obj.attrs["_schema"] = sch
        return self._obj

    def add_role(
        self, cols: Union[Iterable[str], str], role: Union[ColumnRole, str]
    ) -> pd.DataFrame:
        """Add columns to a role."""
        self.schema.add(cols, role)
        return self._obj

    def update_role(
        self, cols: Union[Iterable[str], str], role: Union[ColumnRole, str]
    ) -> pd.DataFrame:
        """Replace columns for a role and clean conflicts."""
        self.schema.update(cols, role)
        return self._obj

    def columns(self, role: Union[ColumnRole, str]) -> List[str]:
        return self._get_columns_for_role(role)

    @property
    def features(self) -> pd.DataFrame:
        """Feature view (excludes target)."""
        return self._get_view_for_role(ColumnRole.FEATURES)

    @features.setter
    def features(self, updated: pd.DataFrame) -> None:
        cols = self._get_columns_for_role(ColumnRole.FEATURES)
        self._assign_block(updated, cols=cols)

    @property
    def target(self) -> pd.DataFrame:
        """Target view."""
        cols = self._get_columns_for_role(ColumnRole.TARGET)
        if not cols:
            raise ValueError("No target is defined in the schema.")
        return self._get_view_for_role(ColumnRole.TARGET)

    @target.setter
    def target(self, updated: pd.DataFrame) -> None:
        cols = self._get_columns_for_role(ColumnRole.TARGET)
        if not cols:
            raise ValueError("No target is defined in the schema.")
        self._assign_block(updated, cols=cols)

    @property
    def numerical(self) -> pd.DataFrame:
        """Numerical columns view."""
        return self._get_view_for_role(ColumnRole.NUMERICAL)

    @numerical.setter
    def numerical(self, updated: Union[pd.DataFrame, pd.Series]) -> None:
        cols = self._get_columns_for_role(ColumnRole.NUMERICAL)
        self._assign_block(updated, cols=cols)

    @property
    def categorical(self) -> pd.DataFrame:
        """Categorical columns view."""
        return self._get_view_for_role(ColumnRole.CATEGORICAL)

    @categorical.setter
    def categorical(self, updated: Union[pd.DataFrame, pd.Series]) -> None:
        cols = self._get_columns_for_role(ColumnRole.CATEGORICAL)
        self._assign_block(updated, cols=cols)

    def set_splits_from_labels(self, mapping: Dict[str, Iterable]) -> pd.DataFrame:
        """Define splits from index labels."""
        sp = Split()
        sp.set_from_labels(mapping)
        self._obj.attrs["_splits"] = sp
        return self._obj

    def set_splits_from_positions(
        self, mapping: Dict[str, Iterable[int]]
    ) -> pd.DataFrame:
        """Define splits from integer positions."""
        sp = Split()
        sp.set_from_positions(mapping, self._obj.index)
        self._obj.attrs["_splits"] = sp
        return self._obj

    def get_split(self, name: str) -> pd.DataFrame:
        """Return dataframe slice for a split with metadata attached."""
        if name not in self.splits.mapping:
            raise KeyError(f"Split '{name}' is not defined.")

        split_index = self.splits.mapping[name]
        if split_index.equals(self._obj.index):
            out = self._obj
        else:
            mask = self._obj.index.isin(split_index)
            out = self._obj[mask]

        return reattach_meta(self._obj, out)

    @property
    def train(self) -> pd.DataFrame:
        return self.get_split(SplitName.TRAIN.value)

    @train.setter
    def train(self, updated: pd.DataFrame) -> None:
        split_index = self.splits.mapping.get(SplitName.TRAIN.value)
        if split_index is not None:
            mask = self._obj.index.isin(split_index)
            self._assign_block(updated, mask=mask, cols=None)

    @property
    def val(self) -> pd.DataFrame:
        return self.get_split(SplitName.VAL.value)

    @val.setter
    def val(self, updated: pd.DataFrame) -> None:
        split_index = self.splits.mapping.get(SplitName.VAL.value)
        if split_index is not None:
            mask = self._obj.index.isin(split_index)
            self._assign_block(updated, mask=mask, cols=None)

    @property
    def test(self) -> pd.DataFrame:
        return self.get_split(SplitName.TEST.value)

    @test.setter
    def test(self, updated: pd.DataFrame) -> None:
        split_index = self.splits.mapping.get(SplitName.TEST.value)
        if split_index is not None:
            mask = self._obj.index.isin(split_index)
            self._assign_block(updated, mask=mask, cols=None)

    # @time_profiler
    def _assign_block(
        self,
        updated: Union[pd.DataFrame, pd.Series],
        cols: Optional[List[str]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Assign updated values aligned by index and columns.

        Args:
            updated: DataFrame or Series with new values
            cols: Column names to update
            mask: Boolean mask for row selection (preferred)
        """
        df = self._obj
        upd = updated.to_frame() if isinstance(updated, pd.Series) else updated

        target_index = df.index[mask] if mask is not None else df.index
        target_cols = df.columns if cols is None else pd.Index(cols)

        common_index = target_index.intersection(upd.index)
        if not len(common_index):
            raise ValueError(
                "No overlapping rows between destination and updated (index misaligned)."
            )

        common_cols = target_cols.intersection(upd.columns)
        if not len(common_cols):
            raise ValueError("No overlapping columns between destination and updated.")

        if mask is None and common_index.equals(df.index):
            # Full replacement, keep dtypes
            df[common_cols] = upd[common_cols]
        else:
            aligned = upd.reindex(index=common_index, columns=common_cols)
            df.loc[common_index, common_cols] = aligned


def reattach_meta(src: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Copy _schema and _splits from src to out as direct references."""
    if not isinstance(src, pd.DataFrame) or not isinstance(out, pd.DataFrame):
        raise ValueError("Both arguments must be pandas DataFrames.")

    schema = src.attrs.get("_schema")
    splits = src.attrs.get("_splits")

    if schema is not None:
        out.attrs["_schema"] = schema
    if splits is not None:
        out.attrs["_splits"] = splits

    return out
