from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

from .schema import Schema, ColumnRole
from .partition import Partition, PartitionName
from ..common.profiler import time_profiler

#Descrizione precisa accurata chiara e semplice di ciò che fa la classe TabAccessor
#La classe TabAccessor è un'estensione personalizzata di un DataFrame di pandas 
#che aggiunge funzionalità specifiche per la gestione dello schema dei dati e delle partizioni.
#Essa consente di definire e manipolare i ruoli delle colonne (come numeriche, categoriali, target e features) 
# e di gestire partizioni dei dati (come training, validation e test) direttamente all'interno del DataFrame.
#Inoltre, fornisce viste filtrate del DataFrame basate sui ruoli delle colonne e sulle partizioni,
# mantenendo sempre la coerenza con lo schema e le partizioni definite.
#i metodi che in questa classe svologono i ruoli principali sono:
#- set_schema: per definire o aggiornare lo schema delle colonne.
#- add_role e update_role: per aggiungere o modificare i ruoli delle colonne.
#- set_partitions_from_labels e set_partitions_from_positions: per definire le partizioni dei dati.
#- get_partition: per ottenere una vista del DataFrame basata su una partizione specifica.
#- proprietà come features, target, numerical e categorical per ottenere viste filtrate del DataFrame.

@register_dataframe_accessor("tab")
class TabAccessor:
    """Schema + partition accessor."""

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

        if "_schema" not in self._obj.attrs:
            self._obj.attrs["_schema"] = Schema()
        if "_partitions" not in self._obj.attrs:
            self._obj.attrs["_partitions"] = Partition()

    def get_meta(self) -> Dict[str, Union[Schema, Partition]]:
        """Get schema and partitions as a dictionary."""
        return {"_schema": self.schema, "_partitions": self.partitions}

    def load_meta(self, meta: Dict[str, Union[Schema, Partition]]) -> pd.DataFrame:
        """Load schema and partitions from a dictionary."""
        if "_schema" in meta and isinstance(meta["_schema"], Schema):
            self._obj.attrs["_schema"] = meta["_schema"]
        if "_partitions" in meta and isinstance(meta["_partitions"], Partition):
            self._obj.attrs["_partitions"] = meta["_partitions"]
        return self._obj

    def _get_columns_for_role(self, role: Union[ColumnRole, str]) -> List[str]:
        """Get columns for role."""
        # Prune schema to keep it in sync with current DataFrame columns
        self.schema.prune_missing(self._obj.columns)
        return self.schema.columns(role)

    def _get_view_for_role(
        self, role: Union[ColumnRole, str]
    ) -> Optional[pd.DataFrame]:
        """Get a view for a specific role with metadata attached."""
        cols = self._get_columns_for_role(role)
        if cols:
            out = self._obj[cols]
            return reattach_meta(self._obj, out)
        return None

    @property
    def schema(self) -> Schema:
        """Get the schema from DataFrame attrs."""
        return self._obj.attrs["_schema"]

    @property
    def partitions(self) -> Partition:
        """Get the partitions from DataFrame attrs."""
        return self._obj.attrs["_partitions"]

    def set_schema(
        self, schema: Optional["Schema"] = None, **roles: List[str]
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

    def set_partitions_from_labels(self, mapping: Dict[str, Iterable]) -> pd.DataFrame:
        """Define partitions from index labels."""
        sp = Partition()
        sp.set_from_labels(mapping)
        self._obj.attrs["_partitions"] = sp
        return self._obj

    def set_partitions_from_positions(
        self, mapping: Dict[str, Iterable[int]]
    ) -> pd.DataFrame:
        """Define partitions from integer positions."""
        sp = Partition()
        sp.set_from_positions(mapping, self._obj.index)
        self._obj.attrs["_partitions"] = sp
        return self._obj

    def get_partition(self, name: str) -> pd.DataFrame:
        """Return dataframe slice for a partition with metadata attached."""
        if name not in self.partitions.mapping:
            raise KeyError(f"Partition '{name}' is not defined.")

        split_index = self.partitions.mapping[name]
        if split_index.equals(self._obj.index):
            out = self._obj
        else:
            mask = self._obj.index.isin(split_index)
            out = self._obj[mask]

        return reattach_meta(self._obj, out)

    @property
    def train(self) -> pd.DataFrame:
        return self.get_partition(PartitionName.TRAIN.value)

    @train.setter
    def train(self, updated: pd.DataFrame) -> None:
        split_index = self.partitions.mapping.get(PartitionName.TRAIN.value)
        if split_index is not None:
            mask = self._obj.index.isin(split_index)
            self._assign_block(updated, mask=mask, cols=None)

    @property
    def val(self) -> pd.DataFrame:
        return self.get_partition(PartitionName.VAL.value)

    @val.setter
    def val(self, updated: pd.DataFrame) -> None:
        split_index = self.partitions.mapping.get(PartitionName.VAL.value)
        if split_index is not None:
            mask = self._obj.index.isin(split_index)
            self._assign_block(updated, mask=mask, cols=None)

    @property
    def test(self) -> pd.DataFrame:
        return self.get_partition(PartitionName.TEST.value)

    @test.setter
    def test(self, updated: pd.DataFrame) -> None:
        split_index = self.partitions.mapping.get(PartitionName.TEST.value)
        if split_index is not None:
            mask = self._obj.index.isin(split_index)
            self._assign_block(updated, mask=mask, cols=None)

    # @time_profiler
    def _assign_block(
        self,
        updated: Union[pd.DataFrame, pd.Series],
        cols: Optional[Union[List[str], str]] = None,
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

        if cols is None:
            target_cols = df.columns
        elif isinstance(cols, str):
            target_cols = pd.Index([cols])
        else:
            target_cols = pd.Index(cols)

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
            # Convert only columns with different dtypes
            src_dtypes = df.dtypes[common_cols]
            upd_dtypes = upd.dtypes[common_cols]

            for col in common_cols:
                if src_dtypes[col] != upd_dtypes[col]:
                    try:
                        df[col] = df[col].astype(upd_dtypes[col])
                    except Exception as e:
                        raise ValueError(
                            f"Could not convert dtype for column '{col}' from "
                            f"{src_dtypes[col]} to {upd_dtypes[col]}."
                        ) from e

            aligned = upd.reindex(index=common_index, columns=common_cols)
            df.loc[common_index, common_cols] = aligned


def reattach_meta(
    src: Union[pd.DataFrame, pd.Series], out: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """Copy _schema and _partitions from src to out as direct references."""
    if not isinstance(src, (pd.DataFrame, pd.Series)) or not isinstance(
        out, (pd.DataFrame, pd.Series)
    ):
        raise ValueError("Both arguments must be pandas DataFrames or Series.")

    schema = src.attrs.get("_schema")
    splits = src.attrs.get("_partitions")

    if schema is not None:
        out.attrs["_schema"] = schema
    if splits is not None:
        out.attrs["_partitions"] = splits

    return out
