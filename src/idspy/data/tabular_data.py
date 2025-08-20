from dataclasses import dataclass, field
from typing import Sequence, Optional, Tuple, Dict, Any, Union

import pandas as pd

from .data import Data, DataView, IndexLike


@dataclass(frozen=True)
class TabularSchema:
    """Declarative schema describing a tabular ML dataset."""
    target: str
    numeric: Sequence[str] = field(default_factory=tuple)
    categorical: Sequence[str] = field(default_factory=tuple)
    extra: Sequence[str] = field(default_factory=tuple)

    @property
    def features(self) -> Tuple[str, ...]:
        """All feature columns (numeric + categorical), in that order."""
        return tuple(self.numeric) + tuple(self.categorical)

    @property
    def all_columns(self) -> Tuple[str, ...]:
        """All schema columns in the order: features, extra, target."""
        return self.features + tuple(self.extra) + (self.target,)

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate that the DataFrame contains the referenced columns
        and that schema partitions don't overlap.
        """
        cols = set(df.columns)
        missing = [c for c in self.all_columns if c not in cols]
        if missing:
            raise ValueError(f"Schema references missing columns: {missing}")

        num_set, cat_set, extra_set = set(self.numeric), set(self.categorical), set(self.extra)

        overlap_nc = num_set & cat_set
        if overlap_nc:
            raise ValueError(f"Columns in both numeric and categorical: {sorted(overlap_nc)}")

        if self.target in (num_set | cat_set | extra_set):
            raise ValueError(f"Target column '{self.target}' must not appear in features or extras.")

        overlap_extra = extra_set & (num_set | cat_set)
        if overlap_extra:
            raise ValueError(f"Extras overlap feature columns: {sorted(overlap_extra)}")

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the schema (lists for sequences)."""
        return {
            "target": self.target,
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "extra": list(self.extra),
        }


def coerce_single_column_frame(name: str, obj: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Ensure we have a single-column DataFrame named `name`.
    Accepts a Series (renamed) or a single-column DataFrame (column renamed if needed).
    """
    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=name)
    elif isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"Expected a single-column DataFrame for '{name}', got {obj.shape[1]} columns.")
        col = obj.columns[0]
        df = obj.rename(columns={col: name}) if col != name else obj
    else:
        raise TypeError(f"Expected pandas Series or single-column DataFrame for '{name}'.")
    return df


@dataclass(eq=False)
class TabularData(Data):
    """
    Data container with an optional TabularSchema attached.

    All Data operations are available; when a schema is set, convenience
    accessors allow selecting numeric/categorical/features/target/extras.
    """
    _schema: Optional[TabularSchema] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self._schema is not None:
            self._schema.validate(self._df)

    def _require_schema(self) -> TabularSchema:
        if self._schema is None:
            raise ValueError("Schema is not set.")
        return self._schema

    @property
    def schema(self) -> Optional[TabularSchema]:
        return self._schema

    @schema.setter
    def schema(self, new_schema: Optional[TabularSchema]) -> None:
        if new_schema is not None and not isinstance(new_schema, TabularSchema):
            raise TypeError("'schema' must be a TabularSchema (or None).")

        self._schema = new_schema
        if new_schema is not None:
            new_schema.validate(self._df)

    @property
    def numeric(self) -> pd.DataFrame:
        sch = self._require_schema()
        if not sch.numeric:
            raise ValueError("Schema defines no numeric columns.")
        return self.get_df(columns=list(sch.numeric))

    @numeric.setter
    def numeric(self, new_df: pd.DataFrame) -> None:
        sch = self._require_schema()
        if not sch.numeric:
            raise ValueError("Schema defines no numeric columns to set.")
        if not new_df.columns.equals(pd.Index(sch.numeric)):
            raise ValueError("Schema defined numeric columns differ from 'new_df' columns.")

        self.set_df(new_df)

    @property
    def categorical(self) -> pd.DataFrame:
        sch = self._require_schema()
        if not sch.categorical:
            raise ValueError("Schema defines no categorical columns.")
        return self.get_df(columns=list(sch.categorical))

    @categorical.setter
    def categorical(self, new_df: pd.DataFrame) -> None:
        sch = self._require_schema()
        if not sch.categorical:
            raise ValueError("Schema defines no categorical columns to set.")
        if not new_df.columns.equals(pd.Index(sch.categorical)):
            raise ValueError("Schema defined categorical columns differ from 'new_df' columns.")

        self.set_df(new_df)

    @property
    def features(self) -> pd.DataFrame:
        sch = self._require_schema()
        if not sch.features:
            raise ValueError("Schema defines no feature columns.")
        return self.get_df(columns=list(sch.features))

    @features.setter
    def features(self, new_df: pd.DataFrame) -> None:
        sch = self._require_schema()
        if not sch.features:
            raise ValueError("Schema defines no feature columns to set.")
        if not new_df.columns.equals(pd.Index(sch.features)):
            raise ValueError("Schema defined features columns differ from 'new_df' columns.")

        self.set_df(new_df)

    @property
    def target(self) -> pd.Series:
        sch = self._require_schema()
        if not sch.target:
            raise ValueError("Schema target is empty.")
        return self.get_df(columns=sch.target)

    @target.setter
    def target(self, new: Union[pd.Series, pd.DataFrame]) -> None:
        sch = self._require_schema()
        if not sch.target:
            raise ValueError("Schema target is empty.")
        new = coerce_single_column_frame(sch.target, new)

        if not new.columns.equals(pd.Index(sch.target)):
            raise ValueError("Schema defined target column differ from 'new_df' columns.")

        self.set_df(new)

    @property
    def extras(self) -> pd.DataFrame:
        sch = self._require_schema()
        if not sch.extra:
            raise ValueError("Schema defines no extra columns.")
        return self.get_df(columns=list(sch.extra))

    @extras.setter
    def extras(self, new_df: pd.DataFrame) -> None:
        sch = self._require_schema()
        if not sch.extra:
            raise ValueError("Schema defines no extra columns to set.")
        if not new_df.columns.equals(pd.Index(sch.extra)):
            raise ValueError("Schema defined extra columns differ from 'new_df' columns.")

        self.set_df(new_df)

    def view(
            self,
            index: Optional[IndexLike] = None,
            strict: bool = True,
    ) -> "TabularView":
        """
        Create a view limited to `index`. If `strict`, ensure all requested
        indices exist in the base index.
        """
        return TabularView(parent=self, _index=self._scope_from(index, strict))


@dataclass(eq=False)
class TabularView(DataView):
    """
    A view over TabularData that respects the parent's schema and index scope.
    """
    parent: TabularData

    def _require_schema(self) -> TabularSchema:
        sch = self.parent.schema
        if sch is None:
            raise ValueError("Schema is not set.")
        return sch

    @property
    def schema(self) -> Optional[TabularSchema]:
        return self.parent.schema

    @schema.setter
    def schema(self, new_schema: Optional[TabularSchema]) -> None:
        self.parent.schema = new_schema

    @property
    def numeric(self) -> pd.DataFrame:
        sch = self._require_schema()
        if not sch.numeric:
            raise ValueError("Schema defines no numeric columns.")
        return self.parent.get_df(index=self.index, columns=list(sch.numeric))

    @numeric.setter
    def numeric(self, new_df: pd.DataFrame) -> None:
        sch = self._require_schema()
        if not sch.numeric:
            raise ValueError("Schema defines no numeric columns to set.")
        self.parent.set_df(new_df, index=self.index)

    @property
    def categorical(self) -> pd.DataFrame:
        sch = self._require_schema()
        if not sch.categorical:
            raise ValueError("Schema defines no categorical columns.")
        return self.parent.get_df(index=self.index, columns=list(sch.categorical))

    @categorical.setter
    def categorical(self, new_df: pd.DataFrame) -> None:
        sch = self._require_schema()
        if not sch.categorical:
            raise ValueError("Schema defines no categorical columns to set.")
        self.parent.set_df(new_df, index=self.index)

    @property
    def features(self) -> pd.DataFrame:
        sch = self._require_schema()
        if not sch.features:
            raise ValueError("Schema defines no feature columns.")
        return self.parent.get_df(index=self.index, columns=list(sch.features))

    @features.setter
    def features(self, new_df: pd.DataFrame) -> None:
        sch = self._require_schema()
        if not sch.features:
            raise ValueError("Schema defines no feature columns to set.")
        self.parent.set_df(new_df, index=self.index)

    @property
    def target(self) -> pd.Series:
        sch = self._require_schema()
        if not sch.target:
            raise ValueError("Schema target is empty.")
        return self.parent.get_df(index=self.index, columns=sch.target)

    @target.setter
    def target(self, new: Union[pd.Series, pd.DataFrame]) -> None:
        sch = self._require_schema()
        if not sch.target:
            raise ValueError("Schema target is empty.")
        new = coerce_single_column_frame(sch.target, new)
        self.parent.set_df(new, index=self.index)

    @property
    def extras(self) -> pd.DataFrame:
        sch = self._require_schema()
        if not sch.extra:
            raise ValueError("Schema defines no extra columns.")
        return self.parent.get_df(index=self.index, columns=list(sch.extra))

    @extras.setter
    def extras(self, new_df: pd.DataFrame) -> None:
        sch = self._require_schema()
        if not sch.extra:
            raise ValueError("Schema defines no extra columns to set.")
        self.parent.set_df(new_df, index=self.index)

    def materialize(self) -> TabularData:
        """Freeze this view into a standalone TabularData object."""
        new_df = self.parent.get_df(index=self._eff_index, copy=False)
        return TabularData(new_df, self.schema)
