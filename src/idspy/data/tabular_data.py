from dataclasses import dataclass, field
from typing import Sequence, Iterable, Optional, Tuple, Any, Union

import pandas as pd

IndexLike = Union[pd.Index, Iterable[Any], pd.Series]


def _to_index(index_like: IndexLike) -> pd.Index:
    """
    Convert to pd.Index.
    - Boolean Series: use the indices where the value is True.
    - Otherwise: build an Index from the values.
    """
    if isinstance(index_like, pd.Series):
        if index_like.dtype == bool:
            return index_like[index_like].index
        return pd.Index(index_like)
    return pd.Index(index_like)


@dataclass(frozen=True)
class TabularSchema:
    target: str
    numeric: Sequence[str] = field(default_factory=tuple)
    categorical: Sequence[str] = field(default_factory=tuple)
    extra: Sequence[str] = field(default_factory=tuple)

    @property
    def features(self) -> Tuple[str, ...]:
        return tuple(self.numeric) + tuple(self.categorical)

    @property
    def all_columns(self) -> Tuple[str, ...]:
        return self.features + tuple(self.extra) + (self.target,)

    def validate(self, df: pd.DataFrame) -> None:
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

    def to_dict(self) -> dict:
        """
        Convert the TabularSchema into a dictionary representation.
        """
        return {
            "target": self.target,
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "extra": list(self.extra),
        }


@dataclass
class TabularData:
    _base: pd.DataFrame
    _schema: Optional[TabularSchema] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._schema is not None:
            self._schema.validate(self._base)

    def _require_schema(self) -> TabularSchema:
        if self._schema is None:
            raise ValueError("Schema is not set.")
        return self._schema

    @property
    def schema(self) -> Optional[TabularSchema]:
        return self._schema

    @schema.setter
    def schema(self, new_schema: TabularSchema) -> None:
        self._schema = new_schema
        if new_schema is not None:
            new_schema.validate(self._base)

    @property
    def index(self) -> pd.Index:
        return self._base.index

    @property
    def data(self) -> pd.DataFrame:
        return self._base

    # numeric
    @property
    def numeric(self) -> pd.DataFrame:
        sch = self._require_schema()
        return self._base.loc[:, list(sch.numeric)] if sch.numeric else self._base.loc[:, []]

    @numeric.setter
    def numeric(self, value) -> None:
        sch = self._require_schema()
        if not sch.numeric:
            raise ValueError("Schema defines no numeric columns to set.")
        self._base.loc[:, list(sch.numeric)] = value

    # categorical
    @property
    def categorical(self) -> pd.DataFrame:
        sch = self._require_schema()
        return self._base.loc[:, list(sch.categorical)] if sch.categorical else self._base.loc[:, []]

    @categorical.setter
    def categorical(self, value) -> None:
        sch = self._require_schema()
        if not sch.categorical:
            raise ValueError("Schema defines no categorical columns to set.")
        self._base.loc[:, list(sch.categorical)] = value

    # features
    @property
    def features(self) -> pd.DataFrame:
        sch = self._require_schema()
        return self._base.loc[:, list(sch.features)]

    @features.setter
    def features(self, value) -> None:
        sch = self._require_schema()
        self._base.loc[:, list(sch.features)] = value

    # target
    @property
    def target(self) -> pd.Series:
        sch = self._require_schema()
        return self._base.loc[:, sch.target]

    @target.setter
    def target(self, value) -> None:
        sch = self._require_schema()
        self._base.loc[:, sch.target] = value

    # extras
    @property
    def extras(self) -> pd.DataFrame:
        sch = self._require_schema()
        if not sch.extra:
            raise ValueError("Schema does not define extras.")
        return self._base.loc[:, list(sch.extra)]

    @extras.setter
    def extras(self, value) -> None:
        sch = self._require_schema()
        if not sch.extra:
            raise ValueError("Schema does not define extras.")
        self._base.loc[:, list(sch.extra)] = value

    def to_dataframe(self, columns: Optional[Iterable[str]] = None, materialize: bool = False) -> pd.DataFrame:
        df = self._base if columns is None else self._base.loc[:, list(columns)]
        return df.copy(deep=True) if materialize else df

    def view(self, index_like: IndexLike, strict: bool = True) -> "TabularView":
        idx = _to_index(index_like)
        if strict:
            missing = idx.difference(self._base.index)
            if len(missing) > 0:
                raise ValueError(f"Some indices are not in base.index: {missing.tolist()}")
            eff = idx
        else:
            eff = idx.intersection(self._base.index)
        return TabularView(parent=self, index=eff)

    def view_from_query(self, expr: str) -> "TabularView":
        return TabularView(parent=self, index=self._base.query(expr).index)

    def view_from_mask(self, mask: pd.Series) -> "TabularView":
        if not mask.index.equals(self._base.index):
            raise ValueError("Mask must be aligned to base.index.")
        return TabularView(parent=self, index=self._base.index[mask])

    def from_index(self, index_like: IndexLike) -> "TabularData":
        return self.view(index_like).materialize()

    def __len__(self) -> int:
        return len(self._base.index)


@dataclass
class TabularView:
    parent: TabularData
    index: pd.Index

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", pd.Index(self.index))
        missing = self.index.difference(self.parent.data.index)
        if len(missing) > 0:
            raise ValueError(f"View index not present in parent: {missing.tolist()}")

    def _require_schema(self) -> TabularSchema:
        sch = self.parent.schema
        if sch is None:
            raise ValueError("Schema is not set.")
        return sch

    @property
    def schema(self) -> Optional[TabularSchema]:
        return self.parent.schema

    # numeric
    @property
    def numeric(self) -> pd.DataFrame:
        sch = self._require_schema()
        return self.parent.data.loc[self.index, list(sch.numeric)] if sch.numeric else self.parent.data.loc[
            self.index, []]

    @numeric.setter
    def numeric(self, value) -> None:
        sch = self._require_schema()
        if not sch.numeric:
            raise ValueError("Schema defines no numeric columns to set.")
        self.parent.data.loc[self.index, list(sch.numeric)] = value

    # categorical
    @property
    def categorical(self) -> pd.DataFrame:
        sch = self._require_schema()
        return self.parent.data.loc[self.index, list(sch.categorical)] if sch.categorical else self.parent.data.loc[
            self.index, []]

    @categorical.setter
    def categorical(self, value) -> None:
        sch = self._require_schema()
        if not sch.categorical:
            raise ValueError("Schema defines no categorical columns to set.")
        self.parent.data.loc[self.index, list(sch.categorical)] = value

    # features
    @property
    def features(self) -> pd.DataFrame:
        sch = self._require_schema()
        return self.parent.data.loc[self.index, list(sch.features)]

    @features.setter
    def features(self, value) -> None:
        sch = self._require_schema()
        self.parent.data.loc[self.index, list(sch.features)] = value

    # target
    @property
    def target(self) -> pd.Series:
        sch = self._require_schema()
        return self.parent.data.loc[self.index, sch.target]

    @target.setter
    def target(self, value) -> None:
        sch = self._require_schema()
        self.parent.data.loc[self.index, sch.target] = value

    # extras
    @property
    def extras(self) -> pd.DataFrame:
        sch = self._require_schema()
        if not sch.extra:
            raise ValueError("Schema does not define extras.")
        return self.parent.data.loc[self.index, list(sch.extra)]

    @extras.setter
    def extras(self, value) -> None:
        sch = self._require_schema()
        if not sch.extra:
            raise ValueError("Schema does not define extras.")
        self.parent.data.loc[self.index, list(sch.extra)] = value

    def to_dataframe(self, columns: Optional[Iterable[str]] = None, materialize: bool = False) -> pd.DataFrame:
        df = self.parent.data.loc[self.index] if columns is None else self.parent.data.loc[self.index, list(columns)]
        return df.copy(deep=True) if materialize else df

    def query(self, expr: str) -> "TabularView":
        new_idx = self.parent.data.loc[self.index].query(expr).index
        return TabularView(parent=self.parent, index=new_idx)

    def mask(self, mask: pd.Series) -> "TabularView":
        if not mask.index.equals(self.parent.data.index):
            raise ValueError("Mask is not aligned to parent._base.index.")
        new_idx = self.index.intersection(self.parent.data.index[mask])
        return TabularView(parent=self.parent, index=new_idx)

    def intersect(self, other: "TabularView") -> "TabularView":
        if other.parent is not self.parent:
            raise ValueError("Cannot intersect views with different parents.")
        return TabularView(parent=self.parent, index=self.index.intersection(other.index))

    def union(self, other: "TabularView") -> "TabularView":
        if other.parent is not self.parent:
            raise ValueError("Cannot union views with different parents.")
        return TabularView(parent=self.parent, index=self.index.union(other.index))

    def difference(self, other: "TabularView") -> "TabularView":
        if other.parent is not self.parent:
            raise ValueError("Cannot difference views with different parents.")
        return TabularView(parent=self.parent, index=self.index.difference(other.index))

    def materialize(self) -> TabularData:
        new_base = self.parent.data.loc[self.index].copy(deep=True)
        return TabularData(_base=new_base, _schema=self.parent.schema)

    def __len__(self) -> int:
        return len(self.index)
