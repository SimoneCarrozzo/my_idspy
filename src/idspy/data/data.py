import weakref
from dataclasses import dataclass, field
from functools import cached_property
from typing import Union, Iterable, Any, Optional, Sequence

import pandas as pd

IndexLike = Union[pd.Index, Iterable[Any], pd.Series]


def to_index(index: IndexLike) -> pd.Index:
    """Normalize various index-like inputs to a pandas Index.

    Note: boolean Series are NOT accepted here (they need an alignment context).
    Use view_from_mask() instead when providing boolean masks.
    """
    if isinstance(index, pd.Index):
        return index
    if isinstance(index, pd.Series):
        if index.dtype == bool:
            raise TypeError("Boolean masks are not accepted by to_index(). Use view_from_mask().")
        return pd.Index(index)
    return pd.Index(index)


@dataclass
class Data:
    """
    Container around a DataFrame that supports creating 'views'
    (index-scoped windows) and invalidating them when the base changes.
    """
    _df: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    _views: "weakref.WeakSet[DataView]" = field(
        init=False, repr=False, default_factory=weakref.WeakSet
    )

    def __post_init__(self) -> None:
        self._df = self._normalize_df(self._df)

    @property
    def index(self) -> pd.Index:
        return self._df.index

    @property
    def df(self) -> pd.DataFrame:
        """Return a deep copy of the normalized DataFrame (safe to mutate)."""
        return self._df.copy(deep=True)

    @df.setter
    def df(self, new_df: pd.DataFrame) -> None:
        self._df = self._normalize_df(new_df)
        self._invalidate_views()

    def get_df(
            self,
            index: Optional[IndexLike] = None,
            columns: Optional[Any] = None,
            copy: bool = False
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Return the whole DataFrame or a .loc-selection by index and columns.

        `columns` can be any valid .loc column selector (label(s), slice, mask, callable).
        """
        idx = to_index(index) if index is not None else self._df.index
        cols = columns if columns is not None else slice(None)
        out = self._df.loc[idx, cols]
        return out.copy(deep=True) if copy else out

    def set_df(
            self,
            new_df: pd.DataFrame,
            index: Optional[IndexLike] = None,
            columns: Optional[Union[pd.Index, Sequence]] = None,
    ) -> None:
        """
        Overwrite rows in the DataFrame at `index` and `columns` with `new_df`.

        - Forbids accidental creation of new rows.
        - Aligns/expands columns without unexpected dtype upcasting.
        """
        base = self._df
        idx = to_index(index) if index is not None else base.index
        incoming = self._normalize_df(new_df)
        target_cols = pd.Index(columns) if columns is not None else incoming.columns

        missing_rows = pd.Index(idx).difference(base.index)
        if len(missing_rows):
            raise ValueError(f"Index contains unknown labels: {missing_rows.tolist()}")

        # Ensure both frames have the necessary columns without dtype surprises
        base = base.reindex(columns=base.columns.union(target_cols))
        incoming = incoming.reindex(columns=target_cols)

        # Align incoming index to target index if necessary
        if not incoming.index.equals(idx):
            if len(incoming) != len(idx):
                raise ValueError(
                    "Length mismatch: new_df must match target index length "
                    "or share the same index."
                )
            incoming.index = pd.Index(idx)

        base.loc[idx, target_cols] = incoming[target_cols]
        self._df = self._normalize_df(base)
        self._invalidate_views()

    def refresh_views(self) -> None:
        self._invalidate_views()

    def register_view(self, view: "DataView") -> None:
        self._views.add(view)

    def _invalidate_views(self) -> None:
        for v in list(self._views):
            try:
                v.invalidate()
            except Exception as e:
                raise e

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a DataFrame with a unique index and deep copy."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not df.index.is_unique:
            raise ValueError("df.index must be unique")
        return df.copy(deep=True)

    def _scope_from(self, index: Optional[IndexLike], strict: bool) -> pd.Index:
        """
        Resolve a requested `index` against the current base index.
        - If `index` is None: returns the full base index.
        - Otherwise: normalizes to an Index and intersects with the base.
        - If `strict` is True, raises if any requested labels are missing.
        """
        base_idx = self._df.index
        if index is None:
            return base_idx

        idx = to_index(index)
        if strict:
            missing = idx.difference(base_idx)
            if len(missing):
                raise ValueError(
                    f"Some indices are not in base index: {missing.tolist()}"
                )
        return idx.intersection(base_idx)

    def view(
            self,
            index: Optional[IndexLike] = None,
            strict: bool = True,
    ) -> "DataView":
        """
        Create a view limited to `index`. If `strict`, ensure all requested
        indices exist in the base index.
        """
        return DataView(parent=self, _index=self._scope_from(index, strict))

    def view_from_query(
            self,
            expr: str,
            index: Optional[IndexLike] = None
    ) -> "DataView":
        """Create a view by filtering with pandas.DataFrame.query()."""
        base_idx = self._df.index
        scope_df = self._df if index is None else self._df.loc[
            to_index(index).intersection(base_idx)
        ]

        eff = scope_df.query(expr).index
        return self.view(strict=False, index=eff)

    def view_from_mask(
            self,
            mask: pd.Series,
            index: Optional[IndexLike] = None
    ) -> "DataView":
        """Create a view from a boolean mask, properly aligned to the scope."""
        if not isinstance(mask, pd.Series) or mask.dtype != bool:
            raise TypeError("mask must be a boolean pandas Series")

        base_idx = self._df.index
        scope_idx = base_idx if index is None else to_index(index).intersection(base_idx)

        if mask.index.equals(scope_idx):
            eff = scope_idx[mask]
        elif mask.index.equals(base_idx):
            eff = scope_idx[mask.loc[scope_idx]]
        else:
            raise ValueError("Mask must be aligned to scope (base_index) or to parent.index.")
        return self.view(strict=False, index=eff)

    def __len__(self) -> int:
        return self._df.shape[0]


@dataclass(eq=False)
class DataView:
    """A windowed view over a parent's index."""
    parent: Data
    _index: pd.Index
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self._index = pd.Index(self._index)
        self.parent.register_view(self)

    @cached_property
    def _eff_index(self) -> pd.Index:
        """Effective index = requested index ∩ current parent index."""
        base_idx = self.parent.index
        return self._index[self._index.isin(base_idx)]

    @property
    def index(self) -> pd.Index:
        return self._eff_index

    def invalidate(self) -> None:
        """Drop cached effective index; next access will recompute."""
        self.__dict__.pop("_eff_index", None)

    @property
    def df(self) -> pd.DataFrame:
        """Return a copy of the parent df restricted to the effective index."""
        return self.parent.get_df(index=self._eff_index, copy=True)

    @df.setter
    def df(self, new_df: pd.DataFrame) -> None:
        self.parent.set_df(new_df, self._eff_index)

    def get_df(
            self,
            index: Optional[IndexLike] = None,
            columns: Optional[Any] = None,
            copy: bool = False
    ) -> pd.DataFrame:
        """
        Return the whole DataFrame or a .loc-selection by index and columns.

        `columns` can be any valid .loc column selector (label(s), slice, mask, callable).
        """
        idx = to_index(index).intersection(self._eff_index) if index is not None else self._eff_index
        return self.parent.get_df(index=idx, columns=columns, copy=copy)

    def set_df(
            self,
            new_df: pd.DataFrame,
            index: Optional[IndexLike] = None,
            columns: Optional[Union[pd.Index, Sequence]] = None,
    ) -> None:
        """
        Overwrite rows in the DataFrame at `index` and `columns` with `new_df`.

        - Forbids accidental creation of new rows.
        - Aligns/expands columns without unexpected dtype upcasting.
        """
        idx = to_index(index).intersection(self._eff_index) if index is not None else self._eff_index
        self.parent.set_df(new_df, index=idx, columns=columns)

    def view(self, index_like: IndexLike, strict: bool = True) -> "DataView":
        """Create a subview within this view's scope."""
        idx = to_index(index_like)
        if strict:
            missing = idx.difference(self._eff_index)
            if len(missing) > 0:
                raise ValueError(
                    f"Some indices are not in this view's index: {missing.tolist()}"
                )
        scope = idx.intersection(self._eff_index)
        return self.parent.view(strict=strict, index=scope)

    def view_from_query(self, expr: str) -> "DataView":
        return self.parent.view_from_query(expr, index=self._eff_index)

    def view_from_mask(self, mask: pd.Series) -> "DataView":
        return self.parent.view_from_mask(mask, index=self._eff_index)

    def intersect(self, other: "DataView") -> "DataView":
        if other.parent is not self.parent:
            raise ValueError("Cannot intersect views with different parents.")
        return self.parent.view(strict=False, index=self._eff_index.intersection(other._eff_index))

    def union(self, other: "DataView") -> "DataView":
        if other.parent is not self.parent:
            raise ValueError("Cannot union views with different parents.")
        return self.parent.view(strict=False, index=self._eff_index.union(other._eff_index))

    def difference(self, other: "DataView") -> "DataView":
        if other.parent is not self.parent:
            raise ValueError("Cannot difference views with different parents.")
        return self.parent.view(strict=False, index=self._eff_index.difference(other._eff_index))

    def materialize(self) -> Data:
        """Freeze this view into a standalone Data object."""
        new_df = self.parent.get_df(index=self._eff_index, copy=False)
        return Data(new_df)

    def __len__(self) -> int:
        return len(self._eff_index)
