from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Union

import pandas as pd


class ColumnRole(Enum):
    """Column roles tracked by the schema."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TARGET = "target"

    @classmethod
    def from_name(cls, name: Union[str, "ColumnRole"]) -> "ColumnRole":
        """Coerce a string or enum to ColumnRole."""
        if isinstance(name, ColumnRole):
            return name
        key = str(name).lower()
        for member in cls:
            if member.value == key:
                return member
        raise KeyError(f"Unknown role: {name}")


@dataclass(slots=True)
class Schema:
    """Store roles (types) of dataframe columns."""

    @staticmethod
    def _as_list(cols: Union[Iterable[str], str]) -> List[str]:
        """Normalize to list[str]."""
        if isinstance(cols, str):
            return [cols]
        return [c for c in cols]

    roles: Dict[ColumnRole, List[str]] = field(
        default_factory=lambda: {
            ColumnRole.NUMERICAL: [],
            ColumnRole.CATEGORICAL: [],
            ColumnRole.TARGET: [],
        }
    )
    strict: bool = False

    def add(self, cols: Union[Iterable[str], str], role: Union[ColumnRole, str]) -> None:
        """Add columns to a role, ensuring exclusivity and order."""
        role = ColumnRole.from_name(role)
        new_cols = self._as_list(cols)
        for r in self.roles:
            if r != role:
                self.roles[r] = [c for c in self.roles[r] if c not in new_cols]
        seen = set(self.roles[role])
        self.roles[role].extend([c for c in new_cols if c not in seen])

    def update(self, cols: Union[Iterable[str], str], role: Union[ColumnRole, str]) -> None:
        """Replace the columns of a role; remove them from other roles."""
        role = ColumnRole.from_name(role)
        new_cols = self._as_list(cols)
        # remove new_cols from all other roles
        for r in self.roles:
            if r != role:
                self.roles[r] = [c for c in self.roles[r] if c not in new_cols]
        # set exact new list (order preserved, dedup)
        seen: set[str] = set()
        self.roles[role] = [c for c in new_cols if not (c in seen or seen.add(c))]

    def columns(self, role: Union[ColumnRole, str]) -> List[str]:
        role = ColumnRole.from_name(role)
        return self.roles[role]

    def feature_columns(self) -> List[str]:
        """Return feature columns (exclude target)."""
        exclude = set(self.roles[ColumnRole.TARGET])
        ordered: List[str] = []
        for r in (ColumnRole.NUMERICAL, ColumnRole.CATEGORICAL):
            ordered.extend([c for c in self.roles[r] if c not in exclude])
        return ordered

    def prune_missing(self, existing: pd.Index) -> None:
        """Remove columns not present in dataframe."""
        missing: List[str] = []
        for r, cols in self.roles.items():
            keep = [c for c in cols if c in existing]
            if len(keep) != len(cols):
                missing.extend([c for c in cols if c not in existing])
            self.roles[r] = keep
        if self.strict and missing:
            raise KeyError(f"Missing columns for schema: {missing}")

    def clone_pruned(self, existing: pd.Index) -> "Schema":
        """Return a copy with roles filtered to existing_cols."""
        new = Schema(strict=self.strict)
        for r, cols in self.roles.items():
            new.roles[r] = [c for c in cols if c in existing]
        return new
