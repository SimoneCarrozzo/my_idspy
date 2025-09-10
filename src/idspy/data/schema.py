from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Union

import pandas as pd


class ColumnRole(Enum):
    """Column roles tracked by the schema."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TARGET = "target"
    FEATURES = "features"

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
            ColumnRole.FEATURES: [],
        }
    )
    strict: bool = False

    def add(
        self, cols: Union[Iterable[str], str], role: Union[ColumnRole, str]
    ) -> None:
        """Add columns to a role, ensuring exclusivity and order."""
        role = ColumnRole.from_name(role)
        new_cols = self._as_list(cols)

        # Remove new_cols from all other roles except FEATURES
        for r in self.roles:
            if r != role and r != ColumnRole.FEATURES:
                self.roles[r] = [c for c in self.roles[r] if c not in new_cols]

        # Add to the specified role
        seen = set(self.roles[role])
        self.roles[role].extend([c for c in new_cols if c not in seen])

        # Auto-manage FEATURES role if not directly modifying it
        if role != ColumnRole.FEATURES:
            self._update_features_role()

    def update(
        self, cols: Union[Iterable[str], str], role: Union[ColumnRole, str]
    ) -> None:
        """Replace the columns of a role; remove them from other roles."""
        role = ColumnRole.from_name(role)
        new_cols = self._as_list(cols)

        # Remove new_cols from all other roles except FEATURES
        for r in self.roles:
            if r != role and r != ColumnRole.FEATURES:
                self.roles[r] = [c for c in self.roles[r] if c not in new_cols]

        # Set exact new list (order preserved, dedup)
        seen: set[str] = set()
        self.roles[role] = [c for c in new_cols if not (c in seen or seen.add(c))]

        # Auto-manage FEATURES role if not directly modifying it
        if role != ColumnRole.FEATURES:
            self._update_features_role()

    def _update_features_role(self) -> None:
        """Automatically update FEATURES role to include NUMERICAL + CATEGORICAL, excluding TARGET."""
        target_cols = set(self.roles[ColumnRole.TARGET])
        features_cols = []

        # Preserve order: first numerical, then categorical
        for r in (ColumnRole.NUMERICAL, ColumnRole.CATEGORICAL):
            features_cols.extend([c for c in self.roles[r] if c not in target_cols])

        self.roles[ColumnRole.FEATURES] = features_cols

    def columns(self, role: Union[ColumnRole, str]) -> List[str]:
        role = ColumnRole.from_name(role)
        return self.roles[role]

    def prune_missing(self, existing: pd.Index) -> None:
        """Remove columns not present in dataframe."""
        # Convert to set for O(1) lookup instead of O(n) for each column
        existing_set = set(existing)
        missing: List[str] = []

        for r, cols in self.roles.items():
            # Skip FEATURES role as it will be updated automatically
            if r == ColumnRole.FEATURES:
                continue

            # Use list comprehension with set lookup for better performance
            keep = [c for c in cols if c in existing_set]
            if len(keep) != len(cols):
                missing.extend([c for c in cols if c not in existing_set])
            self.roles[r] = keep

        # Update FEATURES role after pruning other roles
        self._update_features_role()

        if self.strict and missing:
            raise KeyError(f"Missing columns for schema: {missing}")
