from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_EPS = 1e-9


class SplitName(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(slots=True)
class Split:
    """Hold train/val/test split definitions by index labels."""
    mapping: Dict[str, pd.Index] = field(default_factory=dict)

    def set_from_labels(
            self, name_to_labels: Dict[Union[str, "SplitName"], Iterable]
    ) -> None:
        """Define splits directly from index labels."""
        self.mapping = {str(k): pd.Index(v) for k, v in name_to_labels.items()}

    def set_from_positions(
            self,
            name_to_pos: Dict[Union[str, "SplitName"], Iterable[int]],
            index_labels: pd.Index,
    ) -> None:
        """Define splits from integer positions, converted to labels."""
        self.mapping = {
            str(k): pd.Index(index_labels[list(map(int, v))])
            for k, v in name_to_pos.items()
        }

    def indices_for(
            self, name: Union[str, "SplitName"], df: pd.DataFrame
    ) -> np.ndarray:
        """Return integer positions for a split, robust to reordering."""
        key = str(name)
        if key not in self.mapping:
            raise KeyError(f"Split '{key}' is not defined.")
        saved = self.mapping[key]
        valid = pd.Index(saved).intersection(df.index)
        return df.index.get_indexer_for(valid)


def _validate_split_sizes(
        train_size: float, val_size: float, test_size: float
) -> None:
    for nm, v in (("train_size", train_size), ("val_size", val_size), ("test_size", test_size)):
        if not (0.0 - _EPS <= v <= 1.0 + _EPS):
            raise ValueError(f"{nm} must be in [0, 1]; got {v}")

    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0, atol=_EPS):
        raise ValueError(f"train_size + val_size + test_size must sum to 1.0; got {total}")


def _ensure_index(x: Sequence, like: pd.Index) -> pd.Index:
    """Return a pd.Index with the same dtype as `like` if possible."""
    idx = pd.Index(x)
    try:
        return idx.astype(like.dtype, copy=False)
    except (TypeError, ValueError, RuntimeError):
        return idx


def _empty_like_index(like: pd.Index) -> pd.Index:
    return pd.Index([], dtype=like.dtype)


def _split_indices(
        df: pd.DataFrame,
        train_size: float,
        val_size: float,
        test_size: float,
        target: Optional[pd.Series] = None,
        random_state: Optional[int] = None,
        shuffle: bool = True,
) -> Dict[str, pd.Index]:
    """
    Core splitter: if `target` is provided, uses stratification; otherwise random split.
    Returns a mapping {'train': Index, 'val': Index, 'test': Index}.
    """
    _validate_split_sizes(train_size, val_size, test_size)

    if target is not None and len(target) != len(df):
        raise ValueError(
            f"Length of `target` ({len(target)}) must match number of rows in `df` ({len(df)})"
        )

    train_idx, remaining_idx = train_test_split(
        df.index,
        train_size=train_size,
        stratify=target,
        random_state=random_state,
        shuffle=shuffle,
    )

    train_idx = _ensure_index(train_idx, df.index)
    remaining_idx = _ensure_index(remaining_idx, df.index)

    # Distribute remaining between val/test
    rel = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.0

    if rel <= _EPS:
        return {
            SplitName.TRAIN.value: train_idx,
            SplitName.VAL.value: _empty_like_index(df.index),
            SplitName.TEST.value: remaining_idx,
        }

    if (1.0 - rel) <= _EPS:
        return {
            SplitName.TRAIN.value: train_idx,
            SplitName.VAL.value: remaining_idx,
            SplitName.TEST.value: _empty_like_index(df.index),
        }

    remaining_target = target[remaining_idx] if target is not None else None

    val_idx, test_idx = train_test_split(
        remaining_idx,
        train_size=rel,
        stratify=remaining_target,
        random_state=random_state,
        shuffle=shuffle,
    )

    return {
        SplitName.TRAIN.value: _ensure_index(train_idx, df.index),
        SplitName.VAL.value: _ensure_index(val_idx, df.index),
        SplitName.TEST.value: _ensure_index(test_idx, df.index),
    }


def random_split(
        df: pd.DataFrame,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: Optional[int] = None,
        shuffle: bool = True,
) -> Dict[str, pd.Index]:
    """Randomly split a DataFrame index into a dict of (train/val/test) indices."""
    return _split_indices(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        target=None,
        random_state=random_state,
        shuffle=shuffle,
    )


def stratified_split(
        df: pd.DataFrame,
        target: Union[str, pd.Series],
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: Optional[int] = None,
        shuffle: bool = True,
) -> Dict[str, pd.Index]:
    """Stratified split of a DataFrame index into a dict of (train/val/test) indices."""
    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Target '{target}' must be a column in `df`.")
        target = df[target]

    return _split_indices(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        target=target,
        random_state=random_state,
        shuffle=shuffle,
    )
