from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_EPS = 1e-9


def _validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    for name, v in (("train_size", train_size), ("val_size", val_size), ("test_size", test_size)):
        if not (0.0 - _EPS <= v <= 1.0 + _EPS):
            raise ValueError(f"{name} must be in [0, 1]; got {v}")

    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0, atol=_EPS):
        raise ValueError("train_size + val_size + test_size must sum to 1.0; got {total}")


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
        shuffle: Optional[bool] = True,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Core splitter: if `y` is provided, uses stratification; otherwise random split.
    """
    _validate_split_sizes(train_size, val_size, test_size)
    if target is not None and len(target) != len(df):
        raise ValueError(f"Length of `y` ({len(target)}) must match number of rows in `df` ({len(df)})")

    split = train_test_split(
        df.index,
        train_size=train_size,
        stratify=target,
        random_state=random_state,
        shuffle=shuffle,
    )
    train_idx, remaining_idx = split

    train_idx = _ensure_index(train_idx, df.index)
    remaining_idx = _ensure_index(remaining_idx, df.index)

    # No val/test
    rel = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.0
    if rel <= _EPS:
        return train_idx, _empty_like_index(df.index), remaining_idx
    if (1.0 - rel) <= _EPS:
        return train_idx, remaining_idx, _empty_like_index(df.index)

    remaining_target = target[remaining_idx] if target is not None else None

    split2 = train_test_split(
        remaining_idx,
        train_size=rel,
        stratify=remaining_target,
        random_state=random_state,
        shuffle=shuffle,
    )

    val_idx, test_idx = split2

    return (
        _ensure_index(train_idx, df.index),
        _ensure_index(val_idx, df.index),
        _ensure_index(test_idx, df.index),
    )


def random_split(
        df: pd.DataFrame,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: Optional[int] = None,
        shuffle: Optional[bool] = True,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Randomly split a DataFrame index into (train, val, test)."""
    return _split_indices(df, train_size=train_size, val_size=val_size, test_size=test_size, target=None,
                          random_state=random_state, shuffle=shuffle)


def stratified_split(
        df: pd.DataFrame,
        target: pd.Series,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: Optional[int] = None,
        shuffle: Optional[bool] = True,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Stratified split of a DataFrame index into (train, val, test)."""
    return _split_indices(df, train_size=train_size, val_size=val_size, test_size=test_size, target=target,
                          random_state=random_state, shuffle=shuffle)
