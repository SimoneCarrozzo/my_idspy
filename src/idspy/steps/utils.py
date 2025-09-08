from typing import Any, Tuple, Type


def validate_instance(
        obj: Any,
        types: Type[Any] | Tuple[Type[Any], ...],
        step_name: str,
) -> None:
    """
    Ensure `obj` is an instance or subclass of `types`.

    Raises:
        TypeError: if `obj` does not match.
        ValueError: if `types` is empty.
    """
    types_tuple: Tuple[Type[Any], ...] = (
        (types,) if isinstance(types, type) else tuple(types)
    )

    if not types_tuple:
        raise ValueError(f"{step_name}: 'types' must not be empty.")

    if not all(isinstance(t, type) for t in types_tuple):
        bad = [repr(t) for t in types_tuple if not isinstance(t, type)]
        raise TypeError(f"{step_name}: all entries in 'types' must be types, got {bad}.")

    is_valid = isinstance(obj, types_tuple) or (
            isinstance(obj, type) and issubclass(obj, types_tuple)
    )

    if not is_valid:
        expected = "/".join(t.__name__ for t in types_tuple)
        actual = obj.__name__ if isinstance(obj, type) else type(obj).__name__
        raise TypeError(f"{step_name}: expected {expected}, found {actual}.")


import pandas as pd
from typing import Optional, List


def validate_schema(df: pd.DataFrame, source_name: str) -> None:
    """Ensure dataframe has a schema."""
    if "_schema" not in df.attrs:
        raise ValueError(f"{source_name} must have a '_schema' attribute.")


def validate_split(
        df: pd.DataFrame,
        source_name: str,
        split_names: Optional[List[str]] = None
) -> None:
    """Ensure dataframe has splits and optionally check specific split names."""
    if "_splits" not in df.attrs:
        raise ValueError(f"{source_name} must have a '_splits' attribute.")

    if split_names:
        for split_name in split_names:
            try:
                getattr(df.tab, split_name)
            except KeyError as e:
                raise ValueError(f"{source_name} must have a '{split_name}' split defined.") from e


def validate_schema_and_split(
        df: pd.DataFrame,
        source_name: str,
        split_names: Optional[List[str]] = None
) -> None:
    """Ensure dataframe has schema and required splits."""
    validate_schema(df, source_name)
    validate_split(df, source_name, split_names)
