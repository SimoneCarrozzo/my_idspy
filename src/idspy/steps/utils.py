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
