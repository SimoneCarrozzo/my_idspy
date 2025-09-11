from collections.abc import MutableMapping, Iterator
from types import MappingProxyType
from typing import Any, Mapping

from ..common.predicate import Predicate

_SEPARATOR = "."
StatePredicate = Predicate["State"]


class ScopedView(MutableMapping[str, Any]):
    """Mutable view over keys under a given prefix."""

    __slots__ = ("_data", "_prefix", "_prefix_dot")

    def __init__(self, data: dict[str, Any], prefix: str):
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        if prefix.endswith(_SEPARATOR):
            raise ValueError(f"prefix must not end with '{_SEPARATOR}'")
        self._data = data
        self._prefix = prefix
        self._prefix_dot = prefix + _SEPARATOR

    def _qualify(self, key: str) -> str:
        if not isinstance(key, str) or not key:
            raise KeyError("key must be a non-empty string")
        return f"{self._prefix_dot}{key}"

    def __getitem__(self, key: str) -> Any:
        return self._data[self._qualify(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[self._qualify(key)] = value

    def __delitem__(self, key: str) -> None:
        del self._data[self._qualify(key)]

    def __iter__(self) -> Iterator[str]:
        p = self._prefix_dot
        return (k.removeprefix(p) for k in self._data if k.startswith(p))

    def __len__(self) -> int:
        p = self._prefix_dot
        return sum(k.startswith(p) for k in self._data)

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return isinstance(key, str) and (self._qualify(key) in self._data)

    def clear(self) -> None:
        for k in list(self):
            del self[k]

    def to_dict(self) -> dict[str, Any]:
        """Shallow copy of the scoped mapping."""
        return {k: self[k] for k in self}

    def __repr__(self) -> str:
        keys = list(self)
        preview = ", ".join(keys[:5])
        more = "..." if len(keys) > 5 else ""
        return f"ScopedView(prefix={self._prefix!r}, size={len(keys)}, keys=[{preview}{more}])"


class State(MutableMapping[str, Any]):
    """Mutable key-value store with readonly and scoped views."""

    __slots__ = ("_data",)

    def __init__(self, initial: dict[str, Any] | None = None):
        self._data: dict[str, Any] = dict(initial or {})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._data

    def clear(self) -> None:
        self._data.clear()

    def readonly(self) -> Mapping[str, Any]:
        """Shallow readonly view of the data."""
        return MappingProxyType(self._data)

    def scope(self, prefix: str) -> ScopedView:
        """Mutable view restricted to keys under `prefix + '.'`."""
        return ScopedView(self._data, prefix)

    def copy(self) -> "State":
        """Shallow copy of the state."""
        return State(dict(self._data))

    def to_dict(self) -> dict[str, Any]:
        """Shallow copy as a plain dict."""
        return dict(self._data)

    def __repr__(self) -> str:
        n = len(self._data)
        items = list(self._data.items())[:6]
        body = ", ".join(f"{k!r}: {v!r}" for k, v in items)
        suffix = ", ..." if n > 6 else ""
        return f"State(size={n}, data={{{body}{suffix}}})"


def has_key(key: str) -> StatePredicate:
    """Accept states that contain a certain key."""
    return lambda state: key in state


def key_equals(key: str, value: object) -> StatePredicate:
    """Accept states where a key equals a specific value."""
    return lambda state: state.get(key) == value


def key_not_equals(key: str, value: object) -> StatePredicate:
    """Accept states where a key does not equal a specific value."""
    return lambda state: state.get(key) != value


def key_exists_and(key: str, predicate: Predicate) -> StatePredicate:
    """Accept states where a key exists and its value satisfies the predicate."""
    return lambda state: key in state and predicate(state[key])


def key_is_truthy(key: str) -> StatePredicate:
    """Accept states where a key exists and its value is truthy."""
    return lambda state: bool(state.get(key))


def key_is_falsy(key: str) -> StatePredicate:
    """Accept states where a key does not exist or its value is falsy."""
    return lambda state: not state.get(key)
