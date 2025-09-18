from collections.abc import MutableMapping, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Callable

_SEPARATOR = "."
StatePredicate = Callable[["State"], bool]


class State(MutableMapping[str, Any]):
    """Mutable key-value store with scoped and read-only views."""

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        prefix: str = "",
        alias: bool = False,
    ):
        """
        Create a state.

        Args:
            data: Initial data dict.
            prefix: Scope prefix (without trailing '.').
            alias: If True, reuse the given dict; otherwise copy it.
        """
        if prefix.endswith(_SEPARATOR):
            raise ValueError(f"prefix must not end with '{_SEPARATOR}'")

        self._data: dict[str, Any] = (
            data if (alias and data is not None) else dict(data or {})
        )
        self._prefix = prefix
        self._prefix_dot = f"{prefix}{_SEPARATOR}" if prefix else ""

    def _qualify(self, key: str) -> str:
        """Return qualified key (prefix + key)."""
        if not isinstance(key, str) or not key:
            raise KeyError("key must be a non-empty string")
        return f"{self._prefix_dot}{key}" if self._prefix else key

    def _in_scope(self, qualified_key: str) -> bool:
        """Check if a qualified key belongs to this scope."""
        return not self._prefix or qualified_key.startswith(self._prefix_dot)

    def _unqualify(self, qualified_key: str) -> str:
        """Strip this scope's prefix from a qualified key."""
        if self._prefix and qualified_key.startswith(self._prefix_dot):
            return qualified_key[len(self._prefix_dot) :]
        return qualified_key

    def _scoped_keys(self) -> Iterator[str]:
        """Iterate unqualified keys in this scope."""
        for k in self._data:
            if self._in_scope(k):
                yield self._unqualify(k)

    def __getitem__(self, key: str) -> Any:
        return self._data[self._qualify(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[self._qualify(key)] = value

    def __delitem__(self, key: str) -> None:
        del self._data[self._qualify(key)]

    def __iter__(self) -> Iterator[str]:
        return self._scoped_keys()

    def __len__(self) -> int:
        return sum(1 for _ in self._scoped_keys())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (self._qualify(key) in self._data)

    def clear(self) -> None:
        """Remove all keys in this scope."""
        for k in list(self._scoped_keys()):
            del self[k]

    def readonly(self) -> Mapping[str, Any]:
        """Return a read-only view of the scoped data."""
        return MappingProxyType(self.to_dict())

    def scope(self, prefix: str) -> "State":
        """Return a mutable view limited to keys under `prefix + '.'`."""
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        if prefix.endswith(_SEPARATOR):
            raise ValueError(f"prefix must not end with '{_SEPARATOR}'")
        full_prefix = f"{self._prefix_dot}{prefix}" if self._prefix else prefix
        # Share the same underlying dict intentionally
        return State(self._data, prefix=full_prefix, alias=True)

    def copy(self) -> "State":
        """Return a shallow copy of the current scope."""
        return State(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow dict of the current scope."""
        return {k: self[k] for k in self._scoped_keys()}

    def get_typed(self, key: str, expected_type: Any) -> Any:
        """Get a value and assert its type."""
        qk = self._qualify(key)
        if qk not in self._data:
            raise KeyError(f"Key '{key}' not found in state.")
        value = self._data[qk]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Expected {expected_type} for key '{key}', got {type(value)}."
            )
        return value

    def check_type(self, key: str, expected_type: Any) -> bool:
        """Return True if key exists and value is of expected type."""
        qk = self._qualify(key)
        return qk in self._data and isinstance(self._data[qk], expected_type)

    def __repr__(self) -> str:
        """Return a debug-friendly representation."""
        if not self._prefix:
            n = len(self._data)
            items = list(self._data.items())[:6]
            body = ", ".join(f"{k!r}: {v!r}" for k, v in items)
            suffix = ", ..." if n > 6 else ""
            return f"State(size={n}, data={{{body}{suffix}}})"
        keys = list(self._scoped_keys())
        preview = ", ".join(keys[:5])
        more = "..." if len(keys) > 5 else ""
        return (
            f"State(prefix={self._prefix!r}, size={len(keys)}, keys=[{preview}{more}])"
        )


def has_key(key: str) -> StatePredicate:
    """Accept states that contain `key`."""
    return lambda state: key in state


def key_equals(key: str, value: object) -> StatePredicate:
    """Accept states where `key` equals `value`."""
    return lambda state: state.get(key) == value


def key_not_equals(key: str, value: object) -> StatePredicate:
    """Accept states where `key` does not equal `value`."""
    return lambda state: state.get(key) != value


def key_exists_and(key: str, predicate: Callable[[Any], bool]) -> StatePredicate:
    """Accept states where `key` exists and predicate(value) is True."""
    return lambda state: key in state and predicate(state[key])


def key_is_truthy(key: str) -> StatePredicate:
    """Accept states where `key` exists and is truthy."""
    return lambda state: bool(state.get(key))


def key_is_falsy(key: str) -> StatePredicate:
    """Accept states where `key` is missing or falsy."""
    return lambda state: not state.get(key)
