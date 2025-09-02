from collections.abc import MutableMapping, Iterator
from types import MappingProxyType
from typing import Any, Mapping


class ScopedView(MutableMapping[str, Any]):
    """A mapping that exposes only keys in `data` under a given `prefix`."""

    def __init__(self, data: dict[str, Any], prefix: str):
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        if prefix.endswith("."):
            raise ValueError("prefix must not end with '.'")
        self._data = data
        self._prefix = prefix
        self._prefix_dot = prefix + "."

    def _qualify(self, key: str) -> str:
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

    def __repr__(self) -> str:
        keys = list(self)
        preview = ", ".join(keys[:5])
        more = "..." if len(keys) > 5 else ""
        return f"ScopedView(prefix={self._prefix!r}, keys=[{preview}{more}])"


class State(MutableMapping[str, Any]):
    """A simple mutable key-value store with read-only views and prefix scoping."""

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

    def get_view(self) -> Mapping[str, Any]:
        """Return a shallow, read-only view of the underlying data."""
        return MappingProxyType(self._data)

    def scope(self, prefix: str) -> MutableMapping[str, Any]:
        """Return a mutable view restricted to keys under `prefix + '.'`."""
        return ScopedView(self._data, prefix)

    def copy(self) -> "State":
        """Shallow copy of the state."""
        return State(dict(self._data))

    def __repr__(self) -> str:
        items = list(self._data.items())[:6]
        body = ", ".join(f"{k!r}: {v!r}" for k, v in items)
        suffix = ", ..." if len(self._data) > 6 else ""
        return f"State({{{body}{suffix}}})"
