from __future__ import annotations

from collections.abc import MutableMapping, Iterator
from types import MappingProxyType
from typing import Any, Mapping, Optional


class ScopedView(MutableMapping[str, Any]):
    """
    A mutable mapping that exposes only the keys in `data` under a given prefix.

    Example:
        data = {"user.name": "Ada", "user.age": 37, "sys.version": "3.12"}
        view = ScopedView(data, "user")
        list(view)          # ["name", "age"]
        view["name"]        # "Ada"
        view["age"] = 38    # data["user.age"] becomes 38
        "name" in view      # True
        "version" in view   # False
    """

    __slots__ = ("_data", "_prefix", "_prefix_dot")

    def __init__(self, data: dict[str, Any], prefix: str):
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        if prefix.endswith("."):
            raise ValueError("prefix must not end with a '.'")
        self._data = data
        self._prefix = prefix
        self._prefix_dot = prefix + "."

    def _qualify(self, k: str) -> str:
        """Qualify a local key with the scope prefix."""
        return f"{self._prefix_dot}{k}"

    def _is_scoped(self, k: str) -> bool:
        """Return True if a fully-qualified key belongs to this scope."""
        return k.startswith(self._prefix_dot)

    def __getitem__(self, k: str) -> Any:
        return self._data[self._qualify(k)]

    def __setitem__(self, k: str, v: Any) -> None:
        self._data[self._qualify(k)] = v

    def __delitem__(self, k: str) -> None:
        del self._data[self._qualify(k)]

    def __iter__(self) -> Iterator[str]:
        # iterate local (unqualified) keys only
        p = self._prefix_dot
        return (k[len(p):] for k in self._data if k.startswith(p))

    def __len__(self) -> int:
        p = self._prefix_dot
        return sum(1 for k in self._data if k.startswith(p))

    def __contains__(self, k: object) -> bool:
        return isinstance(k, str) and (self._qualify(k) in self._data)

    def clear(self) -> None:
        # delete only keys within this scope
        for k in list(self):  # list() to avoid runtime mutation issues
            del self[k]

    def update(self, other: Mapping[str, Any] | None = None, /, **kwargs: Any) -> None:
        if other:
            for k, v in other.items():
                self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __repr__(self) -> str:
        keys_preview = ", ".join(list(self)[:5])
        more = "..." if len(self) > 5 else ""
        return f"ScopedView(prefix={self._prefix!r}, keys=[{keys_preview}{more}])"


class State(MutableMapping[str, Any]):
    """
    A simple mutable key-value store with read-only views and prefix scoping.
    """

    __slots__ = ("_data",)

    def __init__(self, initial: Optional[dict[str, Any]] = None):
        self._data: dict[str, Any] = dict(initial or {})

    def __getitem__(self, k: str) -> Any:
        return self._data[k]

    def __setitem__(self, k: str, v: Any) -> None:
        self._data[k] = v

    def __delitem__(self, k: str) -> None:
        del self._data[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get_view(self) -> Mapping[str, Any]:
        """
        Return a shallow, read-only mapping to the underlying data.
        Useful for safe logging / eventing without exposing mutability.
        """
        return MappingProxyType(self._data)

    def scope(self, prefix: str) -> MutableMapping[str, Any]:
        """
        Return a mutable view restricted to keys under `prefix + '.'`.
        """
        return ScopedView(self._data, prefix)

    def copy(self) -> "State":
        """Shallow copy of the state."""
        return State(dict(self._data))

    def __repr__(self) -> str:
        # keep repr compact
        items = list(self._data.items())[:6]
        body = ", ".join(f"{k!r}: {v!r}" for k, v in items)
        suffix = ", ..." if len(self._data) > 6 else ""
        return f"State({{{body}{suffix}}})"
