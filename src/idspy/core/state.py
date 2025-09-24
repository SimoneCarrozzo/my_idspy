from collections.abc import Iterator, Mapping
from typing import Any, Callable, Type, TypeVar
from types import MappingProxyType

T = TypeVar("T")


class State:
    """Typed key-value store with dynamic prefix views."""

    SEPARATOR = "."

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        if data:
            self._data.update(data)

    @staticmethod
    def _prefix(prefix: str) -> str:
        return prefix + State.SEPARATOR if prefix else ""

    # ---------- typed access ----------

    def set(self, key: str, value: T, typ: Type[T]) -> None:
        """Set value enforcing type."""
        if not isinstance(value, typ):
            raise TypeError(
                f"Value for '{key}' must be {typ.__name__}, not {type(value).__name__}"
            )
        # Check existing value type compatibility
        if key in self._data:
            existing_type = type(self._data[key])
            if existing_type is not typ:
                raise TypeError(
                    f"Key '{key}' already exists with type {existing_type.__name__}, not {typ.__name__}"
                )
        self._data[key] = value

    def update(self, key: str, value: T, typ: Type[T]) -> None:
        """Force-set value (replaces existing type)."""
        if not isinstance(value, typ):
            raise TypeError(
                f"Value for '{key}' must be {typ.__name__}, not {type(value).__name__}"
            )
        self._data[key] = value

    def get(self, key: str, typ: Type[T]) -> T:
        """Get value asserting the expected type."""
        if key not in self._data:
            raise KeyError(f"Missing key '{key}'")

        value = self._data[key]
        if not isinstance(value, typ):
            actual_type = type(value).__name__
            expected_type = typ.__name__
            raise TypeError(f"Key '{key}' contains {actual_type}, not {expected_type}")
        return value

    def delete(self, key: str) -> None:
        """Delete key."""
        if key not in self._data:
            raise KeyError(f"Missing key '{key}'")
        del self._data[key]

    def has(self, key: str) -> bool:
        """Return True if key exists."""
        return key in self._data

    # ---------- iteration & introspection ----------

    def keys(self) -> Iterator[str]:
        """Iterate all keys."""
        return iter(self._data.keys())

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate all items."""
        return iter(self._data.items())

    def types(self) -> Mapping[str, Type]:
        """Return inferred types mapping."""
        return {k: type(v) for k, v in self._data.items()}

    def read_only_view(self) -> Mapping[str, Any]:
        """Return a read-only view of the internal data."""
        return MappingProxyType(self._data)

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the internal data as a standard dictionary."""
        return dict(self._data)

    # ---------- views ----------

    def view(self, prefix: str, strict: bool = True) -> "StateView":
        """Return a dynamic view for 'prefix.' with stripped keys.

        Args:
            prefix: The prefix to filter keys by
            strict: If True (default), only include keys with the prefix.
                   If False, include keys with prefix + keys without any prefix.
        """
        return StateView(self, prefix, strict)

    def __repr__(self) -> str:
        keys_types = [(k, type(v).__name__) for k, v in self._data.items()]
        return f"State(keys_and_types={keys_types})"


class StateView:
    """Dynamic view over a State for a given prefix.

    Rules:
    - Keys are accessed as 'prefix.key' in the underlying State.
    - In strict mode (default), only keys with the prefix are accessible.
    - In non-strict mode, keys without the prefix are also accessible but prioritized lower.
    """

    def __init__(self, state: State, prefix: str, strict: bool = True) -> None:
        self._state = state
        self._p = State._prefix(prefix)
        self._strict = strict

    # --- typed access ---

    def set(self, key: str, value: T, typ: Type[T]) -> None:
        """Set value in the view."""
        self._state.set(self._p + key, value, typ)

    def update(self, key: str, value: T, typ: Type[T]) -> None:
        """Force-set value and type in the view."""
        self._state.update(self._p + key, value, typ)

    def get(self, key: str, typ: Type[T]) -> T:
        """Get value from the view."""
        prefixed_key = self._p + key
        try:
            return self._state.get(prefixed_key, typ)
        except KeyError:
            if not self._strict and State.SEPARATOR not in key:
                # Try accessing without prefix when not in strict mode
                return self._state.get(key, typ)
            raise

    def delete(self, key: str) -> None:
        """Delete key from the view."""
        prefixed_key = self._p + key
        try:
            self._state.delete(prefixed_key)
        except KeyError:
            if not self._strict and State.SEPARATOR not in key:
                # Try deleting without prefix when not in strict mode
                self._state.delete(key)
            else:
                raise

    def has(self, key: str) -> bool:
        """Return True if key exists in the view."""
        prefixed_key = self._p + key
        if self._state.has(prefixed_key):
            return True
        if not self._strict and State.SEPARATOR not in key:
            # Try checking without prefix when not in strict mode
            return self._state.has(key)
        return False

    # --- iteration & introspection ---

    def keys(self) -> Iterator[str]:
        """Iterate bare keys in this view."""
        plen = len(self._p)
        for k in self._state.keys():
            if k.startswith(self._p):
                yield k[plen:]
            elif not self._strict and State.SEPARATOR not in k:
                # Include keys without any prefix when not in strict mode
                yield k

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate bare (key, value) in this view."""
        plen = len(self._p)
        for k, v in self._state.items():
            if k.startswith(self._p):
                yield (k[plen:], v)
            elif not self._strict and State.SEPARATOR not in k:
                # Include keys without any prefix when not in strict mode
                yield (k, v)

    def types(self) -> Mapping[str, Type]:
        """Return inferred types for this view (bare keys)."""
        plen = len(self._p)
        state_types = self._state.types()
        result = {}
        for k, t in state_types.items():
            if k.startswith(self._p):
                result[k[plen:]] = t
            elif not self._strict and State.SEPARATOR not in k:
                # Include keys without any prefix when not in strict mode
                result[k] = t
        return result

    def read_only_view(self) -> Mapping[str, Any]:
        """Return a read-only view of the data in this view (bare keys)."""
        return MappingProxyType({k: v for k, v in self.items()})

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the data in this view as a standard dictionary."""
        return dict({k: v for k, v in self.items()})

    def __repr__(self) -> str:
        keys_types = [(k, type(v).__name__) for k, v in self._data.items()]
        return f"StateView(prefix='{self._p}', strict={self._strict}, keys_and_types={keys_types})"


StatePredicate = Callable[[State | StateView], bool]


def has_key(key: str) -> StatePredicate:
    """Accept states that contain `key`."""
    return lambda state: state.has(key)


def key_equals(key: str, value: object) -> StatePredicate:
    """Accept states where `key` equals `value`."""

    def predicate(state):
        try:
            return state.get(key, type(value)) == value
        except (KeyError, TypeError):
            return False

    return predicate


def key_not_equals(key: str, value: object) -> StatePredicate:
    """Accept states where `key` does not equal `value`."""
    return lambda state: not key_equals(key, value)(state)


def key_is_truthy(key: str) -> StatePredicate:
    """Accept states where `key` exists and is truthy."""

    def predicate(state):
        try:
            return bool(state.get(key, bool))
        except (KeyError, TypeError):
            return False

    return predicate


def key_is_falsy(key: str) -> StatePredicate:
    """Accept states where `key` is missing or falsy."""

    def predicate(state):
        try:
            return not bool(state.get(key, bool))
        except (KeyError, TypeError):
            return True  # Missing key is considered falsy

    return predicate
