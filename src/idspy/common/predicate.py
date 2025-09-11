from typing import Callable, TypeVar

T = TypeVar("T")
Predicate = Callable[[T], bool]


def and_(*predicates: Predicate) -> Predicate:
    """Predicate that passes if *all* given predicates pass."""
    return lambda e: all(p(e) for p in predicates)


def or_(*predicates: Predicate) -> Predicate:
    """Predicate that passes if *any* given predicate passes."""
    return lambda e: any(p(e) for p in predicates)


def not_(predicate: Predicate) -> Predicate:
    """Predicate that passes if the given predicate fails."""
    return lambda e: not predicate(e)


def any_of(predicates: list[Predicate]) -> Predicate:
    """Alias for or_ with a list."""
    return or_(*predicates)


def all_of(predicates: list[Predicate]) -> Predicate:
    """Alias for and_ with a list."""
    return and_(*predicates)
