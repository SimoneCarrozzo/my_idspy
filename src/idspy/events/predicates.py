from src.idspy.events.bus import Predicate


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


def only_id(event_id: str) -> Predicate:
    """Accept only events with a specific ID (pipeline/step)."""
    return lambda e: e.id == event_id


def id_startswith(prefix: str) -> Predicate:
    """Accept events whose ID starts with the given prefix."""
    return lambda e: e.id.startswith(prefix)


def has_payload_key(key: str) -> Predicate:
    """Accept events that carry a certain payload key."""
    return lambda e: key in e.payload


def payload_equals(key: str, value: object) -> Predicate:
    """Accept events where a payload key equals a specific value."""
    return lambda e: e.payload.get(key) == value
