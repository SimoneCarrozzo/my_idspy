import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .events import Event, EventType

logger = logging.getLogger(__name__)

Subscriber = Callable[[Event], None]
Predicate = Callable[[Event], bool]


@dataclass(frozen=True)
class _Entry:
    callback: Subscriber
    predicate: Optional[Predicate]
    token: int  # unique id for unsubscription


class EventBus:
    """
    Minimal synchronous EventBus (single-threaded).

    Features:
      - Subscribe to a specific EventType or to ALL events.
      - Optional predicate per subscriber to filter events.
      - Unsubscribe via token (returned by subscribe).
    """

    _ALL: Optional[EventType] = None  # special key for catch-all subscribers

    def __init__(self) -> None:
        # event_type -> list of subscriber entries
        self._subs: Dict[Optional[EventType], List[_Entry]] = {}
        self._next_token = 1

    def subscribe(
            self,
            event_type: EventType,
            callback: Subscriber,
            predicate: Optional[Predicate] = None,
    ) -> int:
        """Register `callback` for events of `event_type` (optionally filtered)."""
        token = self._next_token
        self._next_token += 1
        self._subs.setdefault(event_type, []).append(_Entry(callback, predicate, token))
        return token

    def subscribe_all(
            self,
            callback: Subscriber,
            predicate: Optional[Predicate] = None,
    ) -> int:
        """Register `callback` for *all* event types."""
        token = self._next_token
        self._next_token += 1
        self._subs.setdefault(self._ALL, []).append(_Entry(callback, predicate, token))
        return token

    def unsubscribe(self, token: int) -> bool:
        """Remove a previously subscribed callback by token. Returns True if removed."""
        removed = False
        for key, entries in list(self._subs.items()):
            new_entries = [e for e in entries if e.token != token]
            if len(new_entries) != len(entries):
                removed = True
                if new_entries:
                    self._subs[key] = new_entries
                else:
                    self._subs.pop(key, None)
        return removed

    def publish(self, event: Event) -> None:
        """
        Synchronously dispatch `event` to matching subscribers.
        Exceptions inside callbacks are caught and logged.
        """
        # Support both new `event.event_type` and legacy `event.type`.
        event_type: EventType = getattr(event, "event_type", getattr(event, "type"))

        targets = list(self._subs.get(event_type, ())) + list(self._subs.get(self._ALL, ()))
        # Iterate over a copy so callbacks can unsubscribe safely inside themselves.
        for entry in list(targets):
            try:
                if entry.predicate is None or entry.predicate(event):
                    entry.callback(event)
            except RuntimeError:
                logger.exception(
                    "Subscriber error for %s (id=%r)",
                    event_type.name,
                    event.id,
                )
