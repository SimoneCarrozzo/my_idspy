import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Final

from .events import Event

logger = logging.getLogger(__name__)

Subscriber = Callable[[Event], None]
Predicate = Callable[[Event], bool]


@dataclass(frozen=True, slots=True)
class _Entry:
    """Subscription entry."""
    callback: Subscriber
    predicate: Optional[Predicate]
    token: int  # unique id for unsubscription


class EventBus:
    """Minimal synchronous event bus."""

    ALL: Final[Optional[str]] = None  # subscribe to all events

    def __init__(self) -> None:
        self._subs: Dict[Optional[str], List[_Entry]] = {}
        self._next_token: int = 1

    def subscribe(
            self,
            callback: Subscriber,
            event_type: Optional[str] = None,
            predicate: Optional[Predicate] = None,
    ) -> int:
        """Register callback; returns a token."""
        token = self._next_token
        self._next_token += 1
        self._subs.setdefault(event_type, []).append(_Entry(callback, predicate, token))
        return token

    def on(
            self,
            event_type: Optional[str] = None,
            predicate: Optional[Predicate] = None,
    ) -> Callable[[Subscriber], Subscriber]:
        """Decorator to subscribe a handler."""

        def decorator(fn: Subscriber) -> Subscriber:
            self.subscribe(fn, event_type=event_type, predicate=predicate)  # FIX: callback first
            return fn

        return decorator

    def unsubscribe(self, token: int) -> bool:
        """Unsubscribe by token; returns True if removed."""
        removed = False
        for key, entries in list(self._subs.items()):
            kept = [e for e in entries if e.token != token]
            if len(kept) != len(entries):
                removed = True
                if kept:
                    self._subs[key] = kept
                else:
                    self._subs.pop(key, None)
        return removed

    def publish(self, event: Event) -> None:
        """Dispatch event to subscribers; log handler errors."""
        targets: List[_Entry] = []
        targets.extend(self._subs.get(event.type, ()))
        targets.extend(self._subs.get(self.ALL, ()))

        for entry in list(targets):
            try:
                if entry.predicate is None or entry.predicate(event):
                    entry.callback(event)
            except Exception:
                logger.exception("Subscriber error for %s (id=%s)", event.type, event.id)
