import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Final

from .events import Event

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
    Minimal synchronous event bus.

    - Subscribe to a specific event type (str) or to ALL events (event_type=None).
    - Optional predicate for fine-grained filtering.
    - Unsubscribe via token.
    - Decorator-based registration: `@bus.on("pipeline.start")` or `@bus.on()`.
    """

    ALL: Final[Optional[str]] = None  # sentinel for "all events"

    def __init__(self) -> None:
        self._subs: Dict[Optional[str], List[_Entry]] = {}
        self._next_token: int = 1

    def subscribe(
            self,
            callback: Subscriber,
            event_type: Optional[str] = None,
            predicate: Optional[Predicate] = None,
    ) -> int:
        """Register `callback` for `event_type` (use None for ALL). Returns a token."""
        token = self._next_token
        self._next_token += 1
        self._subs.setdefault(event_type, []).append(_Entry(callback, predicate, token))
        return token

    def on(
            self,
            event_type: Optional[str] = None,
            predicate: Optional[Predicate] = None,
    ) -> Callable[[Subscriber], Subscriber]:
        """
        Decorator to subscribe a handler.
        """

        def decorator(fn: Subscriber) -> Subscriber:
            self.subscribe(event_type, fn, predicate=predicate)
            return fn

        return decorator

    def unsubscribe(self, token: int) -> bool:
        """Remove previously subscribed callback by token. Returns True if removed."""
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
        """
        Synchronously dispatch `event` to matching subscribers.
        Exceptions inside callbacks are caught and logged.
        """
        targets = []
        targets.extend(self._subs.get(event.type, ()))
        targets.extend(self._subs.get(self.ALL, ()))

        for entry in list(targets):
            try:
                if entry.predicate is None or entry.predicate(event):
                    entry.callback(event)
            except Exception:
                logger.exception("Subscriber error for %s (id=%s)", event.type, event.id)
