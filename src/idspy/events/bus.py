import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Final, Union

from .events import Event

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    def handle(self, event: Event) -> None:
        """Process the event."""
        pass

    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the event. Override for custom logic."""
        return True

    def __call__(self, event: Event) -> None:
        """Make handler callable like the original function-based approach."""
        if self.can_handle(event):
            self.handle(event)


FunctionHandler = Callable[[Event], None]
Handler = Union[BaseHandler, FunctionHandler]
Predicate = Callable[[Event], bool]

#tale metodo rappresenta una singola iscrizione a un evento sul bus.
#è dotato di un callback (funzione da chiamare quando l'evento viene pubblicato),
#un predicato opzionale (funzione che filtra gli eventi) e 
#un token unico (intero) per identificare l'iscrizione e permettere la cancellazione.
@dataclass(frozen=True, slots=True)
class _Entry:
    """Subscription entry."""

    callback: Handler
    predicate: Optional[Predicate]
    token: int  # unique id for unsubscription
    priority: int

#tale classe implementa un bus di eventi sincrono minimale.
#Permette di iscriversi a eventi specifici o a tutti gli eventi,
#di pubblicare eventi e di gestire le iscrizioni tramite token unici.
#Quando un evento viene pubblicato, viene inviato a tutti i sottoscrittori registrati che soddisfano il predicato (se presente).
class EventBus:
    """Minimal synchronous event bus."""

    ALL: Final[Optional[str]] = None  # subscribe to all events

    def __init__(self) -> None:
        self._subs: Dict[Optional[str], List[_Entry]] = {}
        self._next_token: int = 1

    def subscribe(
        self,
        callback: Handler,
        event_type: Optional[str] = None,
        predicate: Optional[Predicate] = None,
        priority: int = 1,
    ) -> int:
        """Register callback; returns a token."""
        token = self._next_token
        self._next_token += 1
        self._subs.setdefault(event_type, []).append(
            _Entry(callback, predicate, token, priority)
        )
        return token

    def on(
        self,
        event_type: Optional[str] = None,
        predicate: Optional[Predicate] = None,
        priority: int = 1,
    ) -> Callable[[Handler], Handler]:
        """Decorator to subscribe a handler."""

        def decorator(fn: Handler) -> Handler:
            self.subscribe(
                fn, event_type=event_type, predicate=predicate, priority=priority
            )
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

        targets.sort(key=lambda entry: entry.priority)

        for entry in list(targets):
            try:
                if entry.predicate is None or entry.predicate(event):
                    entry.callback(event)
            except Exception:
                logger.exception(
                    "Subscriber error for %s (id=%s)", event.type, event.id
                )
