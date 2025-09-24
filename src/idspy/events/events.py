from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Dict, Final

from ..common.predicate import Predicate

EMPTY_MAP: Final[Mapping[str, Any]] = MappingProxyType({})
EventPredicate = Predicate["Event"]

#tale metodo converte un mapping (dizionario) in una versione di sola lettura.
#Se il mapping è None, restituisce una costante EMPTY_MAP che rappresenta un mapping vuoto.
#Se il mapping è già una MappingProxyType (già di sola lettura), lo restituisce così com'è.
#Altrimenti, crea una nuova MappingProxyType a partire da una copia del mapping originale.
def _ro(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Readonly mapping proxy (EMPTY_MAP if None)."""
    if mapping is None:
        return EMPTY_MAP
    if isinstance(mapping, MappingProxyType):
        return mapping
    return MappingProxyType(dict(mapping))

#tale classe rappresenta un evento immutabile con attributi come tipo, id, payload, stato e timestamp.
#Fornisce metodi per convertire l'evento in un dizionario e una rappresentazione stringa personalizzata.
#Nello specifico, dispone di diversi metodi, che si occupano rispettivamente di:
#- only_id: accettare solo eventi con un ID specifico (pipeline/step).
#- id_startswith: accettare eventi il cui ID inizia con un prefisso dato.
#- has_payload_key: accettare eventi che contengono una certa chiave nel payload.
#- payload_equals: accettare eventi in cui una chiave del payload è uguale a un valore specifico. 
@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event."""

    type: str
    id: str
    payload: Mapping[str, Any] = field(default_factory=lambda: EMPTY_MAP)
    state: Mapping[str, Any] = field(default_factory=lambda: EMPTY_MAP)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _ro(self.payload))
        object.__setattr__(self, "state", _ro(self.state))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "payload": dict(self.payload),
            "state": dict(self.state),
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        keys = list(self.payload)
        head = keys[:5]
        more = f", +{len(keys) - 5} keys" if len(keys) > 5 else ""
        return f"Event({self.type!r}, id={self.id!r}, payload_keys={head!r}{more})"


def only_id(event_id: str) -> EventPredicate:
    """Accept only events with a specific ID (pipeline/step)."""
    return lambda e: e.id == event_id


def id_startswith(prefix: str) -> EventPredicate:
    """Accept events whose ID starts with the given prefix."""
    return lambda e: e.id.startswith(prefix)


def has_payload_key(key: str) -> EventPredicate:
    """Accept events that carry a certain payload key."""
    return lambda e: key in e.payload


def payload_equals(key: str, value: object) -> EventPredicate:
    """Accept events where a payload key equals a specific value."""
    return lambda e: e.payload.get(key) == value
