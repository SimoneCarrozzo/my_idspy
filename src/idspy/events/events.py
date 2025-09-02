from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Dict, Final

EMPTY_MAP: Final[Mapping[str, Any]] = MappingProxyType({})


def _ro(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if mapping is None:
        return EMPTY_MAP
    if isinstance(mapping, MappingProxyType):
        return mapping
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class Event:
    """
    A tiny, immutable event. The meaning of `type` is application-defined.
    Fields:
      - type: string label for the event (e.g. "pipeline.start", "user.login")
      - id:   subject or correlation id (free-form)
      - payload: extra details (non-sensitive)
      - context: read-only snapshot of relevant context (keep it small)
      - timestamp: UTC creation time
    """
    type: str
    id: str
    payload: Mapping[str, Any] = field(default_factory=lambda: EMPTY_MAP)
    context: Mapping[str, Any] = field(default_factory=lambda: EMPTY_MAP)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _ro(self.payload))
        object.__setattr__(self, "context", _ro(self.context))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "payload": dict(self.payload),
            "context": dict(self.context),
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        keys = list(self.payload)
        head = keys[:5]
        more = f", +{len(keys) - 5} keys" if len(keys) > 5 else ""
        return f"Event({self.type!r}, id={self.id!r}, payload_keys={head!r}{more})"
