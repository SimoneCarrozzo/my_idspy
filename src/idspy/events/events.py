from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Dict


class EventType(Enum):
    PIPELINE_START = auto()
    PIPELINE_END = auto()
    STEP_START = auto()
    STEP_END = auto()
    STEP_ERROR = auto()


def _ro(mapping: Mapping[str, Any] | MutableMapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a read-only (shallow) view of the mapping; empty mapping if None."""
    if mapping is None:
        return MappingProxyType({})
    # Avoid double-wrapping
    if isinstance(mapping, MappingProxyType):
        return mapping
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class Event:
    """
    Immutable event emitted by pipelines/steps.

    Fields:
        type: kind of event (start/end/step/error)
        id: the event identifier
        payload: extra, non-sensitive details (e.g., requires/provides, index, error)
        state_view: read-only snapshot of current state (avoid full state if sensitive)
        timestamp: UTC time when the event object was created
    """

    type: EventType
    id: str
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    state_view: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _ro(self.payload))
        object.__setattr__(self, "state_view", _ro(self.state_view))

    @classmethod
    def pipeline_start(cls, pipeline_id: str, state_view: Mapping[str, Any], **payload: Any) -> "Event":
        return cls(EventType.PIPELINE_START, pipeline_id, payload, state_view)

    @classmethod
    def pipeline_end(cls, pipeline_id: str, state_view: Mapping[str, Any], **payload: Any) -> "Event":
        return cls(EventType.PIPELINE_END, pipeline_id, payload, state_view)

    @classmethod
    def step_start(cls, step_id: str, state_view: Mapping[str, Any], **payload: Any) -> "Event":
        return cls(EventType.STEP_START, step_id, payload, state_view)

    @classmethod
    def step_end(cls, step_id: str, state_view: Mapping[str, Any], **payload: Any) -> "Event":
        return cls(EventType.STEP_END, step_id, payload, state_view)

    @classmethod
    def step_error(
            cls,
            step_id: str,
            state_view: Mapping[str, Any],
            error: Exception | str,
            **payload: Any,
    ) -> "Event":
        details: Dict[str, Any] = {"error": repr(error) if isinstance(error, Exception) else str(error)}
        details.update(payload)
        return cls(EventType.STEP_ERROR, step_id, details, state_view)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (good for logging/JSON)."""
        return {
            "event_type": self.type.name,
            "id": self.id,
            "payload": dict(self.payload),
            "state_view": dict(self.state_view),
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:  # compact debug-friendly repr
        base = (
            f"Event({self.type.name}, id={self.id!r}, payload_keys={list(self.payload)[:5]!r}"
        )
        return base + (", ...)" if len(self.payload) > 5 else ")")
