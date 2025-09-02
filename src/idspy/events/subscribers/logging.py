import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Hashable, Literal, Tuple

from ..events import Event


def _ts_compact(dt: datetime) -> str:
    local_dt = dt.astimezone()
    return local_dt.strftime("%H:%M:%S.%f")[:-3]


def _fmt_delta_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.3f}s"


@dataclass
class Logger:
    """
    Subscriber that logs event information.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    level: int = logging.INFO

    def __call__(self, event: Event) -> None:
        self.logger.log(
            self.level,
            "[Event: type=%s, id=%s, timestamp=%s] -> payload=%s",
            event.type,
            event.id,
            _ts_compact(event.timestamp),
            dict(event.payload),
        )


GroupBy = Literal["id", "type", "all"]


@dataclass
class Tracer:
    """
    Subscriber that traces time between consecutive events.

    - group_by="id":   measure deltas per correlation id (default)
    - group_by="type": measure deltas per event type
    - group_by="all":  single global sequence (any event after any event)
    """

    logger: logging.Logger = logging.getLogger(__name__)
    level: int = logging.INFO
    group_by: GroupBy = "type"
    _last: Dict[Hashable, Tuple[str, datetime]] = field(default_factory=dict, init=False, repr=False)

    def _key(self, event: Event) -> Hashable:
        if self.group_by == "id":
            return event.id
        if self.group_by == "type":
            return event.type
        return "__ALL__"

    def __call__(self, event: Event) -> None:
        k = self._key(event)
        now = event.timestamp

        if k in self._last:
            prev_id, prev_ts = self._last[k]
            delta_ms = (now - prev_ts).total_seconds() * 1000.0
            self.logger.log(
                self.level,
                "[TRACE] Δ=%s | %s → %s",
                _fmt_delta_ms(delta_ms),
                prev_id,
                event.id,
            )
        else:
            self.logger.log(
                self.level,
                "[TRACE] start | type=%s",
                event.type,
            )

        self._last[k] = (event.id, now)

    def reset(self, key: Hashable | None = None) -> None:
        """Clear tracing state (for a specific key or all)."""
        if key is None:
            self._last.clear()
        else:
            self._last.pop(key, None)
