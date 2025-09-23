import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Hashable, Literal, Tuple

import pandas as pd
import numpy as np

from ..events import Event
from ..bus import BaseHandler


def _ts_compact(dt: datetime) -> str:
    """Local time HH:MM:SS.mmm."""
    return dt.astimezone().strftime("%H:%M:%S.%f")[:-3]


def _fmt_delta_ms(ms: float) -> str:
    return f"{ms:.1f}ms" if ms < 1000 else f"{ms / 1000:.3f}s"


def _repr_truncated(obj: Any, max_chars: int) -> str:
    s = repr(obj)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


@dataclass(slots=True)
class Logger(BaseHandler):
    """Log basic event info (human or JSON)."""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    level: int = logging.INFO
    as_json: bool = False
    max_chars: int = 2000

    def handle(self, event: Event) -> None:
        if self.as_json:
            rec = {
                "kind": "event",
                "type": event.type,
                "id": event.id,
                "ts": event.timestamp.isoformat(),
                "payload_keys": list(event.payload.keys()),
            }
            self.logger.log(self.level, json.dumps(rec, default=str))
        else:
            payload_str = _repr_truncated(event.payload.keys(), self.max_chars)
            self.logger.log(
                self.level,
                "EVENT type=%s id=%s ts=%s | payload=%s",
                str(event.type),
                event.id,
                _ts_compact(event.timestamp),
                payload_str,
            )

    def can_handle(self, event: Event) -> bool:
        return True


GroupBy = Literal["id", "type", "all"]


@dataclass(slots=True)
class Tracer(BaseHandler):
    """Trace time between consecutive events."""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    level: int = logging.INFO
    group_by: GroupBy = "type"
    log_first: bool = True
    _last: Dict[Hashable, Tuple[str, datetime]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _key(self, event: Event) -> Hashable:
        if self.group_by == "id":
            return event.id
        if self.group_by == "type":
            return event.type
        return "__ALL__"

    def handle(self, event: Event) -> None:
        k = self._key(event)
        now = event.timestamp
        prev = self._last.get(k)

        if prev is None:
            if self.log_first:
                self.logger.log(
                    self.level,
                    "TRACE group=%s start id=%s",
                    self.group_by,
                    event.id,
                )
        else:
            prev_id, prev_ts = prev
            delta_ms = (now - prev_ts).total_seconds() * 1000.0
            self.logger.log(
                self.level,
                "TRACE group=%s dt=%s | from=%s -> to=%s",
                self.group_by,
                _fmt_delta_ms(delta_ms),
                prev_id,
                event.id,
            )

        self._last[k] = (event.id, now)

    def can_handle(self, event: Event) -> bool:
        return True


def _fmt_bytes(n: int | float) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.1f}{units[i]}"


@dataclass(slots=True)
class DataFrameProfiler(BaseHandler):
    """Profile a DataFrame carried by events."""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    level: int = logging.INFO
    key: str = "data.root"
    deep_memory: bool = True
    include_index: bool = False
    max_chars: int = 2000

    def _pick(self, event: Event) -> Any:
        return event.payload.get(self.key)

    def handle(self, event: Event) -> None:
        df = self._pick(event)
        try:
            rows, cols = getattr(df, "shape", (None, None))
            mem = df.memory_usage(index=self.include_index, deep=self.deep_memory).sum()
            nans = df.isna().sum().sum()

            self.logger.log(
                self.level,
                "DATAFRAME id=%s shape=(%s,%s) mem=%s nans=%s dtypes=%s",
                event.id,
                rows,
                cols,
                _fmt_bytes(mem),
                nans,
                _repr_truncated(df.dtypes.to_dict(), self.max_chars),
            )
        except Exception:
            self.logger.exception("DATAFRAME error type=%s id=%s", event.type, event.id)

    def can_handle(self, event: Event) -> bool:
        df = self._pick(event)
        return df is not None and isinstance(df, pd.DataFrame)
