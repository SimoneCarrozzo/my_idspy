import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Hashable, Literal, Tuple

from ..events import Event


def _ts_compact(dt: datetime) -> str:
    """Local time HH:MM:SS.mmm."""
    return dt.astimezone().strftime("%H:%M:%S.%f")[:-3]


def _fmt_delta_ms(ms: float) -> str:
    return f"{ms:.1f}ms" if ms < 1000 else f"{ms / 1000:.3f}s"


def _repr_truncated(obj: Any, max_chars: int) -> str:
    s = repr(obj)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


@dataclass(slots=True)
class Logger:
    """Log basic event info (human or JSON)."""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    level: int = logging.INFO
    as_json: bool = False
    max_payload_chars: int = 1000  # only for human mode

    def __call__(self, event: Event) -> None:
        if self.as_json:
            rec = {
                "kind": "event",
                "type": event.type,
                "id": event.id,
                "ts": event.timestamp.isoformat(),
                "payload": dict(event.payload),
                "state_keys": list(event.state.keys()),
            }
            # default=str to avoid serialization errors on exotic payloads
            self.logger.log(self.level, json.dumps(rec, default=str))
        else:
            payload_str = _repr_truncated(dict(event.payload), self.max_payload_chars)
            self.logger.log(
                self.level,
                "EVENT type=%s id=%s ts=%s | payload=%s",
                str(event.type),
                event.id,
                _ts_compact(event.timestamp),
                payload_str,
            )


GroupBy = Literal["id", "type", "all"]


@dataclass(slots=True)
class Tracer:
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

    def __call__(self, event: Event) -> None:
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

    def reset(self, key: Hashable | None = None) -> None:
        """Clear tracing state (for a specific key or all)."""
        if key is None:
            self._last.clear()
        else:
            self._last.pop(key, None)


def _fmt_bytes(n: int | float) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.1f}{units[i]}"


@dataclass(slots=True)
class DataFrameProfiler:
    """Profile a DataFrame carried by events."""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    level: int = logging.INFO
    key: str = "data.root"
    deep_memory: bool = True
    include_index: bool = False
    max_dtype_items: int = 8  # top-N dtype counts to show

    def _pick(self, event: Event) -> Any:
        return event.state.get(self.key)

    def __call__(self, event: Event) -> None:
        df = self._pick(event)
        if df is None:
            return
        try:
            # duck-typing to avoid hard dependency on pandas in this module
            rows, cols = getattr(df, "shape", (None, None))
            mem = df.memory_usage(index=self.include_index, deep=self.deep_memory).sum()
            dtypes = getattr(df, "dtypes", None)
            dtype_summary = ""
            if dtypes is not None:
                vc = dtypes.astype(str).value_counts()  # type: ignore[no-untyped-call]
                # stable ordering: count desc, dtype name asc
                items = sorted(vc.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
                items = items[: self.max_dtype_items]
                dtype_summary = ", ".join(f"{dt}:{int(cnt)}" for dt, cnt in items)

            nans = df.isna().sum().sum()

            self.logger.log(
                self.level,
                "DATAFRAME id=%s shape=(%s,%s) mem=%s nans=%s dtypes=%s",
                event.id,
                rows,
                cols,
                _fmt_bytes(mem),
                nans,
                dtype_summary,
            )
        except Exception:
            self.logger.exception("DATAFRAME error type=%s id=%s", event.type, event.id)
