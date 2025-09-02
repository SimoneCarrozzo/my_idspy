import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Any, Dict

import pandas as pd

from ..events import Event


@dataclass
class DataFrameProfiler:
    """
    Log simple, generic stats for a DataFrame stored in `event.context[key]`.

    What it logs:
      - basic summary: rows, columns, n_unique per column, NaNs per column
      - value_counts for selected columns (or all columns if `profile_columns=None`),
        limited to `top_n`, including NaNs
    """

    key: str
    profile_columns: Optional[Sequence[str]] = None  # None -> profile all columns
    top_n: int = 10
    level: int = logging.INFO
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def __call__(self, event: Event) -> None:
        ctx: Dict[str, Any] = getattr(event, "context", {})
        if self.key not in ctx:
            self.logger.log(self.level, "[DFSTATS] event=%s | key=%s not found in context", event.id, self.key)
            return

        obj = ctx[self.key]

        # Accept objects that wrap a DataFrame on `.df`
        if not isinstance(obj, pd.DataFrame) and hasattr(obj, "df"):
            df_candidate = getattr(obj, "df")
            if isinstance(df_candidate, pd.DataFrame):
                obj = df_candidate

        if not isinstance(obj, pd.DataFrame):
            self.logger.log(
                self.level,
                "[DFSTATS] event=%s | key=%s exists but is not a DataFrame (type=%s)",
                event.id,
                self.key,
                type(obj).__name__,
            )
            return

        df: pd.DataFrame = obj

        rows = len(df)
        columns = list(df.columns)
        n_unique = df.nunique(dropna=True).to_dict()
        nans = df.isna().sum().to_dict()

        self.logger.log(
            self.level,
            "[DFSTATS] event=%s | key=%s | rows=%d | columns=%s | n_unique=%s | nans=%s",
            event.id,
            self.key,
            rows,
            columns,
            n_unique,
            nans,
        )

        cols_to_profile = list(columns) if self.profile_columns is None else list(self.profile_columns)

        for col in cols_to_profile:
            if col not in df.columns:
                self.logger.log(
                    self.level,
                    "[DFSTATS] event=%s | key=%s | column '%s' not in DataFrame",
                    event.id,
                    self.key,
                    col,
                )
                continue

            vc = df[col].value_counts(dropna=False).head(self.top_n)
            vc_dict = dict(vc.items())

            self.logger.log(
                self.level,
                "[DFSTATS] event=%s | key=%s | %s: value_counts(top=%d)=%s",
                event.id,
                self.key,
                col,
                self.top_n,
                vc_dict,
            )
