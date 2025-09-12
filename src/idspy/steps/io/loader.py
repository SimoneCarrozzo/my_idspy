from pathlib import Path
from typing import Optional, Any

import pandas as pd

from ...core.state import State
from ...core.step import Step
from ...data.repository import DataFrameRepository
from ...data.schema import Schema


class LoadData(Step):
    """Load data into state."""

    def __init__(
        self,
        path: str | Path,
        target: str = "data.root",
        schema: Optional[Schema] = None,
        load_meta: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.path: Path = Path(path)
        self.target = target
        self.schema = schema
        self.load_meta = load_meta
        self.kwargs = kwargs

        super().__init__(
            name=name or "load_data",
            requires=None,
            provides=[self.target],
        )

    def run(self, state: State) -> None:
        df: pd.DataFrame = DataFrameRepository.load(
            base_path=self.path,
            schema=self.schema,
            load_meta=self.load_meta,
            **self.kwargs,
        )
        state[self.target] = df
