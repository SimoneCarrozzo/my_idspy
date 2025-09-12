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
        path_in: str | Path,
        dataframe_out: str = "data.root",
        schema: Optional[Schema] = None,
        load_meta: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.path_in: Path = Path(path_in)
        self.dataframe_out = dataframe_out
        self.schema = schema
        self.load_meta = load_meta
        self.kwargs = kwargs

        super().__init__(
            name=name or "load_data",
            requires=None,
            provides=[self.dataframe_out],
        )

    def run(self, state: State) -> None:
        dataframe: pd.DataFrame = DataFrameRepository.load(
            base_path=self.path_in,
            schema=self.schema,
            load_meta=self.load_meta,
            **self.kwargs,
        )
        state[self.dataframe_out] = dataframe
