from pathlib import Path
from typing import Optional, Any

import pandas as pd

from ..helpers import validate_instance
from ...core.state import State
from ...core.step import Step
from ...data.repository import DataFrameRepository


class SaveData(Step):
    """Save data from state."""

    def __init__(
        self,
        path_out: str | Path,
        dataframe_in: str = "data.root",
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        save_meta: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.path_out: Path = Path(path_out)
        self.dataframe_in = dataframe_in
        self.fmt = fmt
        self.file_name = file_name or Path(self.dataframe_in).suffix.lstrip(".")
        self.save_meta = save_meta
        self.kwargs = kwargs

        super().__init__(
            name=name or "save_data",
            requires=[self.dataframe_in],
            provides=None,
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        # Default save name from the tail of the state key (e.g., "data.root" -> "root"),
        DataFrameRepository.save(
            dataframe,
            self.path_out,
            name=self.file_name,
            fmt=self.fmt,
            save_meta=self.save_meta,
            **self.kwargs,
        )
