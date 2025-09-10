from pathlib import Path
from typing import Optional, Any, Mapping

import pandas as pd

from .utils import validate_instance, validate_split
from ..core.state import State
from ..core.step import Step
from ..data.repository import DataFrameRepository
from ..data.schema import Schema
from ..data.split import SplitName


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


class SaveData(Step):
    """Save data from state."""

    def __init__(
        self,
        path: str | Path,
        source: str = "data.root",
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        save_meta: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.path: Path = Path(path)
        self.source = source
        self.fmt = fmt
        self.file_name = file_name or Path(self.source).suffix.lstrip(".")
        self.save_meta = save_meta
        self.kwargs = kwargs

        super().__init__(
            name=name or "save_data",
            requires=[self.source],
            provides=None,
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        # Default save name from the tail of the state key (e.g., "data.root" -> "root"),
        DataFrameRepository.save(
            obj,
            self.path,
            name=self.file_name,
            fmt=self.fmt,
            save_meta=self.save_meta,
            **self.kwargs,
        )
