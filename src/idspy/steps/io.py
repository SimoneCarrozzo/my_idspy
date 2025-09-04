from pathlib import Path
from typing import Optional, Any

import pandas as pd

from .utils import validate_instance
from ..core.state import State
from ..core.step import Step
from ..data.repository import DataFrameRepository
from ..data.schema import Schema


class LoadTabularData(Step):
    """Load tabular data into state."""

    def __init__(
            self,
            path: str | Path,
            provide: str = "data.root",
            schema: Optional[Schema] = None,
            name: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        self.path: Path = Path(path)
        self.schema = schema
        self.provide = provide
        self.kwargs = kwargs

        super().__init__(
            name=name or "load_data",
            requires=None,
            provides=[provide],
        )

    def run(self, state: State) -> None:
        tab = DataFrameRepository.load(self.path, schema=self.schema, **self.kwargs)
        state[self.provide] = tab


class SaveTabularData(Step):
    """Save tabular data from state."""

    def __init__(
            self,
            path: str | Path,
            require: str = "data.root",
            name: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        self.path: Path = Path(path)
        self.require = require
        self.kwargs = kwargs

        super().__init__(
            name=name or "save_data",
            requires=[require],
            provides=None,
        )

    def run(self, state: State) -> None:
        df = state[self.require]
        validate_instance(df, pd.DataFrame, self.name)

        # Default save name from the tail of the state key (e.g., "data.root" -> "root"),
        # unless explicitly provided via kwargs.
        save_name = self.kwargs.pop("name", None) or Path(self.require).suffix.lstrip(".")

        DataFrameRepository.save(df, self.path, name=save_name, **self.kwargs)
