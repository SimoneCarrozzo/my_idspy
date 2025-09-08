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
            name: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        self.path: Path = Path(path)
        self.target = target
        self.schema = schema
        self.kwargs = kwargs

        super().__init__(
            name=name or "load_data",
            requires=None,
            provides=[self.target],
        )

    def run(self, state: State) -> None:
        df: pd.DataFrame = DataFrameRepository.load(self.path, schema=self.schema, **self.kwargs)
        state[self.target] = df


class SaveData(Step):
    """Save data from state."""

    def __init__(
            self,
            path: str | Path,
            source: str = "data.root",
            fmt: Optional[str] = None,
            file_name: Optional[str] = None,
            name: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        self.path: Path = Path(path)
        self.source = source
        self.fmt = fmt
        self.file_name = file_name or Path(self.source).suffix.lstrip(".")
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
        # unless explicitly provided via kwargs.
        DataFrameRepository.save(obj, self.path, name=self.file_name, fmt=self.fmt, **self.kwargs)


class SaveSplits(Step):
    """Save data splits from state."""

    def __init__(
            self,
            path: str | Path,
            source: str = "data.root",
            name: Optional[str] = None,
            fmt: Optional[str] = None,
            file_names: Optional[Mapping[str, str]] = None,
            **kwargs: Any,
    ) -> None:
        self.path: Path = Path(path)
        self.source = source
        self.fmt = fmt
        self.file_names = file_names or {}
        self.kwargs = kwargs

        super().__init__(
            name=name or "save_data",
            requires=[self.source],
            provides=None,
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)
        validate_split(obj, self.name)

        for s in SplitName:
            try:
                split_df = getattr(obj.tab, s.value)
            except KeyError:
                continue

            # Default save name from the tail of the state key (e.g., "data.root" -> "root"),
            # unless explicitly provided via kwargs.
            save_name = self.file_names.get(s.value, s.value)
            DataFrameRepository.save(split_df, self.path, name=save_name, fmt=self.fmt, **self.kwargs)
