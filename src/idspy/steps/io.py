from pathlib import Path
from typing import Optional, Dict, Any

from .utils import validate_instance
from ..core.state import State
from ..core.step import Step
from ..data.tabular_data import TabularSchema, TabularData
from ..data.tabular_repository import TabularDataRepository


class LoadTabularData(Step):
    """Pipeline step that loads tabular data into state."""

    def __init__(
            self,
            path: str | Path,
            output_key: str = "data",
            schema: Optional[TabularSchema] = None,
            pandas_kwargs: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
    ) -> None:
        self.path: Path = Path(path)
        self.output_key = output_key
        self.schema = schema
        self.pandas_kwargs: Dict[str, Any] = pandas_kwargs or {}

        super().__init__(
            name=name or "load_tabular",
            requires=None,
            produces=[self.output_key],
        )

    def _run(self, state: State) -> None:
        tab = TabularDataRepository.load(
            self.path,
            schema=self.schema,
            **self.pandas_kwargs,
        )
        state[self.output_key] = tab


class SaveTabularData(Step):
    """Pipeline step that saves tabular data from state."""

    def __init__(
            self,
            path: str | Path,
            input_key: str = "data",
            include_schema: bool = True,
            index: bool = False,
            pandas_kwargs: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
    ) -> None:
        self.path: Path = Path(path)
        self.input_key = input_key
        self.include_schema = include_schema
        self.index = index
        self.pandas_kwargs: Dict[str, Any] = pandas_kwargs or {}

        super().__init__(
            name=name or "save_tabular",
            requires=[self.input_key],
            produces=None,
        )

    def _run(self, state: State) -> None:
        tab: TabularData = state[self.input_key]
        validate_instance(tab, TabularData, self.name)

        TabularDataRepository.save(
            tab,
            self.path,
            include_schema=self.include_schema,
            index=self.index,
            **self.pandas_kwargs,
        )
