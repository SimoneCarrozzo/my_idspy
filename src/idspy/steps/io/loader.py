from pathlib import Path
from sre_parse import State
from typing import Optional, Any, Dict, Union

import pandas as pd

from ...core.step import Step
from ...data.repository import DataFrameRepository
from ...data.schema import Schema


class LoadData(Step):
    """Load data into state."""

    def __init__(
        self,
        path_in: Union[str, Path],
        schema: Optional[Schema] = None,
        load_meta: bool = True,
        in_scope: Optional[str] = "data",
        out_scope: Optional[str] = "data",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.path_in: Path = Path(path_in)
        self.schema = schema
        self.load_meta = load_meta
        self.kwargs = kwargs

        super().__init__(
            name=name or "load_data",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.provides(root=pd.DataFrame)
    def run(self, state: State) -> Optional[Dict[str, Any]]:
        dataframe: pd.DataFrame = DataFrameRepository.load(
            base_path=self.path_in,
            schema=self.schema,
            load_meta=self.load_meta,
            **self.kwargs,
        )
        return {"root": dataframe}
