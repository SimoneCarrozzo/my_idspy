from typing import Optional

import pandas as pd

from ..helpers import validate_instance
from ...core.step import Step
from ...core.state import State
from ...data.dataset import (
    CategoricalTensorDataset,
    NumericalTensorDataset,
    MixedTabularDataset,
)


class BuildDataset(Step):
    """Build dataset from dataframe in state."""

    def __init__(
        self,
        dataframe_in: str = "data.root",
        dataset_out: str = "dataset",
        name: Optional[str] = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataset_out = dataset_out

        super().__init__(
            name=name or "build_dataset",
            requires=[self.dataframe_in],
            provides=[self.dataset_out],
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        numerical_cols = dataframe.tab.schema.numerical
        categorical_cols = dataframe.tab.schema.categorical
        target_cols = dataframe.tab.schema.target

        if numerical_cols and categorical_cols:
            dataset = MixedTabularDataset(
                dataframe,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                target_col=target_cols[0] if target_cols else None,
            )
        elif numerical_cols:
            dataset = NumericalTensorDataset(
                dataframe,
                feature_cols=numerical_cols,
                target_col=target_cols[0] if target_cols else None,
            )
        elif categorical_cols:
            dataset = CategoricalTensorDataset(
                dataframe,
                feature_cols=categorical_cols,
                target_col=target_cols[0] if target_cols else None,
            )
        else:
            raise ValueError(
                f"{self.name}: no numerical or categorical columns defined in schema."
            )

        state[self.dataset_out] = dataset
