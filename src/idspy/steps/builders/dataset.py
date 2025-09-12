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
    """Build dataset from state."""

    def __init__(
        self,
        source: str = "data.root",
        target: str = "dataset",
        name: Optional[str] = None,
    ) -> None:
        self.source = source
        self.target = target

        super().__init__(
            name=name or "build_dataset",
            requires=[self.source],
            provides=[self.target],
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        numerical_cols = obj.tab.schema.numerical
        categorical_cols = obj.tab.schema.categorical
        target_cols = obj.tab.schema.target

        if numerical_cols and categorical_cols:
            ds = MixedTabularDataset(
                obj,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                target_col=target_cols[0] if target_cols else None,
            )
        elif numerical_cols:
            ds = NumericalTensorDataset(
                obj,
                feature_cols=numerical_cols,
                target_col=target_cols[0] if target_cols else None,
            )
        elif categorical_cols:
            ds = CategoricalTensorDataset(
                obj,
                feature_cols=categorical_cols,
                target_col=target_cols[0] if target_cols else None,
            )
        else:
            raise ValueError(
                f"{self.name}: no numerical or categorical columns defined in schema."
            )

        state[self.target] = ds
