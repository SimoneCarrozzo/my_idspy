from typing import Optional, Any, Dict

import pandas as pd
from torch.utils.data import Dataset

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
        in_scope: str = "data",
        out_scope: str = "",
        name: Optional[str] = None,
    ) -> None:

        super().__init__(
            name=name or "build_dataset",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(dataset=Dataset)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        numerical_cols = root.tab.schema.numerical
        categorical_cols = root.tab.schema.categorical
        target_col = root.tab.schema.target

        if numerical_cols and categorical_cols:
            dataset = MixedTabularDataset(
                root,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                target_col=target_col if target_col else None,
            )
        elif numerical_cols:
            dataset = NumericalTensorDataset(
                root,
                feature_cols=numerical_cols,
                target_col=target_col if target_col else None,
            )
        elif categorical_cols:
            dataset = CategoricalTensorDataset(
                root,
                feature_cols=categorical_cols,
                target_col=target_col if target_col else None,
            )
        else:
            raise ValueError(
                f"{self.name}: no numerical or categorical columns defined in schema."
            )

        return {"dataset": dataset}
