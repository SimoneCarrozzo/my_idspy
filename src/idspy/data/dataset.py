from typing import NamedTuple, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularFeatures(NamedTuple):
    numerical: torch.Tensor
    categorical: torch.Tensor

    def to(self, *args, **kwargs) -> "TabularFeatures":
        return TabularFeatures(
            self.numerical.to(*args, **kwargs),
            self.categorical.to(*args, **kwargs),
        )


class TensorDataset(Dataset):
    """
    Wraps a pandas DataFrame (and optional pandas Series) into torch tensors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        feature_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.long,
    ) -> None:
        self.features: torch.Tensor = torch.as_tensor(df.values, dtype=feature_dtype)
        self.target: Optional[torch.Tensor] = (
            torch.as_tensor(target.values, dtype=target_dtype)
            if target is not None
            else None
        )

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        target = self.target[index] if self.target is not None else features
        return features, target


class NumericalTensorDataset(TensorDataset):
    """
    Dataset wrapper for numerical features in a DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: Optional[str] = None,
    ) -> None:
        df = df[feature_cols]
        target = df[target_col] if target_col else None
        super().__init__(
            df,
            target,
            feature_dtype=torch.float32,
            target_dtype=torch.long,
        )


class CategoricalTensorDataset(TensorDataset):
    """
    Dataset wrapper for categorical features in a DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: Optional[str] = None,
    ) -> None:
        df = df[feature_cols]
        target = df[target_col] if target_col else None

        super().__init__(
            df,
            target,
            feature_dtype=torch.long,
            target_dtype=torch.long,
        )


class MixedTabularDataset(Dataset):
    """
    Combines numerical and categorical datasets into a single dataset yielding TabularSample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        numerical_cols: Sequence[str],
        categorical_cols: Sequence[str],
        target_col: Optional[str] = None,
    ) -> None:
        self.numerical_ds = NumericalTensorDataset(df, numerical_cols, target_col=None)
        self.categorical_ds = CategoricalTensorDataset(
            df, categorical_cols, target_col=None
        )

        self.target: Optional[torch.Tensor] = (
            torch.as_tensor(df[target_col].values, dtype=torch.long)
            if target_col
            else None
        )

    def __len__(self) -> int:
        return len(self.numerical_ds)

    def __getitem__(self, index: int):
        numerical_features, _ = self.numerical_ds[index]
        categorical_features, _ = self.categorical_ds[index]
        features = TabularFeatures(numerical_features, categorical_features)
        target = self.target[index] if self.target is not None else features
        return features, target
