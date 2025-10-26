from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from ...core.step import Step
from ...core.state import State
from ...data.partition import random_split, stratified_split


class RandomSplit(Step):
    """Random split into train/val/test."""

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        super().__init__(
            name=name or "random_split",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame, seed=int)
    @Step.provides(root=pd.DataFrame, split_mapping=dict)
    def run(
        self, state: State, root: pd.DataFrame, seed: int
    ) -> Optional[Dict[str, Any]]:

        if root.empty:
            return {"split_mapping": {}, "root": root}

        split_mapping = random_split(
            root,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=seed,
        )

        root.tab.set_partitions_from_labels(split_mapping)
        return {"split_mapping": split_mapping, "root": root}


class StratifiedSplit(Step):
    """Stratified split into train/val/test."""

    def __init__(
        self,
        class_column: str,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.class_column = class_column

        super().__init__(
            name=name or "stratified_split",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame, seed=int)
    @Step.provides(root=pd.DataFrame, split_mapping=dict)
    def run(
        self, state: State, root: pd.DataFrame, seed: int
    ) -> Optional[Dict[str, Any]]:

        if not isinstance(self.class_column, str):
            raise ValueError("stratified_split: 'class_column' must be a string.")

        if root.empty:
            return {"split_mapping": {}, "root": root}

        split_mapping = stratified_split(
            root,
            self.class_column,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=seed,
        )

        root.tab.set_partitions_from_labels(split_mapping)
        return {"split_mapping": split_mapping, "root": root}


class AssignSplitPartitions(Step):  #estrae le partizioni train/val/test dal DataFrame già splittato e le salva nello State con chiavi separate
    """Assign split partitions to separate keys in the State."""
    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or "assign_split_partitions",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        return {"train": root.tab.train, "val": root.tab.val, "test": root.tab.test}


class AssignSplitTarget(Step):
    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "test",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or "assign_split_target",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(targets=np.ndarray)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        return {"targets": root.tab.target.to_numpy()}
