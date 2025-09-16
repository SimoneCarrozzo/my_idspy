import pandas as pd

from ...core.state import State
from ...core.step import Step
from ...data.partition import random_split, stratified_split
from ...steps.helpers import validate_instance


def _validate_sizes(step: str, train: float, val: float, test: float) -> None:
    """Ensure sizes in [0,1] and sum to 1.0."""
    for label, v in (("train_size", train), ("val_size", val), ("test_size", test)):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{step}: {label} must be in [0, 1], got {v}.")
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{step}: sizes must sum to 1.0, got {total}.")


class RandomSplit(Step):
    """Random split into train/val/test."""

    def __init__(
        self,
        dataframe_in: str = "data.root",
        dataframe_out: str | None = None,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int | None = None,
        name: str | None = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        _validate_sizes(name or "random_split", train_size, val_size, test_size)

        super().__init__(
            name=name or "random_split",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out, "mapping.split"],
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        if dataframe.empty:
            state["mapping.split"] = {}
            state[self.dataframe_out] = dataframe
            return

        split_mapping = random_split(
            dataframe,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        dataframe.tab.set_partitions_from_labels(split_mapping)
        state["mapping.split"] = split_mapping
        state[self.dataframe_out] = dataframe


class StratifiedSplit(Step):
    """Stratified split into train/val/test."""

    def __init__(
        self,
        dataframe_in: str = "data.root",
        dataframe_out: str | None = None,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        class_column: str | None = None,
        random_state: int | None = None,
        name: str | None = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.class_column = class_column
        self.random_state = random_state

        _validate_sizes(name or "stratified_split", train_size, val_size, test_size)

        super().__init__(
            name=name or "stratified_split",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out, "mapping.split"],
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        if dataframe.empty:
            state["mapping.split"] = {}
            state[self.dataframe_out] = dataframe
            return

        if not self.class_column:
            raise ValueError("stratified_split: 'class_column' must be provided.")

        split_mapping = stratified_split(
            dataframe,
            self.class_column,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        dataframe.tab.set_partitions_from_labels(split_mapping)
        state["mapping.split"] = split_mapping
        state[self.dataframe_out] = dataframe


class AssignSplitPartitions(Step):
    def __init__(
        self,
        dataframe_in: str = "data.root",
        dataframe_out: str = "data",
        name: str | None = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out

        super().__init__(
            name=name or "assign_split_partitions",
            requires=[self.dataframe_in],
            provides=[
                self.dataframe_out + ".train",
                self.dataframe_out + ".train.target",
                self.dataframe_out + ".val",
                self.dataframe_out + ".val.target",
                self.dataframe_out + ".test",
                self.dataframe_out + ".test.target",
            ],
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        state[self.dataframe_out + ".train"] = dataframe.tab.train
        state[self.dataframe_out + ".train.target"] = (
            dataframe.tab.train.tab.target.values
        )
        state[self.dataframe_out + ".val"] = dataframe.tab.val
        state[self.dataframe_out + ".val.target"] = dataframe.tab.val.tab.target.values
        state[self.dataframe_out + ".test"] = dataframe.tab.test
        state[self.dataframe_out + ".test.target"] = (
            dataframe.tab.test.tab.target.values
        )
