import pandas as pd

from src.idspy.core.state import State
from src.idspy.core.step import Step
from src.idspy.data.split import random_split, stratified_split
from src.idspy.steps.utils import validate_instance, validate_split


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
            source: str = "data.root",
            target: str | None = None,
            train_size: float = 0.7,
            val_size: float = 0.15,
            test_size: float = 0.15,
            random_state: int | None = None,
            name: str | None = None,
    ) -> None:
        self.source = source
        self.target = target or source
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        _validate_sizes(name or "random_split", train_size, val_size, test_size)

        super().__init__(
            name=name or "random_split",
            requires=[self.source],
            provides=[self.target, "mapping.split"],
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        validate_split(obj, self.name)

        if obj.empty:
            state["mapping.split"] = {}
            state[self.target] = obj
            return

        split_mapping = random_split(
            obj,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        obj.tab.set_splits_from_labels(split_mapping)
        state["mapping.split"] = split_mapping
        state[self.target] = obj


class StratifiedSplit(Step):
    """Stratified split into train/val/test."""

    def __init__(
            self,
            source: str = "data.root",
            target: str | None = None,
            train_size: float = 0.7,
            val_size: float = 0.15,
            test_size: float = 0.15,
            class_col: str | None = None,
            random_state: int | None = None,
            name: str | None = None,
    ) -> None:
        self.source = source
        self.target = target or source
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.class_col = class_col
        self.random_state = random_state

        _validate_sizes(name or "stratified_split", train_size, val_size, test_size)

        super().__init__(
            name=name or "stratified_split",
            requires=[self.source],
            provides=[self.target, "mapping.split"],
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, pd.DataFrame, self.name)

        validate_split(obj, self.name)

        if obj.empty:
            state["mapping.split"] = {}
            state[self.target] = obj
            return

        if not self.class_col:
            raise ValueError("stratified_split: 'class_col' must be provided.")

        split_mapping = stratified_split(
            obj,
            self.class_col,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        obj.tab.set_splits_from_labels(split_mapping)
        state["mapping.split"] = split_mapping
        state[self.target] = obj
