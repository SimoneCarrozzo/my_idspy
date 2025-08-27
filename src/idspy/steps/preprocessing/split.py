from typing import Final

from src.idspy.core.state import State
from src.idspy.core.step import Step
from src.idspy.data.split import random_split, stratified_split
from src.idspy.data.tabular_data import Data, DataView
from src.idspy.steps.utils import validate_instance


def _validate_sizes(step: str, train: float, val: float, test: float) -> None:
    """Ensure non-negative sizes that sum ~ 1.0."""
    for label, v in (("train_size", train), ("val_size", val), ("test_size", test)):
        if v < 0 or v > 1:
            raise ValueError(f"{step}: {label} must be in [0, 1], got {v}.")
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{step}: sizes must sum to 1.0, got {total}.")


class RandomSplit(Step):
    """Randomly split `Data` into train/val/test."""

    def __init__(
            self,
            input_key: str = "data.default",
            train_key: str = "data.train",
            val_key: str = "data.val",
            test_key: str = "data.test",
            train_size: float = 0.7,
            val_size: float = 0.15,
            test_size: float = 0.15,
            random_state: int | None = None,
            name: str | None = None,
    ) -> None:
        self.input_key: Final[str] = input_key
        self.train_key: Final[str] = train_key
        self.val_key: Final[str] = val_key
        self.test_key: Final[str] = test_key
        self.train_size: Final[float] = train_size
        self.val_size: Final[float] = val_size
        self.test_size: Final[float] = test_size
        self.random_state: Final[int | None] = random_state

        _validate_sizes(name or "random_split", train_size, val_size, test_size)

        super().__init__(
            name=name or "random_split",
            requires=[self.input_key],
            produces=[self.train_key, self.val_key, self.test_key],
        )

    def _run(self, state: State) -> None:
        data: Data = state[self.input_key]
        validate_instance(data, (Data, DataView), self.name)

        train_idx, val_idx, test_idx = random_split(
            data.df,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        state[self.train_key] = data.view(train_idx)
        state[self.val_key] = data.view(val_idx)
        state[self.test_key] = data.view(test_idx)


class StratifiedSplit(Step):
    """Stratified split on `target` into train/val/test."""

    def __init__(
            self,
            target: str,
            input_key: str = "data.default",
            train_key: str = "data.train",
            val_key: str = "data.val",
            test_key: str = "data.test",
            train_size: float = 0.7,
            val_size: float = 0.15,
            test_size: float = 0.15,
            random_state: int | None = None,
            name: str | None = None,
    ) -> None:
        self.input_key: Final[str] = input_key
        self.train_key: Final[str] = train_key
        self.val_key: Final[str] = val_key
        self.test_key: Final[str] = test_key
        self.train_size: Final[float] = train_size
        self.val_size: Final[float] = val_size
        self.test_size: Final[float] = test_size
        self.target: Final[str] = target
        self.random_state: Final[int | None] = random_state

        _validate_sizes(name or "stratified_split", train_size, val_size, test_size)

        super().__init__(
            name=name or "stratified_split",
            requires=[self.input_key],
            produces=[self.train_key, self.val_key, self.test_key],
        )

    def _run(self, state: State) -> None:
        data: Data = state[self.input_key]
        validate_instance(data, (Data, DataView), self.name)

        train_idx, val_idx, test_idx = stratified_split(
            data.df,
            self.target,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        state[self.train_key] = data.view(train_idx)
        state[self.val_key] = data.view(val_idx)
        state[self.test_key] = data.view(test_idx)
