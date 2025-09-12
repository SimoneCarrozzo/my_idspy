from typing import Optional

from torch.utils.data import Dataset

from ..helpers import validate_instance
from ...core.step import Step
from ...core.state import State


class BuildDataLoader(Step):
    """Build dataloader from state."""

    def __init__(
        self,
        source: str = "dataset",
        target: str = "dataloader",
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        name: Optional[str] = None,
    ) -> None:
        self.source = source
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        super().__init__(
            name=name or "build_dataset",
            requires=[self.source],
            provides=[self.target],
        )

    def run(self, state: State) -> None:
        obj = state[self.source]
        validate_instance(obj, Dataset, self.name)

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            obj,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        state[self.target] = dataloader
