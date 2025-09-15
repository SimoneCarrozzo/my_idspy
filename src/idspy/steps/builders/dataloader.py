from typing import Optional, Callable

from torch.utils.data import Dataset

from ..helpers import validate_instance
from ...core.step import Step
from ...core.state import State


class BuildDataLoader(Step):
    """Build dataloader from dataset in state."""

    def __init__(
        self,
        dataset_in: str = "dataset",
        dataloader_out: str = "dataloader",
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        name: Optional[str] = None,
    ) -> None:
        self.dataset_in = dataset_in
        self.dataloader_out = dataloader_out
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn

        super().__init__(
            name=name or "build_dataloader",
            requires=[self.dataset_in],
            provides=[self.dataloader_out],
        )

    def run(self, state: State) -> None:
        dataset = state[self.dataset_in]
        validate_instance(dataset, Dataset, self.name)

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )
        state[self.dataloader_out] = dataloader
