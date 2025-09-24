from typing import Optional, Callable, Any, Dict

from torch.utils.data import Dataset, DataLoader

from ...core.step import Step
from ...core.state import State


class BuildDataLoader(Step):
    """Build dataloader from dataset in state."""

    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        in_scope: str = "",
        out_scope: str = "",
        name: Optional[str] = None,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn

        super().__init__(
            name=name or "build_dataloader",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(dataset=Dataset)
    @Step.provides(dataloader=DataLoader)
    def run(self, state: State, dataset: Dataset) -> Optional[Dict[str, Any]]:
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )
        return {"dataloader": dataloader}
