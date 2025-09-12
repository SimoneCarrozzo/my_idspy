from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from ..helpers import validate_instance
from ...core.step import Step
from ...core.state import State
from .helpers import run_epoch


class TrainOneEpoch(Step):
    """Train model for one epoch."""

    def __init__(
        self,
        dataloader_source: str = "train.dataloader",
        model_source: str = "model",
        loss_source: str = "loss",
        optimizer_source: str = "optimizer",
        device_source: str = "device",
        profiler_source: Optional[str] = None,
        metrics_target: Optional[str] = "train.history",
        model_target: Optional[str] = None,
        log_dir: Optional[str] = None,
        log_prefix: str = "Train",
        clip_grad_max_norm: Optional[float] = 1.0,
        name: Optional[str] = None,
    ) -> None:
        self.sources = {
            "dataloader": dataloader_source,
            "model": model_source,
            "loss": loss_source,
            "optimizer": optimizer_source,
            "device": device_source,
            "profiler": profiler_source,
        }
        self.targets = {
            "model": model_target or model_source,
            "history": metrics_target,
        }
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir) if log_dir else None
        )
        self.log_prefix = log_prefix
        self.clip_grad_max_norm = clip_grad_max_norm

        super().__init__(
            requires=[v for v in self.sources.values() if v is not None],
            provides=list(self.targets.values()),
            name=name or "train_one_epoch",
        )

    def run(self, state: State) -> None:
        dataloader = state[self.sources["dataloader"]]
        model = state[self.sources["model"]]
        loss_fn = state[self.sources["loss"]]
        optimizer = state[self.sources["optimizer"]]
        device = state[self.sources["device"]]
        profiler = state[self.sources["profiler"]]

        validate_instance(dataloader, torch.utils.data.DataLoader, self.name)
        validate_instance(model, torch.nn.Module, self.name)
        validate_instance(loss_fn, torch.nn.Module, self.name)
        validate_instance(optimizer, torch.optim.Optimizer, self.name)
        validate_instance(device, torch.device, self.name)
        if profiler is not None:
            validate_instance(profiler, torch.profiler.profile, self.name)

        avg_loss, outputs_list = run_epoch(
            desc="Training",
            log_prefix=self.log_prefix,
            is_training=True,
            dataloader=dataloader,
            model=model,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            writer=self.writer,
            profiler=profiler,
            clip_grad_max_norm=self.clip_grad_max_norm,
        )

        state[self.targets["model"]] = model
        state.get_or_create(self.targets["history"], []).append(
            {
                "loss": avg_loss,
                "outputs": outputs_list,
            }
        )
