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
        dataloader_in: str = "train.dataloader",
        model_in: str = "model",
        loss_in: str = "loss",
        optimizer_in: str = "optimizer",
        device_in: str = "device",
        profiler_in: Optional[str] = None,
        metrics_out: Optional[str] = "train.history",
        model_out: Optional[str] = None,
        log_dir: Optional[str] = None,
        log_prefix: str = "Train",
        clip_grad_max_norm: Optional[float] = 1.0,
        name: Optional[str] = None,
    ) -> None:
        self.dataloader_in = dataloader_in
        self.model_in = model_in
        self.loss_in = loss_in
        self.optimizer_in = optimizer_in
        self.device_in = device_in
        self.profiler_in = profiler_in
        self.metrics_out = metrics_out
        self.model_out = model_out or model_in
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir) if log_dir else None
        )
        self.log_prefix = log_prefix
        self.clip_grad_max_norm = clip_grad_max_norm

        requires = [
            self.dataloader_in,
            self.model_in,
            self.loss_in,
            self.optimizer_in,
            self.device_in,
        ]
        if self.profiler_in is not None:
            requires.append(self.profiler_in)

        provides = (
            [self.model_out, self.metrics_out]
            if self.metrics_out is not None
            else [self.model_out]
        )

        super().__init__(
            requires=requires,
            provides=provides,
            name=name or "train_one_epoch",
        )

    def run(self, state: State) -> None:
        dataloader = state[self.dataloader_in]
        model = state[self.model_in]
        loss_function = state[self.loss_in]
        optimizer = state[self.optimizer_in]
        device = state[self.device_in]
        profiler = state[self.profiler_in] if self.profiler_in is not None else None

        validate_instance(dataloader, torch.utils.data.DataLoader, self.name)
        validate_instance(model, torch.nn.Module, self.name)
        validate_instance(loss_function, torch.nn.Module, self.name)
        validate_instance(optimizer, torch.optim.Optimizer, self.name)
        validate_instance(device, torch.device, self.name)
        if profiler is not None:
            validate_instance(profiler, torch.profiler.profile, self.name)

        average_loss, outputs_list = run_epoch(
            desc="Training",
            log_prefix=self.log_prefix,
            is_training=True,
            dataloader=dataloader,
            model=model,
            device=device,
            loss_fn=loss_function,
            optimizer=optimizer,
            writer=self.writer,
            profiler=profiler,
            clip_grad_max_norm=self.clip_grad_max_norm,
        )

        state[self.model_out] = model
        state.get_or_create(self.metrics_out, []).append(
            {
                "loss": average_loss,
                "outputs": outputs_list,
            }
        )
