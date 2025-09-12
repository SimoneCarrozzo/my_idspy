from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from ..helpers import validate_instance
from ...core.state import State
from ...core.step import Step
from .helpers import run_epoch


class ValidateOneEpoch(Step):
    """Validate a model for one epoch (no gradient updates)."""

    def __init__(
        self,
        dataloader_source: str = "val.dataloader",
        model_source: str = "model",
        loss_source: str = "loss",
        device_source: str = "device",
        profiler_source: Optional[str] = None,
        metrics_target: Optional[str] = "val.history",
        log_dir: Optional[str] = None,
        log_prefix: str = "Val",
        name: Optional[str] = None,
    ) -> None:
        self.sources = {
            "dataloader": dataloader_source,
            "model": model_source,
            "loss": loss_source,
            "device": device_source,
            "profiler": profiler_source,
        }
        self.targets = {"history": metrics_target}
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir) if log_dir else None
        )
        self.log_prefix = log_prefix

        super().__init__(
            requires=[v for v in self.sources.values() if v is not None],
            provides=list(self.targets.values()),
            name=name or "validate_one_epoch",
        )

    def run(self, state: State) -> None:
        dataloader = state[self.sources["dataloader"]]
        model = state[self.sources["model"]]
        loss_fn = state[self.sources["loss"]]
        device = state[self.sources["device"]]
        profiler = state[self.sources["profiler"]]

        validate_instance(dataloader, torch.utils.data.DataLoader, self.name)
        validate_instance(model, torch.nn.Module, self.name)
        validate_instance(loss_fn, torch.nn.Module, self.name)
        validate_instance(device, torch.device, self.name)
        if profiler is not None:
            validate_instance(profiler, torch.profiler.profile, self.name)

        avg_loss, outputs_list = run_epoch(
            desc="Validation",
            log_prefix=self.log_prefix,
            is_training=False,
            dataloader=dataloader,
            model=model,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
            writer=self.writer,
            profiler=profiler,
            clip_grad_max_norm=None,
        )

        state.get_or_create(self.targets["history"], []).append(
            {
                "loss": avg_loss,
                "outputs": outputs_list,
            }
        )


class ForwardOnce(Step):
    """Compute a single forward pass: model(input_tensor) -> output."""

    def __init__(
        self,
        input_source: str = "forward.input",
        model_source: str = "model",
        device_source: str = "device",
        output_target: str = "forward.output",
        to_cpu: bool = False,  # move output to CPU before storing
        name: Optional[str] = None,
    ) -> None:
        self.sources = {
            "input": input_source,
            "model": model_source,
            "device": device_source,
        }
        self.targets = {"output": output_target}
        self.to_cpu = to_cpu

        super().__init__(
            requires=[v for v in self.sources.values() if v is not None],
            provides=list(self.targets.values()),
            name=name or "forward_once",
        )

    def run(self, state: State) -> None:
        x = state[self.sources["input"]]
        model = state[self.sources["model"]]
        device = state[self.sources["device"]]

        validate_instance(model, torch.nn.Module, self.name)
        validate_instance(device, torch.device, self.name)

        model.eval()
        x = x.to(device, non_blocking=True)

        with torch.no_grad():
            y = model(x)

        if self.to_cpu and torch.is_tensor(y):
            y = y.detach().cpu()

        state[self.targets["output"]] = y
