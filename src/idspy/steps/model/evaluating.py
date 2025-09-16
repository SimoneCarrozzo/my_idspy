from typing import Callable, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from ..helpers import validate_instance
from ...core.state import State
from ...core.step import Step
from ...nn.models.base import BaseModel
from ...nn.losses.base import BaseLoss
from ...nn.batch import ensure_batch
from .helpers import run_epoch


class ValidateOneEpoch(Step):
    """Validate a model for one epoch (no gradient updates)."""

    def __init__(
        self,
        dataloader_in: str = "val.dataloader",
        model_in: str = "model",
        loss_in: str = "loss",
        device_in: str = "device",
        profiler_in: Optional[str] = None,
        history_out: Optional[str] = "history.val",
        log_dir: Optional[str] = None,
        log_prefix: str = "Val",
        name: Optional[str] = None,
    ) -> None:
        self.dataloader_in = dataloader_in
        self.model_in = model_in
        self.loss_in = loss_in
        self.device_in = device_in
        self.profiler_in = profiler_in
        self.history_out = history_out
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir) if log_dir else None
        )
        self.log_prefix = log_prefix

        requires = [
            self.dataloader_in,
            self.model_in,
            self.loss_in,
            self.device_in,
        ]
        if self.profiler_in is not None:
            requires.append(self.profiler_in)

        provides = (
            [self.history_out + ".outputs", self.history_out + ".loss"]
            if self.history_out is not None
            else []
        )

        super().__init__(
            requires=requires,
            provides=provides,
            name=name or "validate_one_epoch",
        )

    def run(self, state: State) -> None:
        dataloader = state[self.dataloader_in]
        model = state[self.model_in]
        loss_function = state[self.loss_in]
        device = state[self.device_in]
        profiler = state[self.profiler_in] if self.profiler_in is not None else None

        validate_instance(dataloader, torch.utils.data.DataLoader, self.name)
        validate_instance(model, BaseModel, self.name)
        validate_instance(loss_function, BaseLoss, self.name)
        validate_instance(device, torch.device, self.name)
        if profiler is not None:
            validate_instance(profiler, torch.profiler.profile, self.name)

        if self.history_out is not None:
            state[self.history_out + ".loss"] = []
            state[self.history_out + ".outputs"] = []

        average_loss, outputs_list = run_epoch(
            desc="Validation",
            log_prefix=self.log_prefix,
            is_training=False,
            dataloader=dataloader,
            model=model,
            device=device,
            loss_fn=loss_function,
            optimizer=None,
            writer=self.writer,
            profiler=profiler,
            clip_grad_max_norm=None,
        )

        if self.writer is not None:
            self.writer.close()

        if self.history_out is not None:
            state[self.history_out + ".loss"] = average_loss
            state[self.history_out + ".outputs"] = outputs_list


class ForwardOnce(Step):
    """Compute a single forward pass: model(input_tensor) -> output."""

    def __init__(
        self,
        batch_in: str = "forward.input",
        model_in: str = "model",
        device_in: str = "device",
        batch_out: str = "forward.output",
        to_cpu: bool = False,  # move output to CPU before storing
        name: Optional[str] = None,
    ) -> None:
        self.batch_in = batch_in
        self.model_in = model_in
        self.device_in = device_in
        self.batch_out = batch_out
        self.to_cpu = to_cpu

        requires = [self.batch_in, self.model_in, self.device_in]
        provides = [self.batch_out]

        super().__init__(
            requires=requires,
            provides=provides,
            name=name or "forward_once",
        )

    def run(self, state: State) -> None:
        batch = state[self.batch_in]
        model = state[self.model_in]
        device = state[self.device_in]

        batch = ensure_batch(batch)
        validate_instance(model, BaseModel, self.name)
        validate_instance(device, torch.device, self.name)

        model.eval()
        batch = batch.to(device, non_blocking=True)

        with torch.no_grad():
            out = model(batch.features)

        if self.to_cpu and torch.is_tensor(out):
            out = out.detach().cpu()

        state[self.batch_out] = out


class MakePredictions(Step):
    """Make predictions from model outputs."""

    def __init__(
        self,
        pred_fn: Callable,
        outputs_in: str = "history.val.outputs",
        outputs_key: str = "logits",
        pred_out: str = "history.val.preds",
        name: Optional[str] = None,
    ) -> None:
        self.outputs_in = outputs_in
        self.pred_out = pred_out
        self.pred_fn = pred_fn
        self.outputs_key = outputs_key

        super().__init__(
            requires=[self.outputs_in],
            provides=[self.pred_out],
            name=name or "make_predictions",
        )

    def run(self, state: State) -> None:
        outputs = state[self.outputs_in]
        validate_instance(outputs, list, self.name)

        predictions = []
        for output in outputs:
            output = output[self.outputs_key]
            curr_pred = self.pred_fn(output)
            if not torch.is_tensor(curr_pred):
                raise TypeError(
                    f"Expected tensor predictions from 'pred_fn' for step '{self.name}'."
                )
            predictions.append(curr_pred)
        predictions = torch.cat(predictions, dim=0)

        state[self.pred_out] = predictions.detach().cpu().numpy()
