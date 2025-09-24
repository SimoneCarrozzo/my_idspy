from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from ...core.step import Step
from ...core.state import State
from ...nn.models.base import BaseModel
from ...nn.losses.base import BaseLoss
from ...nn.helpers import run_epoch


class TrainOneEpoch(Step):
    """Train model for one epoch."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_prefix: str = "train",
        clip_grad_max_norm: Optional[float] = 1.0,
        save_history: bool = False,
        save_outputs: bool = False,
        in_scope: str = "train",
        out_scope: str = "train",
        name: Optional[str] = None,
    ) -> None:
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir) if log_dir else None
        )
        self.log_prefix = log_prefix
        self.clip_grad_max_norm = clip_grad_max_norm
        self.save_history = save_history
        self.save_outputs = save_outputs

        super().__init__(
            name=name or "train_one_epoch",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(
        dataloader=torch.utils.data.DataLoader,
        model=BaseModel,
        loss=BaseLoss,
        optimizer=torch.optim.Optimizer,
        device=torch.device,
        history=list,
        outputs=list,
        epoch=int,
    )
    @Step.provides(model=BaseModel, history=list, outputs=list, epoch=int)
    def run(
        self,
        state: State,
        dataloader: torch.utils.data.DataLoader,
        model: BaseModel,
        loss: BaseLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        context: Optional[any] = None,
        history: list = [],
        outputs: list = [],
        epoch: int = 0,
    ) -> Optional[Dict[str, Any]]:

        average_loss, outputs_list = run_epoch(
            desc="Training",
            log_prefix=self.log_prefix,
            is_training=True,
            dataloader=dataloader,
            model=model,
            device=device,
            loss_fn=loss,
            optimizer=optimizer,
            writer=self.writer,
            profiler=context,
            clip_grad_max_norm=self.clip_grad_max_norm,
            save_outputs=self.save_outputs,
            epoch=epoch,
        )

        if self.writer is not None:
            self.writer.close()
        if self.save_history:
            history.append(average_loss)
        if self.save_outputs:
            outputs.append(outputs_list)

        return {
            "model": model,
            "history": history,
            "outputs": outputs,
            "epoch": epoch + 1,
        }
