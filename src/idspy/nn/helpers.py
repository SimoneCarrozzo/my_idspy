from typing import Optional, List, Tuple

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from .models.base import BaseModel, ModelOutput
from .losses.base import BaseLoss
from .batch import Batch, ensure_batch


def run_epoch(
    desc: str,
    log_prefix: str,
    is_training: bool,
    dataloader: torch.utils.data.DataLoader,
    model: BaseModel,
    device: torch.device,
    loss_fn: Optional[BaseLoss] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    writer: Optional[SummaryWriter] = None,
    profiler: Optional[torch.profiler.profile] = None,
    clip_grad_max_norm: float = 1.0,
    save_outputs: bool = False,
    epoch: int = 0,
) -> Tuple[float, List]:
    """Run one epoch of training or evaluation."""
    model.train(mode=is_training)
    pbar = tqdm(dataloader, desc=desc, unit="batch")
    total_loss = 0.0
    outputs_list = []

    if is_training:
        assert optimizer is not None, "Optimizer is required for training"
        assert loss_fn is not None, "Loss function is required for training"

    grad_norm = None
    with torch.set_grad_enabled(is_training):
        for idx, batch in enumerate(pbar, start=1):
            batch: Batch = ensure_batch(batch)
            batch = batch.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs: ModelOutput = model(batch.features)
            # Debug prints momentaneo
            # print("target.min():", batch.target.min().item(), "target.max():", batch.target.max().item())
            # print("output.shape:", outputs.logits.shape if hasattr(outputs, "logits") else outputs.shape)

            if save_outputs:
                outputs_list.append(outputs.detach())

            loss = (
                loss_fn(*model.for_loss(outputs, batch))
                if (loss_fn is not None and batch.target is not None)
                else None
            )
            loss_value = loss.item() if loss is not None else 0.0

            if is_training and loss is not None:
                loss.backward()
                if clip_grad_max_norm is not None:
                    grad_norm = clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad_max_norm
                    )
                optimizer.step()

            total_loss += loss_value

            if idx % 100 == 0:
                running_loss = total_loss / idx
                pbar.set_postfix({"loss": f"{running_loss:.4f}"})

            if profiler is not None:
                profiler.step()

    avg_loss = total_loss / max(1, len(dataloader))
    if writer is not None:
        writer.add_scalar(f"{log_prefix}/loss_epoch_avg", avg_loss, epoch)
        writer.add_scalar(
            f"{log_prefix}/grad_norm",
            float(grad_norm) if grad_norm is not None else 0.0,
            epoch,
        )
        writer.add_scalar(
            f"{log_prefix}/lr",
            float(optimizer.param_groups[0]["lr"]) if optimizer is not None else 0.0,
            epoch,
        )

    return avg_loss, outputs_list


def get_device() -> torch.device:
    """Return the appropriate torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")
