from typing import Optional, List, Tuple

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from ...nn.models.base import BaseModel
from ...nn.losses.base import BaseLoss
from ...nn.batch import Batch, ensure_batch


def detach_to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: detach_to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [detach_to_cpu(v) for v in x]
        return type(x)(t) if isinstance(x, tuple) else t
    return x


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

    with torch.set_grad_enabled(is_training):
        for idx, batch in enumerate(pbar, start=1):
            batch: Batch = ensure_batch(batch)
            batch = batch.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(batch.features)
            if save_outputs:
                outputs_list.append(detach_to_cpu(outputs))

            loss = (
                loss_fn(**model.loss_inputs(outputs, batch.target))
                if (loss_fn is not None and batch.target is not None)
                else None
            )
            loss_value = float(loss.item()) if loss is not None else 0.0

            grad_norm = None
            if is_training and loss is not None:
                loss.backward()
                if clip_grad_max_norm is not None:
                    grad_norm = clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad_max_norm
                    )
                optimizer.step()

            total_loss += loss_value
            running_loss = total_loss / idx
            pbar.set_postfix({"loss": f"{running_loss:.4f}"})

            if profiler is not None and hasattr(profiler, "step"):
                profiler.step()

            if writer is not None:
                global_step = epoch * len(dataloader) + idx
                if loss is not None:
                    writer.add_scalar(
                        f"{log_prefix}/loss_batch", loss_value, global_step
                    )
                    writer.add_scalar(
                        f"{log_prefix}/loss_running", running_loss, global_step
                    )
                if is_training:
                    if grad_norm is not None:
                        writer.add_scalar(
                            f"{log_prefix}/grad_norm", float(grad_norm), global_step
                        )
                    if hasattr(optimizer, "param_groups") and optimizer.param_groups:
                        lr = optimizer.param_groups[0].get("lr", None)
                        if lr is not None:
                            writer.add_scalar(
                                f"{log_prefix}/lr", float(lr), global_step
                            )

    avg_loss = total_loss / max(1, len(dataloader))
    if writer is not None:
        writer.add_scalar(f"{log_prefix}/loss_epoch_avg", float(avg_loss), epoch)
    return avg_loss, outputs_list
