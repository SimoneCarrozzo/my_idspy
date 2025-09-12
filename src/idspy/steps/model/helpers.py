from typing import Optional
from typing import Optional, List, Tuple

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter


def run_epoch(
    desc: str,
    log_prefix: str,
    is_training: bool,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    loss_fn: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    writer: Optional[SummaryWriter] = None,
    profiler: Optional[torch.profiler.profile] = None,
    clip_grad_max_norm: Optional[float] = 1.0,
) -> Tuple[float, List]:
    """Run one epoch of training or evaluation."""
    model.train(is_training)
    pbar = tqdm(dataloader, desc=desc, unit="batch")
    total_loss = 0.0
    outputs_list = []

    with torch.set_grad_enabled(is_training):
        for idx, batch in enumerate(pbar, start=1):
            inputs, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = (
                targets.to(device, non_blocking=True) if targets is not None else None
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            outputs_list.append(outputs)
            loss = (
                loss_fn(outputs, targets)
                if (loss_fn is not None and targets is not None)
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
                if loss is not None:
                    writer.add_scalar(f"{log_prefix}/Loss_batch", loss_value, idx)
                    writer.add_scalar(f"{log_prefix}/Loss_running", running_loss, idx)
                if is_training:
                    if grad_norm is not None:
                        writer.add_scalar(f"{log_prefix}/GradNorm", grad_norm, idx)
                    if hasattr(optimizer, "param_groups") and optimizer.param_groups:
                        lr = optimizer.param_groups[0].get("lr", None)
                        if lr is not None:
                            writer.add_scalar(f"{log_prefix}/LR", lr, idx)

    avg_loss = total_loss / max(1, len(dataloader))
    if writer is not None:
        writer.add_scalar(f"{log_prefix}/Loss_epoch_avg", avg_loss)
    return avg_loss, outputs_list
