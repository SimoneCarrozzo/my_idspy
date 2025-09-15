from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
from ..common.path import PathUtils, PathLike


def save_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    fmt: Optional[str] = "pt",
) -> None:
    """Save model, optimizer, scheduler, and extra data to a file."""
    final_path, _ = PathUtils.resolve_path_and_format(path, fmt=fmt)
    PathUtils.ensure_dir_exists(final_path)
    unwrapped = getattr(model, "module", model)
    payload: Dict[str, Any] = {"model": unwrapped.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, final_path)


def load_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = False,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load model, optimizer, scheduler, and extra data from a file."""
    final_path, _ = PathUtils.resolve_path_and_format(path)
    payload = torch.load(final_path, map_location=map_location)
    state_dict = payload.get("model", payload)
    unwrapped = getattr(model, "module", model)
    unwrapped.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except Exception:
            pass
    if (
        scheduler is not None
        and "scheduler" in payload
        and hasattr(scheduler, "load_state_dict")
    ):
        try:
            scheduler.load_state_dict(payload["scheduler"])
        except Exception:
            pass

    return payload.get("extra", {})


def save_weights(path: PathLike, model: nn.Module, fmt: Optional[str] = "pt") -> None:
    """Save only model weights to a file."""
    final_path, _ = PathUtils.resolve_path_and_format(path, fmt=fmt)
    PathUtils.ensure_dir_exists(final_path)
    unwrapped = getattr(model, "module", model)
    torch.save(unwrapped.state_dict(), final_path)


def load_weights(
    path: PathLike,
    model: nn.Module,
    strict: bool = False,
    map_location: Union[str, torch.device] = "cpu",
) -> Tuple[set, set]:
    """Load only model weights from a file."""
    final_path, _ = PathUtils.resolve_path_and_format(path)
    state = torch.load(final_path, map_location=map_location)
    state_dict = (
        state["model"] if isinstance(state, dict) and "model" in state else state
    )
    unwrapped = getattr(model, "module", model)
    missing, unexpected = unwrapped.load_state_dict(state_dict, strict=strict)
    return set(missing), set(unexpected)
