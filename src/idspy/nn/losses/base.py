from typing import Optional

from torch import nn, Tensor


class BaseLoss(nn.Module):
    """
    Base class for loss functions. Defines the interface for forward and reduction handling.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Initialize the loss with a reduction method.
        Args:
            reduction: 'mean', 'sum', or 'none'. Default is 'mean'.
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.reduction = reduction

    def _reduce(self, loss: Tensor) -> Tensor:
        """
        Apply the selected reduction to the loss tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'

    def forward(self, pred: Tensor, target: Optional[Tensor], **extras) -> Tensor:
        """
        Compute the loss. Must be implemented by subclasses.
        Args:
            pred: Model predictions (e.g., logits).
            target: Target tensor or None.
            **extras: Additional optional fields (e.g., mask, weights).
        Returns:
            Loss tensor (scalar or per-sample if reduction='none').
        """
        raise NotImplementedError
