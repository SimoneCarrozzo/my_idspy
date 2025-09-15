from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import BaseLoss


class ClassificationLoss(BaseLoss):
    """Cross-entropy loss for classification with hard labels."""

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        class_weight: Optional[Tensor] = None,
    ) -> None:
        """Initialize classification loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss calculation
            label_smoothing: Label smoothing factor [0, 1)
            class_weight: Per-class weights for imbalanced datasets
        """
        super().__init__(reduction)

        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None

    def forward(
        self,
        pred: Tensor,
        target: Optional[Tensor],
    ) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            pred: Logits tensor [batch_size, num_classes]
            target: Target class indices [batch_size]

        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        if target is None:
            raise ValueError("Target cannot be None for classification loss")

        if target.dtype not in (torch.long, torch.int64, torch.int32):
            raise TypeError(f"Target must be integer type, got {target.dtype}")

        if target.dim() != 1:
            raise ValueError(f"Target must be 1D, got shape {target.shape}")

        if pred.size(0) != target.size(0):
            raise ValueError("Batch size mismatch between pred and target")

        # Move class weights to correct device if needed
        class_weight = self.class_weight
        if class_weight is not None and class_weight.device != pred.device:
            class_weight = class_weight.to(pred.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            pred,
            target,
            weight=class_weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # Handle ignored indices for proper reduction
        if self.reduction in ("mean", "sum"):
            valid_mask = target != self.ignore_index
            if self.reduction == "mean":
                return loss.sum() / valid_mask.sum().clamp_min(1)
            else:  # sum
                return loss.sum()

        return loss  # none
