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
        class_weight = self.class_weight

        loss = F.cross_entropy(
            pred,
            target,
            weight=class_weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        valid_mask = target != self.ignore_index
        # 🔍 Debug prints per capire l'origine del CUDA error
        # print("valid_mask.shape:", valid_mask.shape)
        # print("loss.shape:", loss.shape)
        # print("valid_mask.sum():", valid_mask.sum().item())
        # print("Any valid:", valid_mask.any().item())
        # print("target.min():", target.min().item(), "target.max():", target.max().item())
        # print("pred.shape:", pred.shape)
        if target.max() >= pred.shape[1] or target.min() < 0:
            raise ValueError(f"Target out of bounds: {target.min()}–{target.max()}, "
                     f"expected in [0, {pred.shape[1]-1}]")


        loss = loss[valid_mask]
        return self._reduce(loss)
