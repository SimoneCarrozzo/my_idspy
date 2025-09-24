from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

from torch import nn, Tensor
import torch
from ..batch import Batch


class ModelOutput(NamedTuple):
    logits: Tensor
    latents: Optional[Tensor] = None
    extras: Optional[Dict[str, Tensor]] = None

    def detach(self) -> "ModelOutput":
        return ModelOutput(
            logits=self.logits.detach(),
            latents=None if self.latents is None else self.latents.detach(),
            extras=(
                None
                if self.extras is None
                else {k: v.detach() for k, v in self.extras.items()}
            ),
        )

    def to(self, device: torch.device, non_blocking: bool = True) -> "ModelOutput":
        return ModelOutput(
            logits=self.logits.to(device, non_blocking=non_blocking),
            latents=(
                None
                if self.latents is None
                else self.latents.to(device, non_blocking=non_blocking)
            ),
            extras=(
                None
                if self.extras is None
                else {
                    k: v.to(device, non_blocking=non_blocking)
                    for k, v in self.extras.items()
                }
            ),
        )


class BaseModel(nn.Module):
    """
    Base class for models. Defines the interface for forward and loss_inputs methods.
    """

    def forward(self, batch: Batch | Mapping[str, Any]) -> ModelOutput:
        """
        Forward pass. Must be implemented by subclasses.
        Args:
            batch: Batch or mapping compatible with Batch.
        Returns:
            ModelOutput: NamedTuple, expected to contain at least 'logits'.
        """
        raise NotImplementedError

    def for_loss(
        self,
        output: ModelOutput,
        batch: Batch | Mapping[str, Any],
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepares arguments for the loss function. Default: pred=output['logits'], target=batch.target.
        Override if your model/loss requires different fields.
        """
        raise NotImplementedError
