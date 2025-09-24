from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

from torch import nn, Tensor
from ..batch import Batch


class ModelOutput(NamedTuple):
    logits: Tensor
    latents: Optional[Tensor] = None
    extras: Optional[Dict[str, Any]] = None


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
