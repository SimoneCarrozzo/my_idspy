from typing import Any, Dict, Mapping

from torch import nn, Tensor
from ..batch import Batch


ModelOutput = Dict[str, Tensor]


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
            ModelOutput: dict, expected to contain at least 'logits'.
        """
        raise NotImplementedError

    def loss_inputs(
        self,
        output: ModelOutput,
        batch: Batch | Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepares arguments for the loss function. Default: pred=output['logits'], target=batch.target.
        Override if your model/loss requires different fields.
        """
        raise NotImplementedError
