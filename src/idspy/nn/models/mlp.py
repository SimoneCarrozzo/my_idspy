from typing import Callable, Optional, Sequence, Mapping, Any, Dict

import torch
from torch import nn

from .base import BaseModel, ModelOutput
from ..batch import Batch, ensure_batch


class MLP(BaseModel):
    """Multi-Layer Perceptron model."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int] = (),
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize MLP."""
        super().__init__()

        layers = []
        prev_dim = in_features

        # Build hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, out_features, bias=bias))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Mapping[str, Any]) -> ModelOutput:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, features]

        Returns:
            Model output with 'logits' keys
        """
        if not torch.is_tensor(x):
            raise TypeError("Expected tensor features")
        if x.dim() != 2:
            raise ValueError("Expected 2D tensor [batch_size, features]")

        logits = self.net(x)
        return {"logits": logits}

    def loss_inputs(self, output: ModelOutput, target: torch.Tensor) -> Dict[str, Any]:
        """Prepare loss function inputs.

        Args:
            output: Model output
            target: target tensor

        Returns:
            Loss function arguments
        """
        return {"pred": output["logits"], "target": target}
