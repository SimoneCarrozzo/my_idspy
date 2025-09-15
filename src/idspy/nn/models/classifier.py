from typing import Dict, Sequence, Optional, Callable, Mapping, Any

import torch
from torch import nn

from .base import BaseModel, ModelOutput
from .mlp import MLP
from .embedding import FeatureEmbedding


class MLPClassifier(BaseModel):
    """Two-stage MLP classifier with feature extraction and classification head."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)

        # Feature extraction
        feat_dim = hidden_dims[-1] if hidden_dims else in_features
        extractor_dims = hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims

        self.feature_extractor = MLP(
            in_features=in_features,
            out_features=feat_dim,
            hidden_dims=extractor_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

        # Classification head
        self.classifier_head = nn.Linear(feat_dim, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, features]

        Returns:
            Model output with 'logits' and 'features' keys
        """
        if not torch.is_tensor(x):
            raise TypeError("Expected tensor features")
        if x.dim() != 2:
            raise ValueError("Expected 2D tensor [batch_size, features]")

        features = self.feature_extractor(x)["logits"]
        logits = self.classifier_head(features)
        return {"logits": logits, "features": features}

    def loss_inputs(self, output: ModelOutput, target: torch.Tensor) -> Dict[str, Any]:
        """Prepare loss function inputs.

        Args:
            output: Model output
            target: target tensor

        Returns:
            Loss function arguments
        """
        return {"pred": output["logits"], "target": target}


class TabularClassifier(MLPClassifier):
    """Classifier for mixed tabular data (numerical + categorical features)."""

    def __init__(
        self,
        num_features: int,
        cat_cardinalities: Sequence[int],
        num_classes: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        emb_dim: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        embedding = FeatureEmbedding(list(cat_cardinalities), emb_dim=emb_dim)
        emb_dim_total = sum(embedding.embedding_dims)

        in_features = num_features + emb_dim_total
        super().__init__(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
            bias=bias,
        )
        self.embedding = embedding

    def forward(self, x: Mapping[str, Any]) -> ModelOutput:
        """Forward pass.

        Args:
            x: features dict containing 'num' and 'cat' keys

        Returns:
            Model output with 'logits' and 'features' keys
        """

        if not isinstance(x, Mapping):
            raise TypeError(
                "Expected features dict with 'numerical' and 'categorical' keys"
            )

        x_num = x["numerical"]
        x_cat = x["categorical"]

        cat_emb = self.embedding(x_cat)
        combined = torch.cat((x_num, cat_emb), dim=1)

        extracted_features = self.feature_extractor(combined)["logits"]
        logits = self.classifier_head(extracted_features)

        return {"logits": logits, "features": extracted_features}

    def loss_inputs(self, output: ModelOutput, target: torch.Tensor) -> Dict[str, Any]:
        """Prepare loss function inputs.

        Args:
            output: Model output
            target: target tensor

        Returns:
            Loss function arguments
        """
        return {"pred": output["logits"], "target": target}
