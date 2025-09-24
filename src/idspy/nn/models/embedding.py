from typing import Sequence, Optional

import torch
from torch import nn, Tensor


class FeatureEmbedding(nn.Module):
    """Feature embedding for categorical variables with padding/unknown support."""

    def __init__(self, cardinalities: Sequence[int], emb_dim: Optional[int] = None):
        """Initialize feature embedding.

        Note:
            Index 0 is reserved for padding/unknown values and initialized to zeros.
        """
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.embedding_dims = []

        for card in cardinalities:
            dim = emb_dim or max(1, int(card**0.5))

            emb = nn.Embedding(card + 1, dim, padding_idx=0)
            self.embeddings.append(emb)
            self.embedding_dims.append(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Categorical features [batch_size, num_categorical_features]
               Values with 0 are treated as padding/unknown

        Returns:
            Embedded features [batch_size, sum(embedding_dims)]
        """
        embs = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(x[:, i]))
        return torch.cat(embs, dim=1)
