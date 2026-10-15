"""TransMIL: Transformer based Correlated Multiple Instance Learning (Shao et al., 2021)."""

from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = hasattr(torch, "__version__") and hasattr(nn, "Module")
except (ImportError, AttributeError):
    torch = None
    nn = None
    F = None
    HAS_TORCH = False


if HAS_TORCH:
    class TransMIL(nn.Module):
        """Transformer-based MIL aggregator modeling inter-tile morphological correlations."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            n_heads: int = 4,
            dropout: float = 0.10,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes

            # Projection to transformer dimension
            self.fc = nn.Linear(input_dim, hidden_dim)

            # Class token
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

            # Transformer encoder layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.norm = nn.LayerNorm(hidden_dim)

            # Classification head
            self.classifier = nn.Linear(hidden_dim, num_classes)

        def forward(self, h: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Args:
                h: (N, D) tile embeddings.
            Returns:
                logits: (1, num_classes)
                attention: (N,) relative importance of tiles to [CLS] token
                slide_embedding: (1, hidden_dim)
            """
            if isinstance(h, np.ndarray):
                dev = next(self.parameters()).device
                h = torch.from_numpy(h).float().to(dev)

            # Project tiles
            h_proj = self.fc(h).unsqueeze(0)  # (1, N, hidden_dim)

            # Prepend [CLS] token
            cls_tokens = self.cls_token.expand(1, -1, -1)  # (1, 1, hidden_dim)
            tokens = torch.cat((cls_tokens, h_proj), dim=1)  # (1, N+1, hidden_dim)

            # Transformer encoding
            out = self.transformer(tokens)  # (1, N+1, hidden_dim)
            out = self.norm(out)

            cls_out = out[:, 0, :]  # (1, hidden_dim)
            logits = self.classifier(cls_out)  # (1, num_classes)

            # Tile attention heuristic: cosine similarity between tile outputs and CLS token
            tile_embs = out[0, 1:, :]  # (N, hidden_dim)
            cos_sim = F.cosine_similarity(tile_embs, cls_out.squeeze(0), dim=-1)  # (N,)
            attention_weights = F.softmax(cos_sim * 2.0, dim=0)

            return logits, attention_weights, cls_out

else:
    class TransMIL:
        def __init__(self, *args, **kwargs): pass
