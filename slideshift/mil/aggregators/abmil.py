"""Attention-based Multiple Instance Learning (ABMIL) - Ilse et al. 2018."""

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
    class AttentionMIL(nn.Module):
        """Attention-based MIL with gated attention mechanism (Ilse et al., 2018).
        
        Formula:
            a = softmax(w^T (tanh(V h) * sigmoid(U h)))
            z = sum(a_k * h_k)
            y = classifier(z)
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            dropout: float = 0.25,
            gated: bool = True,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.gated = gated

            # Dimension reduction projection
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Attention network
            self.attention_v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
            )
            if gated:
                self.attention_u = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.Sigmoid(),
                )
            self.attention_w = nn.Linear(hidden_dim // 2, 1)

            # Slide-level classifier head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, h: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Args:
                h: (N, input_dim) tile embeddings.
            Returns:
                logits: (1, num_classes)
                attention: (N,) attention weights summing to 1.0
                slide_embedding: (1, hidden_dim)
            """
            if isinstance(h, np.ndarray):
                dev = next(self.parameters()).device
                h = torch.from_numpy(h).float().to(dev)

            # Project tiles
            h_proj = self.projector(h)  # (N, hidden_dim)

            # Attention scores
            a_v = self.attention_v(h_proj)  # (N, hidden_dim // 2)
            if self.gated:
                a_u = self.attention_u(h_proj)  # (N, hidden_dim // 2)
                a_gate = a_v * a_u
            else:
                a_gate = a_v

            a_raw = self.attention_w(a_gate).squeeze(-1)  # (N,)
            # Numerically stable softmax
            a_weights = F.softmax(a_raw, dim=0)  # (N,)

            # Weighted aggregation
            slide_emb = torch.sum(a_weights.unsqueeze(-1) * h_proj, dim=0, keepdim=True)  # (1, hidden_dim)

            # Slide classification
            logits = self.classifier(slide_emb)  # (1, num_classes)

            return logits, a_weights, slide_emb

else:
    class AttentionMIL:
        def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.25, gated: bool = True):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.gated = gated
