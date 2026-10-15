"""Mean and Max pooling MIL baseline aggregators."""

from typing import Optional, Tuple
import numpy as np
from slideshift.core.types import PredictionOutput

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = hasattr(torch, "__version__") and hasattr(nn, "Module")
except (ImportError, AttributeError):
    torch = None
    nn = None
    HAS_TORCH = False


if HAS_TORCH:
    class MeanPoolingMIL(nn.Module):
        """Mean pooling baseline: averages all tile features and applies a linear classifier."""

        def __init__(self, input_dim: int, num_classes: int = 2, dropout: float = 0.25):
            super().__init__()
            self.input_dim = input_dim
            self.num_classes = num_classes
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_classes),
            )

        def forward(self, h: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Args:
                h: (N, D) tile embeddings.
            Returns:
                logits: (1, C)
                attention: (N,) uniform attention (1/N)
                slide_embedding: (1, D)
            """
            if isinstance(h, np.ndarray):
                dev = next(self.parameters()).device
                h = torch.from_numpy(h).float().to(dev)

            slide_emb = torch.mean(h, dim=0, keepdim=True)  # (1, D)
            logits = self.classifier(slide_emb)  # (1, C)
            n = h.shape[0]
            attention = torch.full((n,), 1.0 / max(1, n), device=h.device, dtype=h.dtype)
            return logits, attention, slide_emb


    class MaxPoolingMIL(nn.Module):
        """Max pooling baseline: takes element-wise maximum across tile features."""

        def __init__(self, input_dim: int, num_classes: int = 2, dropout: float = 0.25):
            super().__init__()
            self.input_dim = input_dim
            self.num_classes = num_classes
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_classes),
            )

        def forward(self, h: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            if isinstance(h, np.ndarray):
                dev = next(self.parameters()).device
                h = torch.from_numpy(h).float().to(dev)

            slide_emb, _ = torch.max(h, dim=0, keepdim=True)  # (1, D)
            logits = self.classifier(slide_emb)
            n = h.shape[0]
            attention = torch.full((n,), 1.0 / max(1, n), device=h.device, dtype=h.dtype)
            return logits, attention, slide_emb

else:
    # Minimal stub when torch is not loaded
    class MeanPoolingMIL:
        def __init__(self, input_dim: int, num_classes: int = 2, dropout: float = 0.25):
            self.input_dim = input_dim
            self.num_classes = num_classes

    class MaxPoolingMIL:
        def __init__(self, input_dim: int, num_classes: int = 2, dropout: float = 0.25):
            self.input_dim = input_dim
            self.num_classes = num_classes
