"""CLAM: Clustering-constrained Attention Multiple Instance Learning (Lu et al., 2021)."""

from typing import Dict, List, Optional, Tuple
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
    class CLAM_SB(nn.Module):
        """CLAM Single-Branch (CLAM-SB). Uses a single attention network and instance clustering."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            dropout: float = 0.25,
            k_sample: int = 8,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.k_sample = k_sample

            # Projection
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Attention
            self.attention_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
            self.attention_u = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Sigmoid())
            self.attention_w = nn.Linear(hidden_dim // 2, 1)

            # Slide classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

            # Instance clustering classifiers (1 per class)
            self.instance_classifiers = nn.ModuleList([
                nn.Linear(hidden_dim, 2) for _ in range(num_classes)
            ])

        def forward(
            self,
            h: "torch.Tensor",
            label: Optional[int] = None,
            instance_eval: bool = False,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
            """
            Args:
                h: (N, D) tile embeddings.
                label: Ground truth slide label (needed for clustering loss).
                instance_eval: Whether to compute instance clustering loss.
            Returns:
                logits: (1, num_classes)
                attention: (N,) attention scores
                slide_embedding: (1, hidden_dim)
                instance_loss: scalar tensor clustering loss or None
            """
            if isinstance(h, np.ndarray):
                dev = next(self.parameters()).device
                h = torch.from_numpy(h).float().to(dev)

            h_proj = self.projector(h)  # (N, hidden_dim)

            # Gated attention
            a_v = self.attention_v(h_proj)
            a_u = self.attention_u(h_proj)
            a_raw = self.attention_w(a_v * a_u).squeeze(-1)  # (N,)
            a_weights = F.softmax(a_raw, dim=0)  # (N,)

            # Slide embedding & logits
            slide_emb = torch.sum(a_weights.unsqueeze(-1) * h_proj, dim=0, keepdim=True)
            logits = self.classifier(slide_emb)

            instance_loss = None
            if instance_eval and label is not None:
                instance_loss = self._compute_instance_clustering_loss(h_proj, a_weights, label)

            return logits, a_weights, slide_emb, instance_loss

        def _compute_instance_clustering_loss(
            self,
            h_proj: "torch.Tensor",
            a_weights: "torch.Tensor",
            label: int,
        ) -> "torch.Tensor":
            """Supervised instance clustering loss over top-k and bottom-k attended tiles."""
            n = h_proj.shape[0]
            k = min(self.k_sample, max(1, n // 4))

            # Sort attention scores
            sorted_idx = torch.argsort(a_weights, descending=True)
            top_k_idx = sorted_idx[:k]
            bottom_k_idx = sorted_idx[-k:]

            # For slide label c: top_k are strong positives for class c, bottom_k are negatives
            top_features = h_proj[top_k_idx]
            bottom_features = h_proj[bottom_k_idx]

            inst_classifier = self.instance_classifiers[label]
            top_preds = inst_classifier(top_features)
            bottom_preds = inst_classifier(bottom_features)

            # Targets: top_k -> 1 (positive instance), bottom_k -> 0 (negative instance)
            top_targets = torch.ones(k, dtype=torch.long, device=h_proj.device)
            bottom_targets = torch.zeros(k, dtype=torch.long, device=h_proj.device)

            loss = F.cross_entropy(top_preds, top_targets) + F.cross_entropy(bottom_preds, bottom_targets)
            return loss * 0.5


    class CLAM_MB(nn.Module):
        """CLAM Multi-Branch (CLAM-MB). Uses distinct attention heads per class."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            dropout: float = 0.25,
            k_sample: int = 8,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.k_sample = k_sample

            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Class-specific attention heads
            self.attention_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
            self.attention_u = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Sigmoid())
            self.attention_w = nn.Linear(hidden_dim // 2, num_classes)  # (N, C)

            # Class classifiers
            self.classifiers = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(num_classes)
            ])

        def forward(
            self,
            h: "torch.Tensor",
            label: Optional[int] = None,
            instance_eval: bool = False,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
            h_proj = self.projector(h)
            a_v = self.attention_v(h_proj)
            a_u = self.attention_u(h_proj)
            a_raw = self.attention_w(a_v * a_u)  # (N, C)
            a_weights = F.softmax(a_raw, dim=0)  # (N, C)

            slide_embs = []
            class_logits = []
            for c in range(self.num_classes):
                a_c = a_weights[:, c]  # (N,)
                emb_c = torch.sum(a_c.unsqueeze(-1) * h_proj, dim=0, keepdim=True)  # (1, hidden_dim)
                logit_c = self.classifiers[c](emb_c)  # (1, 1)
                slide_embs.append(emb_c)
                class_logits.append(logit_c)

            logits = torch.cat(class_logits, dim=-1)  # (1, C)
            mean_emb = torch.mean(torch.stack(slide_embs, dim=0), dim=0)
            primary_attention = a_weights[:, 1] if self.num_classes == 2 else a_weights[:, 0]

            return logits, primary_attention, mean_emb, None

else:
    class CLAM_SB:
        def __init__(self, *args, **kwargs): pass
    class CLAM_MB:
        def __init__(self, *args, **kwargs): pass
