"""DTFD-MIL: Double-Tier Feature Distillation for Multiple Instance Learning.

Reference:
    Zhang et al., "DTFD-MIL: Double-Tier Feature Distillation Multiple Instance
    Learning for Histopathology Whole Slide Image Classification", CVPR 2022.

Key ideas:
  - Tier 1: Split a slide bag into pseudo-bags via Bernoulli sampling.
  - Tier 2: Two-stage attention over pseudo-bag representations.
  - Feature distillation loss: pseudo-bag representations should be consistent
    with the full slide representation.
"""

from typing import List, Optional, Tuple
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


def _make_pseudo_bags(
    features: "torch.Tensor",
    num_bags: int = 4,
    sampling_prob: float = 0.5,
    rng: Optional["torch.Generator"] = None,
) -> List["torch.Tensor"]:
    """Bernoulli-sample pseudo-bags from a slide bag.

    Each tile is independently assigned to each pseudo-bag with probability
    ``sampling_prob``. Bags with zero tiles get one random tile to avoid
    empty-bag pathologies.
    """
    N = features.shape[0]
    pseudo_bags: List["torch.Tensor"] = []

    if rng is not None:
        mask = torch.bernoulli(
            torch.full((num_bags, N), sampling_prob), generator=rng
        ).bool()  # (num_bags, N)
    else:
        mask = torch.bernoulli(torch.full((num_bags, N), sampling_prob)).bool()

    for b in range(num_bags):
        idx = mask[b].nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            idx = torch.randint(N, (1,))
        pseudo_bags.append(features[idx])

    return pseudo_bags


if HAS_TORCH:
    class _GatedAttentionNet(nn.Module):
        """Small gated attention module shared across tiers."""

        def __init__(self, in_dim: int, hidden: int = 128):
            super().__init__()
            self.v = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
            self.u = nn.Sequential(nn.Linear(in_dim, hidden), nn.Sigmoid())
            self.w = nn.Linear(hidden, 1)

        def forward(self, h: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Returns (weighted_sum, attention_weights)."""
            a = self.w(self.v(h) * self.u(h)).squeeze(-1)  # (N,)
            weights = F.softmax(a, dim=0)                  # (N,)
            agg = torch.sum(weights.unsqueeze(-1) * h, dim=0, keepdim=True)  # (1, D)
            return agg, weights

    class DTFDMILTier1(nn.Module):
        """Tier-1: aggregate each pseudo-bag independently."""

        def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.attn = _GatedAttentionNet(hidden_dim, hidden_dim // 2)

        def forward(self, pseudo_bag: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            h = self.proj(pseudo_bag)
            rep, weights = self.attn(h)
            return rep, weights  # (1, hidden_dim), (N,)

    class DTFDMILTier2(nn.Module):
        """Tier-2: aggregate pseudo-bag representations into slide representation."""

        def __init__(self, hidden_dim: int = 256, dropout: float = 0.25):
            super().__init__()
            self.attn = _GatedAttentionNet(hidden_dim, hidden_dim // 2)

        def forward(self, bag_reps: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """bag_reps: (num_bags, hidden_dim)."""
            slide_rep, weights = self.attn(bag_reps)
            return slide_rep, weights  # (1, hidden_dim), (num_bags,)

    class DTFDMIL(nn.Module):
        """Full DTFD-MIL model.

        Args:
            input_dim:  Tile embedding dimension (from foundation model).
            hidden_dim: Internal representation size.
            num_classes: Number of output classes.
            num_pseudo_bags: How many pseudo-bags to sample per slide.
            pseudo_sampling_prob: Bernoulli sampling probability per tile.
            distillation_weight: Loss weight for the distillation term.
            dropout: Dropout probability.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            num_pseudo_bags: int = 4,
            pseudo_sampling_prob: float = 0.5,
            distillation_weight: float = 0.5,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.num_pseudo_bags = num_pseudo_bags
            self.pseudo_sampling_prob = pseudo_sampling_prob
            self.distillation_weight = distillation_weight

            self.tier1 = DTFDMILTier1(input_dim, hidden_dim, dropout)
            self.tier2 = DTFDMILTier2(hidden_dim, dropout)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(
            self,
            h: "torch.Tensor",
            training: bool = False,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
            """
            Args:
                h: (N, input_dim) tile embeddings.
                training: Whether to compute the distillation loss.

            Returns:
                logits: (1, num_classes)
                attention: (N,)  — averaged tile attention across pseudo-bags
                slide_embedding: (1, hidden_dim)
                distillation_loss: scalar tensor | None
            """
            if isinstance(h, np.ndarray):
                dev = next(self.parameters()).device
                h = torch.from_numpy(h).float().to(dev)

            device = h.device
            N = h.shape[0]

            # Tier-1: pseudo-bag aggregation
            if training and N > 1:
                gen = torch.Generator(device=device)
                pseudo_bags = _make_pseudo_bags(
                    h, self.num_pseudo_bags, self.pseudo_sampling_prob, rng=gen
                )
            else:
                # At inference: treat the full bag as a single pseudo-bag
                pseudo_bags = [h]

            tier1_reps = []
            tile_attn_acc = torch.zeros(N, device=device)

            for pb in pseudo_bags:
                rep, w = self.tier1(pb)  # (1, hidden_dim), (n_pb,)
                tier1_reps.append(rep.squeeze(0))  # (hidden_dim,)

            bag_reps = torch.stack(tier1_reps, dim=0)  # (num_bags, hidden_dim)

            # Tier-2: slide aggregation
            slide_rep, tier2_attn = self.tier2(bag_reps)  # (1, hidden_dim), (num_bags,)

            # Compute pseudo-bag attention (per tile, for heatmap)
            # We accumulate attention for tiles that appeared in each pseudo-bag
            if training and N > 1:
                with torch.no_grad():
                    gen2 = torch.Generator(device=device)
                    mask = torch.bernoulli(
                        torch.full((len(pseudo_bags), N), self.pseudo_sampling_prob), generator=gen2
                    ).bool()
                    for b_idx in range(len(pseudo_bags)):
                        indices = mask[b_idx].nonzero(as_tuple=True)[0]
                        if len(indices) == 0:
                            indices = torch.randint(N, (1,))
                        pb_rep, w_pb = self.tier1(h[indices])
                        tile_attn_acc[indices] += tier2_attn[b_idx] * w_pb
                    denom = tile_attn_acc.sum() + 1e-8
                    tile_attn_acc = tile_attn_acc / denom
            else:
                _, w_full = self.tier1(h)
                tile_attn_acc = w_full

            # Classification
            logits = self.classifier(slide_rep)  # (1, num_classes)

            # Optional distillation loss: pseudo-bag reps should be close to
            # the slide representation (cosine-similarity based).
            dist_loss: Optional["torch.Tensor"] = None
            if training:
                slide_norm = F.normalize(slide_rep.detach(), dim=-1)
                bag_norm = F.normalize(bag_reps, dim=-1)
                # Cosine distance: 1 - cos_sim
                dist_loss = (1 - (bag_norm * slide_norm).sum(dim=-1)).mean()

            return logits, tile_attn_acc, slide_rep, dist_loss

else:
    class DTFDMIL:  # type: ignore
        """NumPy fallback for DTFD-MIL (no PyTorch)."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            num_pseudo_bags: int = 4,
            pseudo_sampling_prob: float = 0.5,
            distillation_weight: float = 0.5,
            dropout: float = 0.25,
        ):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes

        def __call__(self, h: np.ndarray, training: bool = False):
            N = h.shape[0]
            attn = np.ones(N, dtype=np.float32) / max(1, N)
            mean_h = np.mean(h, axis=0)
            if len(mean_h) >= self.hidden_dim:
                slide_emb = mean_h[:self.hidden_dim].reshape(1, self.hidden_dim).astype(np.float32)
            else:
                slide_emb = np.pad(mean_h, (0, self.hidden_dim - len(mean_h))).reshape(1, self.hidden_dim).astype(np.float32)
            logits = np.zeros((1, self.num_classes), dtype=np.float32)
            return logits, attn, slide_emb, None

