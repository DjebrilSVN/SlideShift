"""ACMIL: Adaptive Cross-staining Multiple Instance Learning.

Inspired by:
    Yang et al., "Revisiting Multiple Instance Learning for Whole Slide Image
    Classification", NeurIPS 2023 (Adaptation-Centered MIL).

Key ideas:
  - Multiple attention branches (heads) that each focus on different semantic
    aspects of the slide.
  - Adaptive aggregation: a context-dependent gate decides how much each branch
    contributes to the final prediction.
  - Orthogonality regularisation encourages branches to be diverse.
"""

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
    class _MultiHeadAttentionBranch(nn.Module):
        """One attention branch: project → gated attention → aggregate."""

        def __init__(self, in_dim: int, hidden: int, dropout: float = 0.25):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.attn_v = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh())
            self.attn_u = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Sigmoid())
            self.attn_w = nn.Linear(hidden // 2, 1)

        def forward(
            self, h: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Returns:
                proj: (N, hidden)   — projected tile embeddings for this branch
                attn: (N,)          — attention weights
                agg:  (1, hidden)   — aggregated representation
            """
            proj = self.proj(h)                             # (N, hidden)
            raw = self.attn_w(self.attn_v(proj) * self.attn_u(proj)).squeeze(-1)  # (N,)
            attn = F.softmax(raw, dim=0)                    # (N,)
            agg = torch.sum(attn.unsqueeze(-1) * proj, dim=0, keepdim=True)       # (1, hidden)
            return proj, attn, agg

    class ACMIL(nn.Module):
        """Adaptive Cross-MIL with multi-branch attention and diversity regularisation.

        Args:
            input_dim:    Tile embedding dimension.
            hidden_dim:   Internal dimension for each branch.
            num_classes:  Number of output classes.
            num_branches: Number of parallel attention branches.
            dropout:      Dropout probability.
            ortho_weight: Coefficient for the orthogonality regularisation loss.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            num_branches: int = 4,
            dropout: float = 0.25,
            ortho_weight: float = 0.1,
        ):
            super().__init__()
            self.num_branches = num_branches
            self.hidden_dim = hidden_dim
            self.ortho_weight = ortho_weight

            self.branches = nn.ModuleList(
                [_MultiHeadAttentionBranch(input_dim, hidden_dim, dropout) for _ in range(num_branches)]
            )

            # Context-dependent adaptive gate: takes the mean-pooled tile embedding
            # and outputs a weight for each branch.
            self.gate = nn.Sequential(
                nn.Linear(input_dim, num_branches),
                nn.Softmax(dim=-1),
            )

            # Final classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(
            self, h: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
            """
            Args:
                h: (N, input_dim)

            Returns:
                logits:          (1, num_classes)
                attention:       (N,)  — gate-weighted average of branch attentions
                slide_embedding: (1, hidden_dim)
                ortho_loss:      scalar tensor — orthogonality regularisation
            """
            if isinstance(h, np.ndarray):
                dev = next(self.parameters()).device
                h = torch.from_numpy(h).float().to(dev)

            # Adaptive gate from context (mean-pooled bag)
            context = h.mean(dim=0, keepdim=True)           # (1, input_dim)
            gate_weights = self.gate(context).squeeze(0)    # (num_branches,)

            branch_attns = []
            branch_aggs = []

            for i, branch in enumerate(self.branches):
                _, attn, agg = branch(h)                    # (N,), (1, hidden_dim)
                branch_attns.append(attn)
                branch_aggs.append(agg.squeeze(0))          # (hidden_dim,)

            # Stack: (num_branches, N) and (num_branches, hidden_dim)
            attn_stack = torch.stack(branch_attns, dim=0)   # (B, N)
            agg_stack = torch.stack(branch_aggs, dim=0)     # (B, hidden_dim)

            # Aggregate: gate_weights (B,) × branch attentions
            tile_attn = (gate_weights.unsqueeze(-1) * attn_stack).sum(dim=0)  # (N,)
            slide_emb = (gate_weights.unsqueeze(-1) * agg_stack).sum(dim=0, keepdim=True)  # (1, hidden_dim)

            logits = self.classifier(slide_emb)             # (1, num_classes)

            # Orthogonality loss: branch projections should be diverse.
            # We use the Gram matrix of (normalised) branch aggregation vectors.
            branch_norms = F.normalize(agg_stack, dim=-1)   # (B, hidden_dim)
            gram = torch.mm(branch_norms, branch_norms.t()) # (B, B)
            eye = torch.eye(self.num_branches, device=h.device)
            ortho_loss = ((gram - eye) ** 2).sum() * self.ortho_weight

            return logits, tile_attn, slide_emb, ortho_loss

else:
    class ACMIL:  # type: ignore
        """NumPy fallback for ACMIL (no PyTorch)."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            num_branches: int = 4,
            dropout: float = 0.25,
            ortho_weight: float = 0.1,
        ):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes

        def __call__(self, h: np.ndarray):
            N = h.shape[0]
            attn = np.ones(N, dtype=np.float32) / max(1, N)
            mean_h = np.mean(h, axis=0)
            if len(mean_h) >= self.hidden_dim:
                slide_emb = mean_h[:self.hidden_dim].reshape(1, self.hidden_dim).astype(np.float32)
            else:
                slide_emb = np.pad(mean_h, (0, self.hidden_dim - len(mean_h))).reshape(1, self.hidden_dim).astype(np.float32)
            logits = np.zeros((1, self.num_classes), dtype=np.float32)
            return logits, attn, slide_emb, None

