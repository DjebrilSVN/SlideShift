"""Patch-GCN: Graph Convolutional Network over spatial tile graphs.

Reference:
    Chen et al., "Whole Slide Images are 2D Point Clouds: Context-Aware
    Survival Prediction using Patch-based Graph Convolutional Networks",
    MICCAI 2021 / NeurIPS 2021.

Architecture:
  1. Build a k-NN spatial graph from tile (x, y) coordinates.
  2. Run multi-layer GraphSAGE-style message passing.
  3. Global pooling (sum + mean) → slide representation → classifier.
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


def build_knn_adj(
    coords: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """Build a k-NN adjacency matrix from tile (row, col) coordinates.

    Returns:
        adj: (N, N) float32 array, symmetric, self-loops included.
    """
    N = len(coords)
    if N <= 1:
        return np.eye(N, dtype=np.float32)

    k = min(k, N - 1)
    if hasattr(coords, "cpu"):
        coords = coords.cpu().numpy()
    coords_f = np.asarray(coords, dtype=np.float32)
    # Pairwise squared Euclidean distances
    sq_dists = (
        np.sum(coords_f ** 2, axis=1, keepdims=True)
        - 2 * coords_f @ coords_f.T
        + np.sum(coords_f ** 2, axis=1)
    )
    sq_dists = np.clip(sq_dists, 0.0, None)

    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        nn_idx = np.argsort(sq_dists[i])[1 : k + 1]  # exclude self
        adj[i, nn_idx] = 1.0
        adj[nn_idx, i] = 1.0
    # Self-loops
    np.fill_diagonal(adj, 1.0)

    # Symmetric normalisation: D^{-1/2} A D^{-1/2}
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]

    return adj_norm.astype(np.float32)


if HAS_TORCH:
    class _GraphConvLayer(nn.Module):
        """One GraphSAGE-style GCN layer: h' = ReLU(A_norm h W)."""

        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.25):
            super().__init__()
            self.lin = nn.Linear(in_dim, out_dim)
            self.norm = nn.LayerNorm(out_dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, h: "torch.Tensor", adj: "torch.Tensor") -> "torch.Tensor":
            agg = torch.mm(adj, h)             # (N, in_dim)
            out = self.lin(agg)                # (N, out_dim)
            out = self.norm(out)
            out = F.gelu(out)
            out = self.drop(out)
            return out

    class PatchGCN(nn.Module):
        """Spatial GCN aggregator for WSI tiles.

        Args:
            input_dim:   Tile embedding dimension.
            hidden_dim:  GCN hidden dimension.
            num_classes: Number of output classes.
            num_layers:  Number of GCN message-passing layers.
            k_neighbours: k for k-NN graph construction.
            dropout:     Dropout probability.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            num_layers: int = 4,
            k_neighbours: int = 8,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.k = k_neighbours
            self.num_layers = num_layers

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # GCN layers with residual connections
            self.gcn_layers = nn.ModuleList(
                [_GraphConvLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
            )

            # Hierarchical pooling: concat [mean, sum, max] → richer representation
            self.pool_proj = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Slide-level attention for interpretability
            self.attn_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
            self.attn_u = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Sigmoid())
            self.attn_w = nn.Linear(hidden_dim // 2, 1)

            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def _build_adj(self, coords: np.ndarray, device: "torch.device") -> "torch.Tensor":
            adj_np = build_knn_adj(coords, k=self.k)
            return torch.from_numpy(adj_np).to(device)

        def forward(
            self,
            h: "torch.Tensor",
            coords: Optional[np.ndarray] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Args:
                h:      (N, input_dim) tile embeddings.
                coords: (N, 2) tile spatial coordinates. If None, uses positional indices.

            Returns:
                logits:          (1, num_classes)
                attention:       (N,)
                slide_embedding: (1, hidden_dim)
            """
            if isinstance(h, np.ndarray):
                dev = next(self.parameters()).device
                h = torch.from_numpy(h).float().to(dev)

            N = h.shape[0]
            device = h.device

            # Build adjacency matrix
            if coords is None:
                # Fallback: 1-D positional coords
                coords = np.column_stack([np.arange(N), np.zeros(N)])
            adj = self._build_adj(coords, device)            # (N, N)

            # Input projection
            x = self.input_proj(h)                           # (N, hidden_dim)

            # GCN message passing with residual connections
            for layer in self.gcn_layers:
                x_new = layer(x, adj)
                x = x + x_new                               # residual

            # Gated attention (for heatmap)
            a_raw = self.attn_w(self.attn_v(x) * self.attn_u(x)).squeeze(-1)   # (N,)
            attn = F.softmax(a_raw, dim=0)                   # (N,)

            # Hierarchical pooling
            mean_pool = x.mean(dim=0)                        # (hidden_dim,)
            sum_pool = x.sum(dim=0)                          # (hidden_dim,)
            max_pool, _ = x.max(dim=0)                       # (hidden_dim,)
            pool_cat = torch.cat([mean_pool, sum_pool, max_pool], dim=-1)  # (3 * hidden_dim,)

            slide_emb = self.pool_proj(pool_cat).unsqueeze(0)  # (1, hidden_dim)
            logits = self.classifier(slide_emb)               # (1, num_classes)

            return logits, attn, slide_emb

else:
    class PatchGCN:  # type: ignore
        """NumPy fallback for Patch-GCN (no PyTorch)."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_classes: int = 2,
            num_layers: int = 4,
            k_neighbours: int = 8,
            dropout: float = 0.25,
        ):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.k = k_neighbours

        def __call__(self, h: np.ndarray, coords: Optional[np.ndarray] = None):
            N = h.shape[0]
            attn = np.ones(N, dtype=np.float32) / max(1, N)
            mean_h = np.mean(h, axis=0)
            if len(mean_h) >= self.hidden_dim:
                slide_emb = mean_h[:self.hidden_dim].reshape(1, self.hidden_dim).astype(np.float32)
            else:
                slide_emb = np.pad(mean_h, (0, self.hidden_dim - len(mean_h))).reshape(1, self.hidden_dim).astype(np.float32)
            logits = np.zeros((1, self.num_classes), dtype=np.float32)
            return logits, attn, slide_emb

