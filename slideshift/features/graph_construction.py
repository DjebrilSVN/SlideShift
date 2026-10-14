"""Spatial k-NN graph construction for tile-level GNN aggregation.

Builds connectivity graphs from tile (row, col) coordinates using multiple
proximity strategies. Used by PatchGCN and future graph-based aggregators.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class TileGraph:
    """Adjacency representation of a slide's tile spatial graph."""
    num_nodes: int
    edge_index: np.ndarray          # (2, E) — source, dest pairs
    edge_weight: np.ndarray         # (E,)   — normalised weights
    adj_dense: Optional[np.ndarray] # (N, N) optional dense adjacency


def build_knn_graph(
    coordinates: np.ndarray,
    k: int = 8,
    self_loops: bool = True,
    normalise: bool = True,
) -> TileGraph:
    """Build a k-nearest-neighbour spatial graph.

    Args:
        coordinates: (N, 2) tile (row, col) centres in pixel or tile units.
        k:           Number of neighbours per tile.
        self_loops:  Add self-loops (recommended for GCN).
        normalise:   Apply symmetric degree normalisation D^{-1/2} A D^{-1/2}.

    Returns:
        TileGraph with edge_index and edge_weight for sparse message-passing.
    """
    N = len(coordinates)
    if N == 0:
        return TileGraph(0, np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32), None)

    k = min(k, N - 1)
    coords_f = coordinates.astype(np.float64)

    # Pairwise squared L2
    sq_dists = (
        np.sum(coords_f ** 2, axis=1, keepdims=True)
        - 2.0 * coords_f @ coords_f.T
        + np.sum(coords_f ** 2, axis=1)
    )
    sq_dists = np.clip(sq_dists, 0.0, None)

    adj = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        # Exclude self (index 0 when sorted)
        nn_sorted = np.argsort(sq_dists[i])
        nn_idx = nn_sorted[1 : k + 1]
        adj[i, nn_idx] = 1.0

    # Make symmetric
    adj = np.maximum(adj, adj.T)

    if self_loops:
        np.fill_diagonal(adj, 1.0)

    if normalise:
        deg = adj.sum(axis=1)
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        adj = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]

    # Convert to COO edge list
    src, dst = np.nonzero(adj)
    edge_weight = adj[src, dst].astype(np.float32)
    edge_index = np.stack([src.astype(np.int64), dst.astype(np.int64)], axis=0)

    return TileGraph(N, edge_index, edge_weight, adj.astype(np.float32))


def build_radius_graph(
    coordinates: np.ndarray,
    radius: float = 2.0,
    self_loops: bool = True,
    normalise: bool = True,
) -> TileGraph:
    """Connect tiles within a given grid-unit radius.

    For tiles extracted at stride S, a radius of 1.5 connects all 8-neighbours;
    radius ≈ 2.8 adds second-ring neighbours.
    """
    N = len(coordinates)
    if N == 0:
        return TileGraph(0, np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32), None)

    coords_f = coordinates.astype(np.float64)
    sq_dists = (
        np.sum(coords_f ** 2, axis=1, keepdims=True)
        - 2.0 * coords_f @ coords_f.T
        + np.sum(coords_f ** 2, axis=1)
    )
    sq_dists = np.clip(sq_dists, 0.0, None)

    adj = (sq_dists <= radius ** 2).astype(np.float32)
    if not self_loops:
        np.fill_diagonal(adj, 0.0)

    if normalise:
        deg = adj.sum(axis=1)
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        adj = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]

    src, dst = np.nonzero(adj)
    edge_weight = adj[src, dst].astype(np.float32)
    edge_index = np.stack([src.astype(np.int64), dst.astype(np.int64)], axis=0)

    return TileGraph(N, edge_index, edge_weight, adj.astype(np.float32))


def compute_graph_statistics(graph: TileGraph) -> dict:
    """Compute basic graph statistics for quality checks."""
    if graph.num_nodes == 0:
        return {}
    degrees = np.bincount(graph.edge_index[0], minlength=graph.num_nodes)
    return {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.edge_index.shape[1],
        "mean_degree": float(np.mean(degrees)),
        "max_degree": int(np.max(degrees)),
        "min_degree": int(np.min(degrees)),
        "density": float(graph.edge_index.shape[1]) / max(1, graph.num_nodes ** 2),
    }

