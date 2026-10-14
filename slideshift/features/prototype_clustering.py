"""Prototype-based tile clustering for bag-size reduction.

Large slides can have 10,000+ tiles, making attention-based MIL memory-intensive.
This module clusters tile embeddings into K representative prototypes per slide
(similar in spirit to DSMIL and Patch-GCN prototype selection), reducing bag
size from ~10K to ~256 without discarding discriminative information.

References:
  - Li et al., "Dual-stream Multiple Instance Learning Network for Whole
    Slide Image Classification", CVPR 2021.
  - Shao et al., "Transmil: Transformer based Correlated MIL for WSI", NeurIPS 2021.
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans


def cluster_bag_to_prototypes(
    features: np.ndarray,
    coordinates: np.ndarray,
    num_prototypes: int = 256,
    use_mini_batch: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster tile features into K prototype representatives.

    Args:
        features:       (N, D) tile feature matrix.
        coordinates:    (N, 2) tile (row, col) coordinates.
        num_prototypes: Target number of prototype clusters (K).
        use_mini_batch: Use MiniBatchKMeans for speed (recommended for N > 5000).
        random_state:   RNG seed for reproducibility.

    Returns:
        proto_features: (K, D) cluster centroid features.
        proto_coords:   (K, 2) representative tile coordinates (nearest to centroid).
        proto_weights:  (K,)   normalised cluster sizes (useful as prior attention).
    """
    N = features.shape[0]
    if N <= num_prototypes:
        # Nothing to reduce; return as-is
        weights = np.ones(N, dtype=np.float32) / N
        return features.astype(np.float32), coordinates.astype(np.float32), weights

    k = min(num_prototypes, N)

    if use_mini_batch and N > 2000:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=min(1024, N),
            n_init=3,
        )
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")

    labels = km.fit_predict(features)              # (N,)
    centroids = km.cluster_centers_.astype(np.float32)  # (K, D)

    # Representative coordinate: tile closest to its centroid
    proto_coords = np.zeros((k, 2), dtype=np.float32)
    proto_weights = np.zeros(k, dtype=np.float32)

    for c in range(k):
        mask = labels == c
        cluster_size = int(np.sum(mask))
        proto_weights[c] = cluster_size

        if cluster_size == 0:
            continue

        cluster_feats = features[mask]
        cluster_coords = coordinates[mask]
        diffs = cluster_feats - centroids[c]
        dists = np.sum(diffs ** 2, axis=1)
        nearest = int(np.argmin(dists))
        proto_coords[c] = cluster_coords[nearest]

    # Normalise weights
    total = proto_weights.sum()
    if total > 0:
        proto_weights = proto_weights / total

    return centroids, proto_coords, proto_weights


def adaptive_bag_reduction(
    features: np.ndarray,
    coordinates: np.ndarray,
    max_tiles: int = 4096,
    num_prototypes: int = 256,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive two-stage reduction:
    
    1. If N > max_tiles, subsample max_tiles tiles uniformly.
    2. Then cluster to num_prototypes.

    This keeps memory use bounded even for very large slides (100k+ tiles).
    """
    N = features.shape[0]

    if N > max_tiles:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(N, size=max_tiles, replace=False)
        features = features[idx]
        coordinates = coordinates[idx]

    return cluster_bag_to_prototypes(
        features, coordinates, num_prototypes, random_state=random_state
    )

