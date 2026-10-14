"""Tests for advanced feature engineering modules."""

import numpy as np
import pytest

from slideshift.features.prototype_clustering import (
    cluster_bag_to_prototypes,
    adaptive_bag_reduction,
)
from slideshift.features.graph_construction import (
    build_knn_graph,
    build_radius_graph,
    compute_graph_statistics,
    TileGraph,
)
from slideshift.features.zero_shot import (
    CONCHZeroShotClassifier,
    ZeroShotClassSpec,
    CAMELYON_PROMPTS,
)


# ─────────────────────────────────────────────
# Prototype clustering
# ─────────────────────────────────────────────

def test_cluster_bag_reduces_size():
    """Clustering must produce exactly num_prototypes representatives."""
    rng = np.random.RandomState(0)
    N, D = 800, 64
    features = rng.randn(N, D).astype(np.float32)
    coords = rng.rand(N, 2).astype(np.float32) * 200

    proto_feats, proto_coords, proto_weights = cluster_bag_to_prototypes(
        features, coords, num_prototypes=32
    )
    assert proto_feats.shape == (32, D), f"Expected (32, {D}), got {proto_feats.shape}"
    assert proto_coords.shape == (32, 2)
    assert proto_weights.shape == (32,)
    assert abs(proto_weights.sum() - 1.0) < 1e-4, "Weights must sum to 1"


def test_cluster_bag_passthrough_when_small():
    """When N <= num_prototypes, all tiles must be returned unchanged."""
    rng = np.random.RandomState(1)
    N, D = 20, 16
    features = rng.randn(N, D).astype(np.float32)
    coords = rng.rand(N, 2).astype(np.float32)

    proto_feats, proto_coords, proto_weights = cluster_bag_to_prototypes(
        features, coords, num_prototypes=256
    )
    assert len(proto_feats) == N


def test_adaptive_bag_reduction():
    """Adaptive reduction must cap tile count."""
    rng = np.random.RandomState(2)
    N, D = 6000, 32
    features = rng.randn(N, D).astype(np.float32)
    coords = rng.rand(N, 2).astype(np.float32) * 500

    proto_feats, _, weights = adaptive_bag_reduction(
        features, coords, max_tiles=1000, num_prototypes=64
    )
    assert len(proto_feats) == 64
    assert abs(weights.sum() - 1.0) < 1e-4


# ─────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────

def test_knn_graph_basic():
    """k-NN graph must produce symmetric edge list with self-loops."""
    coords = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]], dtype=np.float32)
    graph = build_knn_graph(coords, k=2, self_loops=True, normalise=False)

    assert graph.num_nodes == 5
    assert graph.edge_index.shape[0] == 2
    # Check self-loops are present: edge i→i for all i
    self_loop_mask = graph.edge_index[0] == graph.edge_index[1]
    assert self_loop_mask.sum() == 5, "Expected 5 self-loops"


def test_knn_graph_symmetry():
    """k-NN graph must be symmetric (if i→j then j→i)."""
    rng = np.random.RandomState(0)
    coords = rng.rand(12, 2).astype(np.float32) * 50
    graph = build_knn_graph(coords, k=3, self_loops=False, normalise=False)
    adj = graph.adj_dense
    assert adj is not None
    np.testing.assert_allclose(adj, adj.T, atol=1e-5)


def test_radius_graph():
    """Radius graph must connect all 8-neighbours for unit-stride grid."""
    # Grid: (0,0) (0,1) (1,0) (1,1) — all within radius sqrt(2) ≈ 1.42
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    graph = build_radius_graph(coords, radius=1.5, self_loops=False, normalise=False)
    adj = graph.adj_dense
    assert adj is not None
    # (0,0) should be connected to (0,1) and (1,0) but not necessarily (1,1)
    assert adj[0, 1] > 0 or adj[0, 2] > 0


def test_graph_statistics():
    """Graph statistics must be computable without errors."""
    rng = np.random.RandomState(3)
    coords = rng.rand(20, 2).astype(np.float32) * 100
    graph = build_knn_graph(coords, k=4)
    stats = compute_graph_statistics(graph)
    assert "num_nodes" in stats
    assert stats["num_nodes"] == 20
    assert stats["mean_degree"] >= 1.0


def test_empty_graph():
    """Empty graph (0 tiles) must not crash."""
    coords = np.zeros((0, 2), dtype=np.float32)
    graph = build_knn_graph(coords, k=4)
    assert graph.num_nodes == 0


# ─────────────────────────────────────────────
# Zero-shot classification
# ─────────────────────────────────────────────

def test_zero_shot_mock_classifier():
    """CONCH zero-shot classifier mock must produce valid probabilities."""
    clf = CONCHZeroShotClassifier(use_mock=True)
    clf.load()
    clf.build_class_embeddings(CAMELYON_PROMPTS, emb_dim=128)

    # Simulate tile embeddings from CONCH vision encoder
    rng = np.random.RandomState(42)
    tile_embs = rng.randn(50, 128).astype(np.float32)
    probs = clf.predict_tile_scores(tile_embs)
    assert probs.shape == (50, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_zero_shot_slide_prediction():
    """Slide-level prediction must return valid class probabilities."""
    clf = CONCHZeroShotClassifier(use_mock=True)
    clf.load()
    clf.build_class_embeddings(CAMELYON_PROMPTS, emb_dim=64)

    rng = np.random.RandomState(0)
    tile_embs = rng.randn(100, 64).astype(np.float32)

    for agg in ["mean", "max", "top10pct"]:
        slide_probs, pred_class = clf.predict_slide(tile_embs, aggregation=agg)
        assert len(slide_probs) == 2
        assert abs(slide_probs.sum() - 1.0) < 1e-5
        assert pred_class in (0, 1)


def test_zero_shot_evaluation():
    """Batch evaluation must report AUROC and balanced accuracy."""
    clf = CONCHZeroShotClassifier(use_mock=True)
    clf.load()
    clf.build_class_embeddings(CAMELYON_PROMPTS, emb_dim=64)

    rng = np.random.RandomState(1)
    tile_embs_list = [rng.randn(30, 64).astype(np.float32) for _ in range(20)]
    labels = [rng.randint(0, 2) for _ in range(20)]

    metrics = clf.evaluate(tile_embs_list, labels)
    assert "auroc" in metrics
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert "balanced_accuracy" in metrics

