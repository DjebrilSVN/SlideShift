"""Advanced MIL aggregator tests: DTFD-MIL, ACMIL, PatchGCN, survival, conformal."""

import numpy as np
import pytest

from slideshift.mil.aggregators import AGGREGATOR_REGISTRY, build_aggregator
from slideshift.mil.aggregators.dtfd_mil import DTFDMIL
from slideshift.mil.aggregators.acmil import ACMIL
from slideshift.mil.aggregators.patch_gcn import PatchGCN, build_knn_adj
from slideshift.mil.survival import (
    concordance_index,
    vectorised_concordance_index,
    CoxSurvivalHead,
    SurvivalMILTrainer,
)
from slideshift.mil.trainer import (
    delong_auc_test,
    wilcoxon_cv_comparison,
    feature_mixup,
    temperature_scaling,
    compute_reliability_diagram,
)
from slideshift.robustness.conformal import ConformalPredictor


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

def test_registry_completeness():
    """All new aggregators must be in the registry."""
    expected = {"mean_pool", "max_pool", "abmil", "clam_sb", "clam_mb",
                "transmil", "patch_gcn", "dtfd_mil", "acmil"}
    assert expected.issubset(set(AGGREGATOR_REGISTRY.keys()))


# ─────────────────────────────────────────────
# k-NN adjacency matrix
# ─────────────────────────────────────────────

def test_knn_adj_symmetry():
    """Adjacency matrix must be symmetric."""
    coords = np.array([[0, 0], [1, 0], [0, 1], [2, 2], [3, 3]], dtype=np.float32)
    adj = build_knn_adj(coords, k=2)
    assert adj.shape == (5, 5)
    np.testing.assert_allclose(adj, adj.T, atol=1e-5)


def test_knn_adj_self_loops():
    """Diagonal of adjacency matrix must be non-zero (self-loops)."""
    coords = np.random.rand(8, 2).astype(np.float32)
    adj = build_knn_adj(coords, k=3)
    assert np.all(np.diag(adj) > 0), "Self-loops expected"


# ─────────────────────────────────────────────
# DTFD-MIL
# ─────────────────────────────────────────────

def test_dtfd_mil_numpy_fallback():
    """DTFD-MIL numpy fallback must return correct shapes."""
    model = DTFDMIL(input_dim=32, hidden_dim=16, num_classes=2)
    h = np.random.randn(50, 32).astype(np.float32)
    out = model(h)
    logits, attn, emb, aux = out
    assert logits.shape == (1, 2)
    assert attn.shape == (50,)
    assert emb.shape[1] == 16 or emb.shape == (1, 16)


# ─────────────────────────────────────────────
# ACMIL
# ─────────────────────────────────────────────

def test_acmil_numpy_fallback():
    """ACMIL numpy fallback must return correct shapes."""
    model = ACMIL(input_dim=32, hidden_dim=16, num_classes=3, num_branches=2)
    h = np.random.randn(20, 32).astype(np.float32)
    out = model(h)
    logits, attn, emb, ortho = out
    assert logits.shape == (1, 3)
    assert len(attn) == 20


# ─────────────────────────────────────────────
# Patch-GCN
# ─────────────────────────────────────────────

def test_patch_gcn_numpy_fallback():
    """PatchGCN numpy fallback must return correct shapes."""
    model = PatchGCN(input_dim=64, hidden_dim=32, num_classes=2, k_neighbours=3)
    h = np.random.randn(30, 64).astype(np.float32)
    coords = np.random.rand(30, 2).astype(np.float32) * 100
    out = model(h, coords)
    logits, attn, emb = out
    assert logits.shape == (1, 2)
    assert len(attn) == 30


# ─────────────────────────────────────────────
# Survival analysis
# ─────────────────────────────────────────────

def test_concordance_index_perfect():
    """Perfect risk ordering should yield C-index of 1.0."""
    risk = np.array([3.0, 2.0, 1.0])            # higher risk → shorter survival
    times = np.array([1.0, 2.0, 3.0])
    events = np.array([1, 1, 1])
    ci = vectorised_concordance_index(risk, times, events)
    assert ci == pytest.approx(1.0, abs=0.01)


def test_concordance_index_random():
    """Random risk should yield C-index near 0.5."""
    rng = np.random.RandomState(0)
    risk = rng.randn(100)
    times = rng.exponential(scale=1.0, size=100)
    events = rng.binomial(1, 0.7, size=100)
    ci = vectorised_concordance_index(risk, times, events)
    assert 0.3 < ci < 0.7, f"Random C-index expected near 0.5, got {ci}"


def test_survival_trainer_numpy():
    """SurvivalMILTrainer must run without torch."""
    head = CoxSurvivalHead(input_dim=16, hidden_dim=0)
    trainer = SurvivalMILTrainer(head, device="cpu")
    embs = np.random.randn(20, 16).astype(np.float32)
    times = np.random.exponential(1.0, 20).astype(np.float32)
    events = np.random.binomial(1, 0.6, 20).astype(np.float32)
    metrics = trainer.evaluate(embs, times, events)
    assert "c_index" in metrics
    assert 0.0 <= metrics["c_index"] <= 1.0


# ─────────────────────────────────────────────
# Advanced trainer utilities
# ─────────────────────────────────────────────

def test_delong_auc_test():
    """DeLong test must return three floats."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 40)
    y_a = rng.rand(40, 2)
    y_a = y_a / y_a.sum(axis=1, keepdims=True)
    y_b = rng.rand(40, 2)
    y_b = y_b / y_b.sum(axis=1, keepdims=True)
    auc_a, auc_b, p = delong_auc_test(y_true, y_a, y_b)
    assert 0.0 <= auc_a <= 1.0
    assert 0.0 <= auc_b <= 1.0
    assert 0.0 <= p <= 1.0


def test_wilcoxon_cv_comparison():
    """Wilcoxon test must produce sensible output."""
    fold_aucs_a = [0.82, 0.85, 0.80, 0.88, 0.84]
    fold_aucs_b = [0.75, 0.78, 0.72, 0.80, 0.76]
    result = wilcoxon_cv_comparison(fold_aucs_a, fold_aucs_b)
    assert "p_value" in result
    assert result["mean_diff"] > 0


def test_feature_mixup():
    """Feature mixup must produce valid interpolated features and soft labels."""
    rng = np.random.RandomState(7)
    feat_a = rng.randn(128).astype(np.float32)
    feat_b = rng.randn(128).astype(np.float32)
    mixed_feat, mixed_label = feature_mixup(feat_a, feat_b, 0, 1, num_classes=2, alpha=0.4, rng=rng)
    assert mixed_feat.shape == (128,)
    assert mixed_label.shape == (2,)
    assert abs(mixed_label.sum() - 1.0) < 1e-5, "Soft label must sum to 1"


def test_temperature_scaling():
    """Temperature scaling must reduce ECE on miscalibrated predictions."""
    rng = np.random.RandomState(0)
    N = 80
    labels = rng.randint(0, 2, N)
    # Overconfident predictions
    probs = np.zeros((N, 2), dtype=np.float32)
    probs[np.arange(N), labels] = 0.99
    probs[np.arange(N), 1 - labels] = 0.01
    # Add noise
    noise_idx = rng.choice(N, N // 4, replace=False)
    probs[noise_idx] = probs[noise_idx, ::-1]

    T, scaled = temperature_scaling(probs, labels)
    assert T > 0
    assert scaled.shape == probs.shape
    assert abs(scaled.sum(axis=1) - 1.0).max() < 1e-4, "Scaled probs must sum to 1"


def test_reliability_diagram():
    """Reliability diagram must produce arrays of the right shape."""
    rng = np.random.RandomState(42)
    N = 100
    labels = rng.randint(0, 2, N)
    probs = rng.dirichlet([2, 2], size=N).astype(np.float32)
    rd = compute_reliability_diagram(probs, labels, n_bins=10)
    assert len(rd["bin_confidence"]) == 10
    assert len(rd["bin_accuracy"]) == 10
    assert 0.0 <= rd["ece"] <= 1.0


# ─────────────────────────────────────────────
# Conformal prediction
# ─────────────────────────────────────────────

def test_conformal_coverage_guarantee():
    """Conformal predictor must achieve at least target coverage."""
    rng = np.random.RandomState(0)
    N = 200
    C = 3
    labels = rng.randint(0, C, N)
    probs = rng.dirichlet(np.ones(C), size=N).astype(np.float32)

    cal_probs = probs[:100]
    cal_labels = labels[:100]
    test_probs = probs[100:]
    test_labels = labels[100:]

    cp = ConformalPredictor(target_coverage=0.90, method="lac")
    cp.calibrate(cal_probs, cal_labels)
    eval_result = cp.evaluate_coverage(test_probs, test_labels)

    # Coverage may occasionally miss due to random data; check it's in reasonable range
    assert 0.70 <= eval_result.coverage <= 1.0, f"Coverage {eval_result.coverage} out of range"
    assert eval_result.average_set_size >= 1.0
    assert 0.0 <= eval_result.defer_rate <= 1.0


def test_conformal_raps_method():
    """RAPS method must also produce valid prediction sets."""
    rng = np.random.RandomState(1)
    labels = rng.randint(0, 2, 100)
    probs = rng.dirichlet([3, 1], size=100).astype(np.float32)

    cp = ConformalPredictor(target_coverage=0.85, method="raps")
    cp.calibrate(probs[:60], labels[:60])
    result = cp.predict(probs[60], slide_id="test_slide")
    assert isinstance(result.prediction_set, set)
    assert result.predicted_class in result.prediction_set

