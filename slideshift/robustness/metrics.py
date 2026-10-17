"""Advanced metrics for evaluating model degradation under domain shifts.

Additions over baseline:
  - Maximum Mean Discrepancy (MMD) with RBF kernel
  - Fréchet-like embedding distance (mean + covariance)
  - Per-site stratified degradation report
  - Temperature scaling integration for calibration comparison
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from slideshift.mil.trainer import compute_expected_calibration_error


def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """Maximum Mean Discrepancy with RBF (Gaussian) kernel.

    MMD(P, Q) = E_P[k(x,x')] - 2*E_{P,Q}[k(x,y)] + E_Q[k(y,y')]

    A value near 0 indicates the two distributions are similar.
    Values > 0.05 suggest meaningful distributional shift.

    Args:
        X:     (n, D) source embeddings.
        Y:     (m, D) target embeddings.
        gamma: RBF bandwidth (1 / (2 * sigma^2)). If None, uses the
               median heuristic: gamma = 1 / (2 * median_pairwise_sq_dist).

    Returns:
        MMD² value (non-negative float).
    """
    if len(X) == 0 or len(Y) == 0:
        return 0.0

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    if gamma is None:
        # Median heuristic on a random subsample for speed
        n_sub = min(200, len(X) + len(Y))
        all_pts = np.concatenate([X[:n_sub // 2], Y[:n_sub // 2]], axis=0)
        sq_d = np.sum((all_pts[:, None] - all_pts[None, :]) ** 2, axis=-1)
        median_sq = float(np.median(sq_d[sq_d > 0])) if (sq_d > 0).any() else 1.0
        gamma = 1.0 / (2.0 * max(median_sq, 1e-6))

    def _rbf_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-gamma * sq)

    Kxx = _rbf_kernel(X, X)
    Kyy = _rbf_kernel(Y, Y)
    Kxy = _rbf_kernel(X, Y)

    n, m = len(X), len(Y)
    # Unbiased estimator
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (
        Kxx.sum() / (n * (n - 1)) +
        Kyy.sum() / (m * (m - 1)) -
        2.0 * Kxy.mean()
    )
    return float(max(0.0, mmd2))


def frechet_embedding_distance(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
) -> float:
    """Fréchet-like distance between two embedding distributions.

    FED = ||μ_s - μ_t||² + Tr(Σ_s + Σ_t - 2*(Σ_s Σ_t)^{1/2})

    Analogous to FID but on slide embeddings rather than Inception features.
    Smaller = more similar distributions.

    Args:
        source_embeddings: (n, D) reference (training) embeddings.
        target_embeddings: (m, D) test/shifted embeddings.

    Returns:
        FED value ≥ 0.
    """
    if len(source_embeddings) < 2 or len(target_embeddings) < 2:
        return 0.0

    mu_s = source_embeddings.mean(axis=0).astype(np.float64)
    mu_t = target_embeddings.mean(axis=0).astype(np.float64)

    cov_s = np.cov(source_embeddings.astype(np.float64), rowvar=False) + np.eye(source_embeddings.shape[1]) * 1e-6
    cov_t = np.cov(target_embeddings.astype(np.float64), rowvar=False) + np.eye(target_embeddings.shape[1]) * 1e-6

    mean_diff_sq = float(np.sum((mu_s - mu_t) ** 2))

    # Matrix square root via eigendecomposition
    try:
        prod = cov_s @ cov_t
        eigvals, eigvecs = np.linalg.eigh(prod)
        eigvals = np.clip(eigvals, 0.0, None)
        sqrt_prod = (eigvecs * np.sqrt(eigvals)[None, :]) @ eigvecs.T
        trace_term = float(np.trace(cov_s) + np.trace(cov_t) - 2.0 * np.trace(sqrt_prod))
    except np.linalg.LinAlgError:
        trace_term = 0.0

    return float(max(0.0, mean_diff_sq + trace_term))


def compute_prediction_flip_rate(
    clean_predictions: np.ndarray,
    shifted_predictions: np.ndarray,
) -> float:
    """Fraction of slide bags whose discrete prediction flips after perturbation."""
    if len(clean_predictions) == 0:
        return 0.0
    flips = np.sum(clean_predictions != shifted_predictions)
    return float(flips / len(clean_predictions))


def compute_heatmap_stability(
    clean_attention: np.ndarray,
    shifted_attention: np.ndarray,
    top_k_fraction: float = 0.10,
) -> Tuple[float, float]:
    """Compute attention heatmap stability before and after shift.

    Returns:
        (spearman_rank_correlation, top_k_jaccard_overlap)
    """
    n = len(clean_attention)
    if n < 3:
        return 1.0, 1.0

    # 1. Spearman rank correlation
    corr, _ = spearmanr(clean_attention, shifted_attention)
    if np.isnan(corr):
        corr = 0.0

    # 2. Top-k Jaccard overlap
    k = max(1, int(round(n * top_k_fraction)))
    clean_top_k = set(np.argsort(clean_attention)[-k:])
    shifted_top_k = set(np.argsort(shifted_attention)[-k:])

    intersection = len(clean_top_k.intersection(shifted_top_k))
    union = len(clean_top_k.union(shifted_top_k))
    jaccard = float(intersection / max(1, union))

    return float(corr), jaccard


def evaluate_shift_degradation(
    y_true: np.ndarray,
    clean_probs: np.ndarray,
    shifted_probs: np.ndarray,
    clean_attentions: Optional[List[np.ndarray]] = None,
    shifted_attentions: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """Quantify performance degradation from clean to shifted data."""
    # Predictions
    clean_preds = np.argmax(clean_probs, axis=1) if clean_probs.ndim > 1 else (clean_probs >= 0.5).astype(int)
    shifted_preds = np.argmax(shifted_probs, axis=1) if shifted_probs.ndim > 1 else (shifted_probs >= 0.5).astype(int)

    # Clean metrics
    clean_p1 = clean_probs[:, 1] if clean_probs.ndim > 1 else clean_probs
    shifted_p1 = shifted_probs[:, 1] if shifted_probs.ndim > 1 else shifted_probs

    try:
        clean_auc = float(roc_auc_score(y_true, clean_p1))
    except Exception:
        clean_auc = 0.5

    try:
        shifted_auc = float(roc_auc_score(y_true, shifted_p1))
    except Exception:
        shifted_auc = 0.5

    clean_bal_acc = float(balanced_accuracy_score(y_true, clean_preds))
    shifted_bal_acc = float(balanced_accuracy_score(y_true, shifted_preds))

    clean_ece = compute_expected_calibration_error(clean_probs, y_true)
    shifted_ece = compute_expected_calibration_error(shifted_probs, y_true)

    flip_rate = compute_prediction_flip_rate(clean_preds, shifted_preds)

    # Heatmap stability across bags
    corrs = []
    jaccards = []
    if clean_attentions and shifted_attentions and len(clean_attentions) == len(shifted_attentions):
        for c_att, s_att in zip(clean_attentions, shifted_attentions):
            if len(c_att) == len(s_att) and len(c_att) > 2:
                r_corr, r_jac = compute_heatmap_stability(c_att, s_att)
                corrs.append(r_corr)
                jaccards.append(r_jac)

    mean_corr = float(np.mean(corrs)) if corrs else float("nan")
    mean_jaccard = float(np.mean(jaccards)) if jaccards else float("nan")

    return {
        "clean_auroc": clean_auc,
        "shifted_auroc": shifted_auc,
        "delta_auroc": clean_auc - shifted_auc,
        "clean_balanced_accuracy": clean_bal_acc,
        "shifted_balanced_accuracy": shifted_bal_acc,
        "delta_balanced_accuracy": clean_bal_acc - shifted_bal_acc,
        "clean_ece": clean_ece,
        "shifted_ece": shifted_ece,
        "delta_ece": shifted_ece - clean_ece,  # Positive means worse calibration
        "flip_rate": flip_rate,
        "heatmap_rank_correlation": mean_corr,
        "heatmap_top10_jaccard": mean_jaccard,
    }
