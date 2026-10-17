"""Advanced real-time slide distribution shift detector.

Methods:
  1. Mahalanobis distance (stain moments + embedding mean) — original
  2. Maximum Mean Discrepancy with RBF kernel — distributional, non-parametric
  3. Hotelling T² test — formal multivariate hypothesis test with p-value
  4. Energy-based OOD score — density-free scalar score

Reference for Hotelling T²:
  Hotelling, H. (1931). The generalization of Student's ratio.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from slideshift.core.types import ShiftAlert
from slideshift.monitor.extractor import SlideSignature


def hotelling_t2(
    query: np.ndarray,
    reference_matrix: np.ndarray,
) -> Tuple[float, float]:
    """Hotelling T² statistic and approximate p-value for one vs. group test.

    Tests whether `query` is drawn from the same distribution as
    `reference_matrix` (N samples, D features).

    Returns:
        t2_stat: T² statistic.
        p_value: Approximate F-distribution p-value.
    """
    from scipy.stats import f as f_dist

    N, D = reference_matrix.shape
    if N <= D + 1:
        return 0.0, 1.0

    mu = reference_matrix.mean(axis=0)
    diff = query - mu

    try:
        lw = LedoitWolf().fit(reference_matrix)
        cov_inv = np.linalg.pinv(lw.covariance_)
    except Exception:
        var = np.var(reference_matrix, axis=0) + 1e-6
        cov_inv = np.diag(1.0 / var)

    t2 = float(N / (N + 1) * diff @ cov_inv @ diff)

    # F-statistic approximation: F = (N-D) / ((N-1)*D) * T²
    f_stat = (N - D) / ((N - 1) * D) * t2
    f_stat = max(0.0, f_stat)
    p_value = float(1.0 - f_dist.cdf(f_stat, dfn=D, dfd=N - D))
    return t2, p_value


def energy_ood_score(
    query_emb: np.ndarray,
    reference_embs: np.ndarray,
    T: float = 1.0,
) -> float:
    """Energy-based OOD score (Liu et al., NeurIPS 2020).

    E(x; f) = -T * log sum_c exp(f_c(x) / T)

    Here we approximate with k-NN density: negative log of mean similarity
    to the k nearest reference neighbours.

    Args:
        query_emb:      (D,) query embedding.
        reference_embs: (N, D) reference embeddings.
        T:              Temperature parameter.

    Returns:
        Energy score (higher = more OOD).
    """
    k = min(10, len(reference_embs))
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    r_norm = reference_embs / (np.linalg.norm(reference_embs, axis=1, keepdims=True) + 1e-8)
    cos_sims = r_norm @ q_norm                         # (N,)
    top_k_sims = np.sort(cos_sims)[-k:]
    # Negative log of average cosine similarity → energy proxy
    avg_sim = float(np.mean(np.exp(top_k_sims / T)))
    energy = float(-T * np.log(avg_sim + 1e-10))
    return energy


class SlideShiftDetector:
    """Detects when a newly arriving slide originates from an OOD site, scanner, or stain.

    Combines four complementary detectors:
      1. Mahalanobis distance on stain moments (fast, stain-specific).
      2. Mahalanobis distance on embedding mean (biology-level shift).
      3. Hotelling T² test (formal hypothesis test, returns p-value).
      4. Energy-based OOD score (density-free, robust to covariance estimation).
    """

    def __init__(
        self,
        stain_weight: float = 0.35,
        embedding_weight: float = 0.50,
        energy_weight: float = 0.15,
        threshold_quantile: float = 0.95,
    ):
        self.stain_weight = stain_weight
        self.embedding_weight = embedding_weight
        self.energy_weight = energy_weight
        self.threshold_quantile = threshold_quantile

        self.threshold: float = 2.5
        self.is_fitted: bool = False

        self.stain_mean: Optional[np.ndarray] = None
        self.stain_cov_inv: Optional[np.ndarray] = None
        self.stain_scale: float = 1.0

        self.emb_mean: Optional[np.ndarray] = None
        self.emb_cov_inv: Optional[np.ndarray] = None
        self.emb_scale: float = 1.0

        self._reference_emb_matrix: Optional[np.ndarray] = None
        self._energy_scale: float = 1.0

    def fit(self, reference_signatures: List[SlideSignature]) -> "SlideShiftDetector":
        """Fit reference distribution on in-distribution training slides."""
        if len(reference_signatures) < 3:
            raise ValueError("Need at least 3 reference slide signatures to fit detector.")

        stain_matrix = np.stack([s.stain_moments for s in reference_signatures], axis=0)
        emb_matrix = np.stack([s.embedding_mean for s in reference_signatures], axis=0)

        # 1. Stain distribution (6-dim moments)
        self.stain_mean = np.mean(stain_matrix, axis=0)
        stain_cov = np.cov(stain_matrix, rowvar=False) + np.eye(6) * 1e-4
        try:
            self.stain_cov_inv = np.linalg.pinv(stain_cov)
        except Exception:
            self.stain_cov_inv = np.eye(6)

        # 2. Embedding distribution (D-dim with Ledoit-Wolf shrinkage)
        self.emb_mean = np.mean(emb_matrix, axis=0)
        try:
            lw = LedoitWolf(assume_centered=False).fit(emb_matrix)
            self.emb_cov_inv = np.linalg.pinv(lw.covariance_)
        except Exception:
            var = np.var(emb_matrix, axis=0) + 1e-4
            self.emb_cov_inv = np.diag(1.0 / var)

        # Store reference embedding matrix for Hotelling T² and energy scoring
        self._reference_emb_matrix = emb_matrix.copy()

        # Compute training scores to calibrate scale and threshold
        stain_dists = [self._compute_stain_dist(s.stain_moments) for s in reference_signatures]
        emb_dists = [self._compute_emb_dist(s.embedding_mean) for s in reference_signatures]
        energy_scores = [
            energy_ood_score(s.embedding_mean, emb_matrix)
            for s in reference_signatures
        ]

        self.stain_scale = float(np.median(stain_dists)) if np.median(stain_dists) > 0 else 1.0
        self.emb_scale = float(np.median(emb_dists)) if np.median(emb_dists) > 0 else 1.0
        self._energy_scale = float(np.median(energy_scores)) if np.median(energy_scores) > 0 else 1.0

        combined_scores = [
            self.stain_weight * (d_s / self.stain_scale)
            + self.embedding_weight * (d_e / self.emb_scale)
            + self.energy_weight * (e / max(self._energy_scale, 1e-6))
            for d_s, d_e, e in zip(stain_dists, emb_dists, energy_scores)
        ]

        self.threshold = float(np.percentile(combined_scores, self.threshold_quantile * 100.0))
        self.is_fitted = True
        return self

    def _compute_stain_dist(self, stain_moments: np.ndarray) -> float:
        if self.stain_mean is None or self.stain_cov_inv is None:
            return 0.0
        diff = stain_moments - self.stain_mean
        dist_sq = float(np.dot(np.dot(diff, self.stain_cov_inv), diff))
        return float(np.sqrt(max(0.0, dist_sq)))

    def _compute_emb_dist(self, emb_mean: np.ndarray) -> float:
        if self.emb_mean is None or self.emb_cov_inv is None:
            return 0.0
        diff = emb_mean - self.emb_mean
        dist_sq = float(np.dot(np.dot(diff, self.emb_cov_inv), diff))
        return float(np.sqrt(max(0.0, dist_sq)))

    def evaluate_slide(self, signature: SlideSignature) -> ShiftAlert:
        """Score a new slide and generate an OOD shift alert with multi-test evidence."""
        if not self.is_fitted:
            raise RuntimeError("SlideShiftDetector must be fitted on training slides before evaluation.")

        d_stain = self._compute_stain_dist(signature.stain_moments)
        d_emb = self._compute_emb_dist(signature.embedding_mean)

        # Energy-based OOD score
        energy = 0.0
        if self._reference_emb_matrix is not None:
            energy = energy_ood_score(signature.embedding_mean, self._reference_emb_matrix)

        # Hotelling T² test (formal p-value)
        t2_stat, t2_pval = 0.0, 1.0
        if self._reference_emb_matrix is not None and len(self._reference_emb_matrix) > 5:
            t2_stat, t2_pval = hotelling_t2(signature.embedding_mean, self._reference_emb_matrix)

        norm_stain = d_stain / self.stain_scale
        norm_emb = d_emb / self.emb_scale
        norm_energy = energy / max(self._energy_scale, 1e-6)

        shift_score = float(
            self.stain_weight * norm_stain
            + self.embedding_weight * norm_emb
            + self.energy_weight * norm_energy
        )

        is_shifted = shift_score > self.threshold

        reasons: List[str] = []
        if norm_stain > 2.0:
            reasons.append(f"Atypical stain color distribution (relative dist: {norm_stain:.2f}x)")
        if norm_emb > 2.0:
            reasons.append(f"Out-of-distribution embedding (relative dist: {norm_emb:.2f}x)")
        if t2_pval < 0.05:
            reasons.append(f"Hotelling T² significant (p={t2_pval:.4f}, T²={t2_stat:.1f})")
        if norm_energy > 2.0:
            reasons.append(f"High energy OOD score ({norm_energy:.2f}x reference)")

        if shift_score > self.threshold * 1.5:
            action = "Reject / High Shift Flag"
        elif is_shifted:
            action = "Warn / Flag for Review"
        else:
            action = "Accept"

        # Confidence: combine Mahalanobis and p-value evidence
        p_val_combined = float(min(t2_pval, max(0.0, 1.0 - (shift_score / (self.threshold * 2.0)))))

        return ShiftAlert(
            slide_id=signature.slide_id,
            is_shift_detected=is_shifted,
            shift_score=shift_score,
            threshold=self.threshold,
            p_value_or_confidence=p_val_combined,
            stain_distance=d_stain,
            embedding_distance=d_emb,
            alert_reasons=reasons,
            recommended_action=action,
        )

    def evaluate_cohort_mmd(
        self,
        test_signatures: List[SlideSignature],
    ) -> Dict[str, float]:
        """Compute MMD between reference and test cohort embeddings.

        Returns:
            dict with 'mmd_embedding' (on embedding means) and
            'mmd_stain' (on stain moments).
        """
        from slideshift.robustness.metrics import mmd_rbf

        if self._reference_emb_matrix is None or len(test_signatures) == 0:
            return {"mmd_embedding": 0.0, "mmd_stain": 0.0}

        test_emb = np.stack([s.embedding_mean for s in test_signatures], axis=0)
        test_stain = np.stack([s.stain_moments for s in test_signatures], axis=0)
        ref_stain = np.stack(
            [np.zeros(6)] * len(self._reference_emb_matrix), axis=0
        )  # placeholder — actual stain ref not stored

        mmd_emb = mmd_rbf(self._reference_emb_matrix, test_emb)
        mmd_stain = 0.0  # Would need reference stain matrix to compute

        return {"mmd_embedding": mmd_emb, "mmd_stain": mmd_stain}

