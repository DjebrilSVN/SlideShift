"""Advanced conformal prediction for MIL slide classifiers.

Provides distribution-free coverage guarantees at the slide level:
  - Standard conformal prediction sets (Angelopoulos & Bates, 2021).
  - Adaptive prediction sets (RAPS) that shrink or grow with difficulty.
  - A "defer to pathologist" threshold: when the prediction set contains
    all classes, flag the slide for expert review.

Reference:
  Angelopoulos et al., "Learn then Test: Calibrating Predictive Algorithms to
  Achieve Risk Control", 2021.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class ConformalResult:
    """Per-slide conformal prediction output."""
    slide_id: str
    predicted_class: int
    prediction_set: Set[int]
    conformity_score: float
    defer_to_expert: bool
    true_label: Optional[int] = None


@dataclass
class ConformalEvaluation:
    """Aggregate evaluation of conformal coverage and efficiency."""
    coverage: float              # Empirical coverage ≥ target_coverage
    average_set_size: float      # Smaller = more informative
    defer_rate: float            # Fraction needing expert review
    target_coverage: float
    num_samples: int
    coverage_satisfied: bool


class ConformalPredictor:
    """Conformal predictor for multi-class slide classification.

    Workflow:
      1. Call ``calibrate()`` on a held-out calibration set (≥ 50 slides).
      2. Call ``predict()`` on new slides to get calibrated prediction sets.
      3. Check ``ConformalResult.defer_to_expert`` for ambiguous slides.

    Args:
        target_coverage: Desired marginal coverage (e.g. 0.90 = 90% guarantee).
        method:          'lac' (Least Ambiguous Classifier) or 'raps'
                         (Regularized Adaptive Prediction Sets).
        raps_lambda:     Regularisation strength for RAPS.
        raps_k_reg:      Regularisation threshold for RAPS.
    """

    def __init__(
        self,
        target_coverage: float = 0.90,
        method: str = "lac",
        raps_lambda: float = 0.1,
        raps_k_reg: int = 1,
    ):
        if not 0 < target_coverage < 1:
            raise ValueError("target_coverage must be in (0, 1).")
        self.target_coverage = target_coverage
        self.method = method
        self.raps_lambda = raps_lambda
        self.raps_k_reg = raps_k_reg

        self._qhat: Optional[float] = None
        self._num_classes: int = 2
        self._is_calibrated: bool = False

    def _conformity_score_lac(self, probs: np.ndarray, label: int) -> float:
        """1 - p(true_class)  ∈ [0, 1]."""
        return float(1.0 - probs[label])

    def _conformity_score_raps(
        self,
        probs: np.ndarray,
        label: int,
        o: np.ndarray,
    ) -> float:
        """Regularised score (sum of probabilities up to and including label
        in the sorted order) plus a penalty for uncertainty.

        Args:
            probs: (C,) softmax probabilities.
            label: True class index.
            o:     Argsort of probs descending (pre-computed).
        """
        rank = int(np.where(o == label)[0][0])
        cum_prob = float(np.sum(probs[o[: rank + 1]]))
        penalty = self.raps_lambda * max(0, rank + 1 - self.raps_k_reg)
        return cum_prob + penalty

    def calibrate(
        self,
        calibration_probs: np.ndarray,
        calibration_labels: np.ndarray,
    ) -> "ConformalPredictor":
        """Compute conformal threshold from a calibration set.

        Args:
            calibration_probs:  (N, C) softmax probabilities on calibration slides.
            calibration_labels: (N,)  true integer labels.

        Returns:
            self (for chaining).
        """
        N, C = calibration_probs.shape
        self._num_classes = C

        scores = np.zeros(N, dtype=np.float64)
        for i in range(N):
            y = int(calibration_labels[i])
            if self.method == "raps":
                o = np.argsort(calibration_probs[i])[::-1]
                scores[i] = self._conformity_score_raps(calibration_probs[i], y, o)
            else:
                scores[i] = self._conformity_score_lac(calibration_probs[i], y)

        # (1 - alpha) quantile with finite-sample correction
        alpha = 1.0 - self.target_coverage
        level = np.ceil((N + 1) * (1 - alpha)) / N
        level = np.clip(level, 0.0, 1.0)
        self._qhat = float(np.quantile(scores, level))
        self._is_calibrated = True
        return self

    def predict(
        self,
        probs: np.ndarray,
        slide_id: str = "slide",
        true_label: Optional[int] = None,
    ) -> ConformalResult:
        """Build a prediction set for one slide.

        Args:
            probs:     (C,) softmax probabilities.
            slide_id:  Identifier for logging.
            true_label: Optional ground truth for evaluation.

        Returns:
            ConformalResult with set-valued prediction.
        """
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before predict().")

        C = len(probs)
        o = np.argsort(probs)[::-1]

        prediction_set: Set[int] = set()
        cum_score = 0.0

        for rank, cls in enumerate(o):
            if self.method == "raps":
                s = float(np.sum(probs[o[: rank + 1]])) + self.raps_lambda * max(
                    0, rank + 1 - self.raps_k_reg
                )
            else:
                s = float(1.0 - probs[cls])

            prediction_set.add(int(cls))
            cum_score = s

            if s >= self._qhat:
                break

        # Conformity score for this prediction
        if self.method == "raps":
            conform_score = float(np.sum(probs[o[: len(prediction_set)]])) + self.raps_lambda * max(
                0, len(prediction_set) - self.raps_k_reg
            )
        else:
            conform_score = float(1.0 - probs[int(o[0])])

        defer = len(prediction_set) == C

        return ConformalResult(
            slide_id=slide_id,
            predicted_class=int(o[0]),
            prediction_set=prediction_set,
            conformity_score=conform_score,
            defer_to_expert=defer,
            true_label=true_label,
        )

    def evaluate_coverage(
        self,
        test_probs: np.ndarray,
        test_labels: np.ndarray,
        slide_ids: Optional[List[str]] = None,
    ) -> ConformalEvaluation:
        """Measure empirical coverage, set size, and defer rate on a test set."""
        N = len(test_labels)
        if slide_ids is None:
            slide_ids = [f"slide_{i}" for i in range(N)]

        results = [
            self.predict(test_probs[i], slide_ids[i], int(test_labels[i]))
            for i in range(N)
        ]

        covered = sum(
            1 for r in results if r.true_label is not None and r.true_label in r.prediction_set
        )
        coverage = covered / N if N > 0 else 0.0
        avg_set_size = float(np.mean([len(r.prediction_set) for r in results]))
        defer_rate = float(np.mean([r.defer_to_expert for r in results]))

        return ConformalEvaluation(
            coverage=coverage,
            average_set_size=avg_set_size,
            defer_rate=defer_rate,
            target_coverage=self.target_coverage,
            num_samples=N,
            coverage_satisfied=coverage >= self.target_coverage,
        )

    def temperature_scale_and_calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        temperatures: Optional[List[float]] = None,
    ) -> "ConformalPredictor":
        """Find the optimal temperature and calibrate the conformal predictor.

        Temperature scaling (Guo et al., 2017) is applied before conformal
        calibration for best coverage.
        """
        if temperatures is None:
            temperatures = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]

        best_T = 1.0
        best_ece = float("inf")

        for T in temperatures:
            scaled = self._apply_temperature(cal_probs, T)
            ece = self._compute_ece(scaled, cal_labels)
            if ece < best_ece:
                best_ece = ece
                best_T = T

        # Apply best temperature to calibration probs then calibrate
        scaled_probs = self._apply_temperature(cal_probs, best_T)
        self.best_temperature = best_T
        self.calibrate(scaled_probs, cal_labels)
        return self

    @staticmethod
    def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
        """Apply temperature scaling to probability vectors."""
        log_probs = np.log(probs + 1e-10)
        scaled = log_probs / T
        max_s = scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        return (exp_s / exp_s.sum(axis=1, keepdims=True)).astype(np.float32)

    @staticmethod
    def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error for choosing temperature."""
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)
        ece = 0.0
        N = len(labels)
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            mask = (confidences > lo) & (confidences <= hi)
            n_b = mask.sum()
            if n_b > 0:
                ece += (n_b / N) * abs(accuracies[mask].mean() - confidences[mask].mean())
        return float(ece)

