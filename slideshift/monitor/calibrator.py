"""Calibration and validation of the slide shift monitor."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score
from slideshift.core.types import ShiftAlert
from slideshift.monitor.detector import SlideShiftDetector
from slideshift.monitor.extractor import SlideSignature


def evaluate_shift_monitor(
    detector: SlideShiftDetector,
    in_distribution_signatures: List[SlideSignature],
    shifted_signatures: List[SlideSignature],
    shifted_performance_drops: Optional[List[bool]] = None,
) -> Dict[str, float]:
    """Quantify shift detector performance: false alarm rate, detection rate, and AUROC.
    
    Args:
        detector: Fitted SlideShiftDetector.
        in_distribution_signatures: Validation slides from known training domain.
        shifted_signatures: Slides from shifted domains/scanners.
        shifted_performance_drops: Optional boolean mask indicating which shifted slides
                                   actually suffered a prediction error.

    Returns:
        Dictionary with false alarm rate, detection rate, and OOD detection AUROC.
    """
    id_alerts: List[ShiftAlert] = [detector.evaluate_slide(s) for s in in_distribution_signatures]
    ood_alerts: List[ShiftAlert] = [detector.evaluate_slide(s) for s in shifted_signatures]

    # False Alarm Rate on in-distribution data
    false_alarms = sum(1 for a in id_alerts if a.is_shift_detected)
    far = float(false_alarms / max(1, len(id_alerts)))

    # Detection Rate on shifted data
    true_detections = sum(1 for a in ood_alerts if a.is_shift_detected)
    detection_rate = float(true_detections / max(1, len(ood_alerts)))

    # Shift Discrimination AUROC
    y_true = np.concatenate([np.zeros(len(id_alerts)), np.ones(len(ood_alerts))])
    scores = np.concatenate([[a.shift_score for a in id_alerts], [a.shift_score for a in ood_alerts]])

    try:
        detection_auroc = float(roc_auc_score(y_true, scores))
    except Exception:
        detection_auroc = 0.5

    # If error masks provided: AUROC predicting actual model failures
    failure_auroc = float("nan")
    if shifted_performance_drops is not None and len(shifted_performance_drops) == len(shifted_signatures):
        has_failures = any(shifted_performance_drops) and not all(shifted_performance_drops)
        if has_failures:
            ood_scores = np.array([a.shift_score for a in ood_alerts])
            try:
                failure_auroc = float(roc_auc_score(np.array(shifted_performance_drops, dtype=int), ood_scores))
            except Exception:
                pass

    return {
        "false_alarm_rate": far,
        "detection_rate": detection_rate,
        "shift_detection_auroc": detection_auroc,
        "model_failure_prediction_auroc": failure_auroc,
        "in_distribution_count": len(id_alerts),
        "shifted_count": len(ood_alerts),
    }
