"""Integrated production inference combining classifier prediction, confidence, and shift alert."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from slideshift.core.types import SlideBag, PredictionOutput, ShiftAlert
from slideshift.mil.trainer import MILTrainer
from slideshift.monitor.detector import SlideShiftDetector
from slideshift.monitor.extractor import SlideSignatureExtractor, SlideSignature


@dataclass
class MonitoredInferenceResult:
    """Production deployment prediction bundle."""
    slide_id: str
    predicted_class: int
    confidence: float
    probabilities: np.ndarray
    shift_alert: ShiftAlert
    attention_scores: Optional[np.ndarray] = None
    slide_embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "predicted_class": self.predicted_class,
            "confidence": float(self.confidence),
            "probabilities": [float(p) for p in self.probabilities],
            "shift_warning": bool(self.shift_alert.is_shift_detected),
            "shift_score": float(self.shift_alert.shift_score),
            "threshold": float(self.shift_alert.threshold),
            "recommended_action": self.shift_alert.recommended_action,
            "alert_reasons": self.shift_alert.alert_reasons,
        }


class ShiftMonitoredClassifier:
    """Wraps MIL slide classifier with real-time shift monitoring."""

    def __init__(
        self,
        trainer: MILTrainer,
        detector: SlideShiftDetector,
    ):
        self.trainer = trainer
        self.detector = detector

    def predict_slide(
        self,
        bag: SlideBag,
        thumbnail_rgb: Optional[np.ndarray] = None,
    ) -> MonitoredInferenceResult:
        """Run classification inference, quantify confidence, and test for OOD shift."""
        # 1. Slide-level classification prediction
        pred: PredictionOutput = self.trainer.predict_bag(bag)
        conf = float(np.max(pred.probabilities))

        # 2. Extract signature and evaluate shift detector
        sig: SlideSignature = SlideSignatureExtractor.extract_from_bag(bag, thumbnail_rgb)
        alert: ShiftAlert = self.detector.evaluate_slide(sig)

        return MonitoredInferenceResult(
            slide_id=bag.slide_id,
            predicted_class=pred.predicted_class,
            confidence=conf,
            probabilities=pred.probabilities,
            shift_alert=alert,
            attention_scores=pred.attention_scores,
            slide_embedding=pred.slide_embedding,
        )

