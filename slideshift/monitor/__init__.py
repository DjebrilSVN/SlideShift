"""Real-time slide distribution shift monitoring and out-of-distribution detection."""

from slideshift.monitor.extractor import (
    SlideSignature,
    SlideSignatureExtractor,
)
from slideshift.monitor.detector import SlideShiftDetector
from slideshift.monitor.calibrator import evaluate_shift_monitor
from slideshift.monitor.pipeline import (
    MonitoredInferenceResult,
    ShiftMonitoredClassifier,
)

__all__ = [
    "SlideSignature",
    "SlideSignatureExtractor",
    "SlideShiftDetector",
    "evaluate_shift_monitor",
    "MonitoredInferenceResult",
    "ShiftMonitoredClassifier",
]

