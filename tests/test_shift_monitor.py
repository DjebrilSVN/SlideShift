"""Tests for Shift Monitor: signature extraction, detector, and OOD alerts."""

import pytest
import numpy as np
from slideshift.core.types import SlideBag
from slideshift.monitor.extractor import SlideSignatureExtractor
from slideshift.monitor.detector import SlideShiftDetector
from slideshift.monitor.calibrator import evaluate_shift_monitor


def test_signature_and_detector():
    # In-distribution training slides
    ref_sigs = []
    for i in range(15):
        bag = SlideBag(
            slide_id=f"ref_{i}",
            features=np.random.normal(0.0, 1.0, size=(30, 32)).astype(np.float32),
            coordinates=np.zeros((30, 2), dtype=int),
        )
        thumb = np.full((64, 64, 3), 180, dtype=np.uint8)
        sig = SlideSignatureExtractor.extract_from_bag(bag, thumb)
        ref_sigs.append(sig)

    detector = SlideShiftDetector(threshold_quantile=0.95)
    detector.fit(ref_sigs)
    assert detector.is_fitted is True

    # Test an in-distribution slide
    test_id_bag = SlideBag(
        slide_id="test_id",
        features=np.random.normal(0.0, 1.0, size=(30, 32)).astype(np.float32),
        coordinates=np.zeros((30, 2), dtype=int),
    )
    sig_id = SlideSignatureExtractor.extract_from_bag(test_id_bag, np.full((64, 64, 3), 180, dtype=np.uint8))
    alert_id = detector.evaluate_slide(sig_id)

    # Test an out-of-distribution shifted slide (mean shifted by +5.0)
    test_ood_bag = SlideBag(
        slide_id="test_ood",
        features=np.random.normal(5.0, 1.0, size=(30, 32)).astype(np.float32),
        coordinates=np.zeros((30, 2), dtype=int),
    )
    sig_ood = SlideSignatureExtractor.extract_from_bag(test_ood_bag, np.full((64, 64, 3), 60, dtype=np.uint8))
    alert_ood = detector.evaluate_slide(sig_ood)

    assert alert_ood.shift_score > alert_id.shift_score
    assert alert_ood.is_shift_detected is True
    assert alert_ood.recommended_action != "Accept"


def test_monitor_calibration():
    # Evaluate TPR vs FPR
    ref_sigs = [
        SlideSignatureExtractor.extract_from_bag(
            SlideBag(f"ref_{i}", np.random.normal(0, 1, (20, 16)), np.zeros((20, 2)))
        ) for i in range(12)
    ]
    detector = SlideShiftDetector().fit(ref_sigs)

    ood_sigs = [
        SlideSignatureExtractor.extract_from_bag(
            SlideBag(f"ood_{i}", np.random.normal(4, 1, (20, 16)), np.zeros((20, 2)))
        ) for i in range(8)
    ]

    metrics = evaluate_shift_monitor(detector, ref_sigs, ood_sigs)
    assert metrics["shift_detection_auroc"] >= 0.80

