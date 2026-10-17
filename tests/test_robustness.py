"""Tests for robustness perturbations, degradation metrics, and shortcut baselines."""

import pytest
import numpy as np
from PIL import Image
from slideshift.robustness.perturbations import (
    apply_he_stain_jitter,
    apply_gaussian_blur,
    apply_jpeg_compression,
    apply_resolution_mismatch,
)
from slideshift.robustness.metrics import evaluate_shift_degradation
from slideshift.robustness.embedding_shift import evaluate_site_linear_probe, compute_knn_neighborhood_purity
from slideshift.robustness.shortcuts import evaluate_shortcut_baseline
from slideshift.mil.dataset import SlideBagDataset
from slideshift.core.types import SlideBag


def test_perturbations():
    img = Image.new("RGB", (64, 64), (200, 140, 180))

    jittered = apply_he_stain_jitter(img, severity=2)
    assert jittered.size == (64, 64)

    blurred = apply_gaussian_blur(img, severity=2)
    assert blurred.size == (64, 64)

    compressed = apply_jpeg_compression(img, severity=3)
    assert compressed.size == (64, 64)

    mismatch = apply_resolution_mismatch(img, severity=1)
    assert mismatch.size == (64, 64)


def test_degradation_metrics():
    y_true = np.array([0, 0, 1, 1])
    clean_probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    # Shifted: model degrades and flips one prediction
    shifted_probs = np.array([[0.6, 0.4], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])

    deg = evaluate_shift_degradation(y_true, clean_probs, shifted_probs)
    assert deg["clean_auroc"] == 1.0
    assert deg["delta_auroc"] >= 0.0
    assert deg["flip_rate"] == 0.25  # 1 out of 4 flipped


def test_site_linear_probe_and_shortcuts():
    # Construct a dataset where site strongly correlates with label (shortcut!)
    bags = []
    for i in range(12):
        site = "site_A" if i < 6 else "site_B"
        label = 0 if site == "site_A" else 1  # 100% confounded!
        bag = SlideBag(
            slide_id=f"slide_{i}",
            features=np.random.randn(10, 16).astype(np.float32),
            coordinates=np.zeros((10, 2), dtype=int),
            label=label,
            site=site,
        )
        bags.append(bag)

    ds = SlideBagDataset(bags)
    shortcut = evaluate_shortcut_baseline(ds)
    assert shortcut["shortcut_detected"] is True
    assert shortcut["site_only_auroc"] > 0.90

