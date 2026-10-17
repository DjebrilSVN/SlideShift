"""Robustness benchmarking suite for natural and synthetic domain shifts."""

from slideshift.robustness.perturbations import (
    PERTURBATION_REGISTRY,
    apply_he_stain_jitter,
    apply_hue_saturation_shift,
    apply_gaussian_blur,
    apply_jpeg_compression,
    apply_additive_noise,
    apply_resolution_mismatch,
)
from slideshift.robustness.metrics import (
    compute_prediction_flip_rate,
    compute_heatmap_stability,
    evaluate_shift_degradation,
)
from slideshift.robustness.embedding_shift import (
    evaluate_site_linear_probe,
    compute_knn_neighborhood_purity,
)
from slideshift.robustness.shortcuts import evaluate_shortcut_baseline
from slideshift.robustness.lab import RobustnessLab

__all__ = [
    "PERTURBATION_REGISTRY",
    "apply_he_stain_jitter",
    "apply_hue_saturation_shift",
    "apply_gaussian_blur",
    "apply_jpeg_compression",
    "apply_additive_noise",
    "apply_resolution_mismatch",
    "compute_prediction_flip_rate",
    "compute_heatmap_stability",
    "evaluate_shift_degradation",
    "evaluate_site_linear_probe",
    "compute_knn_neighborhood_purity",
    "evaluate_shortcut_baseline",
    "RobustnessLab",
]

