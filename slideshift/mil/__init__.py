"""Multiple-Instance Learning modules for whole-slide classification."""

from slideshift.mil.dataset import SlideBagDataset
from slideshift.mil.regularization import apply_instance_dropout, apply_feature_noise
from slideshift.mil.aggregators import (
    MeanPoolingMIL,
    MaxPoolingMIL,
    AttentionMIL,
    CLAM_SB,
    CLAM_MB,
    TransMIL,
    build_aggregator,
)
from slideshift.mil.trainer import (
    MILTrainer,
    compute_expected_calibration_error,
    compute_bootstrap_ci,
)

__all__ = [
    "SlideBagDataset",
    "apply_instance_dropout",
    "apply_feature_noise",
    "MeanPoolingMIL",
    "MaxPoolingMIL",
    "AttentionMIL",
    "CLAM_SB",
    "CLAM_MB",
    "TransMIL",
    "build_aggregator",
    "MILTrainer",
    "compute_expected_calibration_error",
    "compute_bootstrap_ci",
]

