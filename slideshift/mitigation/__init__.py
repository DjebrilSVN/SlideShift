"""Shift mitigations: stain normalization, standardization, DANN, ensembling."""

from slideshift.mitigation.stain_norm import (
    BaseStainNormalizer,
    ReinhardNormalizer,
    MacenkoNormalizer,
)
from slideshift.mitigation.standardization import standardize_slide_bag
from slideshift.mitigation.domain_adversarial import DomainAdversarialMIL
from slideshift.mitigation.ensemble import ensemble_predictions

__all__ = [
    "BaseStainNormalizer",
    "ReinhardNormalizer",
    "MacenkoNormalizer",
    "standardize_slide_bag",
    "DomainAdversarialMIL",
    "ensemble_predictions",
]

