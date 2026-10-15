"""Regularization techniques for Multiple-Instance Learning under domain shift."""

from typing import Optional, Tuple
import numpy as np


def apply_instance_dropout(
    features: np.ndarray,
    coordinates: np.ndarray,
    dropout_rate: float = 0.10,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly drop a fraction of tiles from the bag during training.
    
    Prevents the MIL aggregator from over-relying on a single dominant tile or artifact.
    """
    if dropout_rate <= 0.0 or len(features) <= 2:
        return features, coordinates

    if rng is None:
        rng = np.random.RandomState()

    n = len(features)
    keep_count = max(2, int(round(n * (1.0 - dropout_rate))))
    keep_indices = rng.choice(n, size=keep_count, replace=False)
    keep_indices.sort()

    return features[keep_indices], coordinates[keep_indices]


def apply_feature_noise(
    features: np.ndarray,
    noise_std: float = 0.02,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Add small Gaussian jitter to tile embedding vectors during training.
    
    Smooths decision boundaries in feature space and regularizes attention scoring.
    """
    if noise_std <= 0.0 or len(features) == 0:
        return features

    if rng is None:
        rng = np.random.RandomState()

    noise = rng.normal(0.0, noise_std, size=features.shape).astype(features.dtype)
    return features + noise
