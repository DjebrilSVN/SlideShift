"""Per-slide feature standardization for fast shift mitigation."""

from typing import Union
import numpy as np
from slideshift.core.types import SlideBag


def standardize_slide_bag(
    bag: SlideBag,
    method: str = "zscore",
    eps: float = 1e-7,
) -> SlideBag:
    """Apply feature normalization across all tiles in a slide bag.

    Args:
        bag: Original SlideBag with features (N, D).
        method:
            'zscore': Zero mean and unit variance per feature dimension across the slide's tiles.
            'l2': Project each tile feature vector to unit sphere (L2 norm = 1).
            'center': Subtract mean embedding across tiles.

    Returns:
        New SlideBag with standardized features.
    """
    if bag.num_tiles == 0:
        return bag

    feats = bag.features.astype(np.float32)

    if method == "zscore":
        mean = np.mean(feats, axis=0, keepdims=True)
        std = np.std(feats, axis=0, keepdims=True) + eps
        norm_feats = (feats - mean) / std

    elif method == "l2":
        norms = np.linalg.norm(feats, axis=-1, keepdims=True) + eps
        norm_feats = feats / norms

    elif method == "center":
        mean = np.mean(feats, axis=0, keepdims=True)
        norm_feats = feats - mean

    else:
        norm_feats = feats

    return SlideBag(
        slide_id=bag.slide_id,
        features=norm_feats.astype(bag.features.dtype),
        coordinates=bag.coordinates,
        label=bag.label,
        patient_id=bag.patient_id,
        site=bag.site,
        scanner=bag.scanner,
        qc_report=bag.qc_report,
        tile_mpp=bag.tile_mpp,
        metadata={**bag.metadata, "mitigation": f"standardize_{method}"},
    )

