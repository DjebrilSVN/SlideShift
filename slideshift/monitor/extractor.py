"""Slide signature extraction for distribution shift detection."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
from slideshift.core.types import SlideBag


@dataclass
class SlideSignature:
    """Compact statistical fingerprint of a slide's stain profile and embedding distribution."""
    slide_id: str
    stain_moments: np.ndarray  # (6,) [mean_L, mean_A, mean_B, std_L, std_A, std_B]
    embedding_mean: np.ndarray  # (D,) Mean tile feature vector
    embedding_std: np.ndarray  # (D,) Std of tile feature vectors
    metadata: Dict[str, any]


class SlideSignatureExtractor:
    """Extracts low-dimensional stain and feature space statistics from slide bags and thumbnails."""

    @staticmethod
    def extract_from_bag(
        bag: SlideBag,
        thumbnail_rgb: Optional[np.ndarray] = None,
    ) -> SlideSignature:
        """Compute slide signature from extracted features and optional thumbnail."""
        # 1. Feature space moments
        if bag.num_tiles > 0:
            feat_f = bag.features.astype(np.float32)
            emb_mean = np.mean(feat_f, axis=0)
            emb_std = np.std(feat_f, axis=0)
        else:
            dim = bag.feature_dim or 512
            emb_mean = np.zeros(dim, dtype=np.float32)
            emb_std = np.zeros(dim, dtype=np.float32)

        # 2. Stain color moments (CIELAB)
        if thumbnail_rgb is not None and thumbnail_rgb.size > 0:
            # Mask out background (>225 average)
            mask = np.mean(thumbnail_rgb, axis=-1) < 225
            if np.any(mask):
                pixels = thumbnail_rgb[mask].astype(np.float32) / 255.0
                # Fast perceptual LAB approximation
                r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
                l_c = 0.299 * r + 0.587 * g + 0.114 * b
                a_c = 0.500 * r - 0.419 * g - 0.081 * b
                b_c = -0.169 * r - 0.331 * g + 0.500 * b

                stain_moments = np.array([
                    np.mean(l_c), np.mean(a_c), np.mean(b_c),
                    np.std(l_c), np.std(a_c), np.std(b_c),
                ], dtype=np.float32)
            else:
                stain_moments = np.zeros(6, dtype=np.float32)
        else:
            # Fallback to feature-derived color proxy
            stain_moments = np.zeros(6, dtype=np.float32)

        return SlideSignature(
            slide_id=bag.slide_id,
            stain_moments=stain_moments,
            embedding_mean=emb_mean,
            embedding_std=emb_std,
            metadata={"num_tiles": bag.num_tiles},
        )

