"""Tissue segmentation using saturation and Otsu thresholding with morphological operations."""

from typing import Tuple
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import filters, morphology


def segment_tissue(
    thumbnail_rgb: np.ndarray,
    min_speck_size: int = 50,
    hole_fill_size: int = 200,
    otsu_multiplier: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Segment tissue from slide glass background on a low-resolution thumbnail.
    
    Uses saturation channel thresholding and Otsu method to distinguish
    stained tissue (pink/purple) from clear glass / mounting medium.

    Args:
        thumbnail_rgb: RGB image as uint8 numpy array (H, W, 3).
        min_speck_size: Minimum pixel area to keep as valid tissue island.
        hole_fill_size: Max hole area to fill in binary mask.
        otsu_multiplier: Sensitivity multiplier for Otsu threshold.

    Returns:
        (binary_tissue_mask, tissue_fraction)
        binary_tissue_mask: boolean array (H, W) where True indicates tissue.
        tissue_fraction: float in [0.0, 1.0].
    """
    if thumbnail_rgb.ndim != 3 or thumbnail_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {thumbnail_rgb.shape}")

    # Convert to float [0, 1]
    img_f = thumbnail_rgb.astype(np.float32) / 255.0

    # Convert RGB to HSV to extract the Saturation channel
    # Tissue (eosin/hematoxylin) has distinct color saturation compared to glass
    max_c = np.max(img_f, axis=-1)
    min_c = np.min(img_f, axis=-1)
    delta = max_c - min_c

    saturation = np.zeros_like(max_c)
    non_zero = max_c > 1e-5
    saturation[non_zero] = delta[non_zero] / max_c[non_zero]

    # Also compute grayscale intensity to reject pure bright white regions
    grayscale = 0.2989 * img_f[..., 0] + 0.5870 * img_f[..., 1] + 0.1140 * img_f[..., 2]

    # Otsu threshold on saturation
    try:
        sat_thresh = filters.threshold_otsu(saturation) * otsu_multiplier
    except Exception:
        sat_thresh = 0.05

    # Tissue is either sufficiently saturated OR sufficiently dark compared to bright glass (>0.92)
    tissue_mask = (saturation > sat_thresh) & (grayscale < 0.94)

    # Morphological closing to bridge small intra-tissue gaps
    struct = morphology.disk(3)
    if hasattr(morphology, "closing"):
        closed = morphology.closing(tissue_mask, struct)
    else:
        closed = morphology.binary_closing(tissue_mask, struct)

    # Fill small holes
    filled = ndimage.binary_fill_holes(closed)

    # Remove small specks (dust, glass imperfections)
    try:
        cleaned = morphology.remove_small_objects(filled, max_size=min_speck_size)
    except TypeError:
        cleaned = morphology.remove_small_objects(filled, min_size=min_speck_size)

    # Calculate tissue fraction
    tissue_fraction = float(np.mean(cleaned))

    return cleaned, tissue_fraction
