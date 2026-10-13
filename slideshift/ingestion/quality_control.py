"""Slide quality control, artifact detection, and sharpness scoring."""

from typing import List, Tuple
import numpy as np
from PIL import Image
from scipy import ndimage
from slideshift.core.types import QCReport


def compute_blur_score(rgb_image: np.ndarray) -> float:
    """Compute sharpness using the variance of the 2D Laplacian operator.
    
    A higher score indicates sharp cellular details; low values indicate defocus blur.
    """
    if rgb_image.ndim == 3:
        # Grayscale luminance
        gray = 0.2989 * rgb_image[..., 0] + 0.5870 * rgb_image[..., 1] + 0.1140 * rgb_image[..., 2]
    else:
        gray = rgb_image.astype(np.float32)

    # 3x3 Laplacian filter kernel
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = ndimage.convolve(gray, laplacian_kernel, mode="reflect")
    return float(np.var(lap))


def detect_pen_marks(thumbnail_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect surgical / pathologist marker pen marks (blue, green, red, black ink).
    
    Pathologists often mark coverslips with felt markers, which neural networks
    frequently latch onto as high-attention shortcuts.
    """
    img_f = thumbnail_rgb.astype(np.float32) / 255.0
    r, g, b = img_f[..., 0], img_f[..., 1], img_f[..., 2]

    # Blue marker ink: B >> R and B >> G
    blue_marker = (b > 0.45) & (b > 1.35 * r) & (b > 1.25 * g)
    # Green marker ink: G >> R and G >> B
    green_marker = (g > 0.40) & (g > 1.30 * r) & (g > 1.25 * b)
    # Dark black ink marker: very low intensity with high edge sharpness outside tissue
    black_marker = (r < 0.20) & (g < 0.20) & (b < 0.20)

    marker_mask = blue_marker | green_marker | black_marker
    pen_fraction = float(np.mean(marker_mask))
    return marker_mask, pen_fraction


def detect_folds_and_tears(thumbnail_rgb: np.ndarray, tissue_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect tissue folds (hyperdense overlapping tissue) and tears.
    
    Tissue folds appear significantly darker and more saturated than monolayer stroma.
    """
    if not np.any(tissue_mask):
        return np.zeros_like(tissue_mask, dtype=bool), 0.0

    img_f = thumbnail_rgb.astype(np.float32) / 255.0
    intensity = np.mean(img_f, axis=-1)

    # Within tissue, folds have substantially lower transmission (<0.35)
    fold_mask = (intensity < 0.35) & tissue_mask
    tissue_pixels = np.sum(tissue_mask)
    fold_fraction = float(np.sum(fold_mask) / max(1, tissue_pixels))
    return fold_mask, fold_fraction


def detect_air_bubbles(thumbnail_rgb: np.ndarray, tissue_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect air bubbles and cover slip mounting voids."""
    img_f = thumbnail_rgb.astype(np.float32) / 255.0
    r, g, b = img_f[..., 0], img_f[..., 1], img_f[..., 2]

    # Air bubbles have high circularity and a characteristic dark refractive boundary
    # with a very bright or distorted interior
    brightness = (r + g + b) / 3.0
    bubble_interior = (brightness > 0.95) & (~tissue_mask)
    bubble_fraction = float(np.mean(bubble_interior))
    return bubble_interior, bubble_fraction


def assess_slide_quality(
    slide_id: str,
    thumbnail_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    tissue_fraction: float,
    blur_threshold: float = 80.0,
    min_tissue_fraction: float = 0.05,
    max_fold_fraction: float = 0.20,
    max_pen_fraction: float = 0.02,
) -> QCReport:
    """Run full quality control assessment on slide thumbnail.
    
    Records quality flags: blur score, fold area, pen marks, bubble artifacts, tissue fraction.
    These flags are saved with the slide and reused later in the robustness analysis,
    enabling the audit of whether pen marks and blur attract spurious high attention.
    """
    flags: List[str] = []

    # Sharpness / Blur
    blur_score = compute_blur_score(thumbnail_rgb)
    is_blurry = blur_score < blur_threshold
    if is_blurry:
        flags.append(f"BLUR_DETECTED (score={blur_score:.1f} < {blur_threshold})")

    # Insufficient tissue
    if tissue_fraction < min_tissue_fraction:
        flags.append(f"LOW_TISSUE_CONTENT ({tissue_fraction*100:.2f}% < {min_tissue_fraction*100:.1f}%)")

    # Pen marks
    _, pen_frac = detect_pen_marks(thumbnail_rgb)
    if pen_frac > max_pen_fraction:
        flags.append(f"PEN_MARKS_DETECTED ({pen_frac*100:.2f}%)")

    # Folds & tears
    _, fold_frac = detect_folds_and_tears(thumbnail_rgb, tissue_mask)
    if fold_frac > max_fold_fraction:
        flags.append(f"EXCESSIVE_FOLDS ({fold_frac*100:.2f}%)")

    # Air bubbles
    _, bubble_frac = detect_air_bubbles(thumbnail_rgb, tissue_mask)
    if bubble_frac > 0.15:
        flags.append(f"AIR_BUBBLES_DETECTED ({bubble_frac*100:.2f}%)")

    passed_qc = len(flags) == 0

    return QCReport(
        slide_id=slide_id,
        blur_score=blur_score,
        is_blurry=is_blurry,
        tissue_fraction=tissue_fraction,
        fold_fraction=fold_frac,
        pen_mark_fraction=pen_frac,
        bubble_fraction=bubble_frac,
        passed_qc=passed_qc,
        flags=flags,
    )

