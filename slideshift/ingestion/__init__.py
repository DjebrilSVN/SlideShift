"""Slide ingestion, tissue segmentation, quality control, and physical tiling."""

from slideshift.ingestion.tissue_mask import segment_tissue
from slideshift.ingestion.quality_control import (
    compute_blur_score,
    detect_pen_marks,
    detect_folds_and_tears,
    detect_air_bubbles,
    assess_slide_quality,
)
from slideshift.ingestion.tiling import SlideTiler
from slideshift.ingestion.synthetic_wsi import (
    generate_synthetic_slide_data,
    save_synthetic_slide,
)

__all__ = [
    "segment_tissue",
    "compute_blur_score",
    "detect_pen_marks",
    "detect_folds_and_tears",
    "detect_air_bubbles",
    "assess_slide_quality",
    "SlideTiler",
    "generate_synthetic_slide_data",
    "save_synthetic_slide",
]

