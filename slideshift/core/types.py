"""Core types and data representations for SlideShift."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class SlideFormat(str, Enum):
    """Supported whole slide image formats."""
    SVS = "svs"
    TIF = "tif"
    TIFF = "tiff"
    NDPI = "ndpi"
    MRXS = "mrxs"
    SYNTHETIC = "synthetic"
    GENERIC = "generic"


@dataclass
class QCReport:
    """Quality control metrics for a whole-slide image."""
    slide_id: str
    blur_score: float  # Variance of Laplacian (higher is sharper)
    is_blurry: bool
    tissue_fraction: float  # Fraction of slide containing valid tissue
    fold_fraction: float  # Fraction of tissue area suspected of tissue folding
    pen_mark_fraction: float  # Fraction of area with pen/marker ink
    bubble_fraction: float  # Fraction of area with air bubbles / mounting voids
    passed_qc: bool
    flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "blur_score": float(self.blur_score),
            "is_blurry": bool(self.is_blurry),
            "tissue_fraction": float(self.tissue_fraction),
            "fold_fraction": float(self.fold_fraction),
            "pen_mark_fraction": float(self.pen_mark_fraction),
            "bubble_fraction": float(self.bubble_fraction),
            "passed_qc": bool(self.passed_qc),
            "flags": self.flags,
            "metadata": self.metadata,
        }


@dataclass
class TileMetadata:
    """Metadata for an individual patch/tile extracted from a slide."""
    tile_id: str
    slide_id: str
    coord_x: int  # Top-left x coordinate at level 0 (physical / native base)
    coord_y: int  # Top-left y coordinate at level 0
    tile_size_px: int  # Pixel width/height of the extracted patch
    target_mpp: float  # Target resolution in microns per pixel
    native_mpp: float  # Native slide resolution in microns per pixel
    tissue_content: float  # Percentage of tissue in the tile (0.0 to 1.0)
    qc_flags: List[str] = field(default_factory=list)


@dataclass
class SlideBag:
    """In-memory representation of a slide's extracted feature bag."""
    slide_id: str
    features: np.ndarray  # Shape: (N, D) - N tiles, D embedding dimension
    coordinates: np.ndarray  # Shape: (N, 2) - [x, y] coordinates at level 0
    label: Optional[int] = None
    patient_id: Optional[str] = None
    site: Optional[str] = None
    scanner: Optional[str] = None
    qc_report: Optional[QCReport] = None
    tile_mpp: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_tiles(self) -> int:
        return self.features.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[1] if self.features.ndim > 1 else 0


@dataclass
class PredictionOutput:
    """Slide-level classifier prediction."""
    slide_id: str
    logits: np.ndarray  # (C,)
    probabilities: np.ndarray  # (C,)
    predicted_class: int
    true_label: Optional[int] = None
    attention_scores: Optional[np.ndarray] = None  # (N,) raw or normalized attention
    slide_embedding: Optional[np.ndarray] = None  # (D_slide,) aggregated slide embedding
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeatmapResult:
    """Attention heatmap visualization and validation output."""
    slide_id: str
    attention_percentiles: np.ndarray  # (N,) percentile ranks [0, 1]
    coordinates: np.ndarray  # (N, 2)
    thumbnail: np.ndarray  # Low-res RGB thumbnail
    overlay: np.ndarray  # RGBA or blended RGB heatmap overlay
    tumor_overlap_auroc: Optional[float] = None
    top_k_tumor_overlap: Optional[float] = None
    deletion_curve: Optional[List[float]] = None
    insertion_curve: Optional[List[float]] = None
    artifact_attention_fraction: Optional[Dict[str, float]] = None


@dataclass
class ShiftAlert:
    """Result from the Shift Monitor for a test slide."""
    slide_id: str
    is_shift_detected: bool
    shift_score: float  # Outlier score (e.g. Mahalanobis distance or MMD)
    threshold: float
    p_value_or_confidence: float
    stain_distance: float
    embedding_distance: float
    alert_reasons: List[str] = field(default_factory=list)
    recommended_action: str = "Accept"  # "Accept", "Warn", "Flag for Review", "Reject"

