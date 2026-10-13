"""Tests for tissue segmentation, slide quality control, and physical tiling."""

import pytest
import numpy as np
from PIL import Image
from slideshift.core.slide import SyntheticSlideReader
from slideshift.ingestion.tissue_mask import segment_tissue
from slideshift.ingestion.quality_control import (
    compute_blur_score,
    detect_pen_marks,
    assess_slide_quality,
)
from slideshift.ingestion.tiling import SlideTiler
from slideshift.ingestion.synthetic_wsi import generate_synthetic_slide_data


def test_tissue_segmentation():
    img_rgb, _, _ = generate_synthetic_slide_data(dimensions=(512, 512), seed=42)
    mask, frac = segment_tissue(img_rgb)
    assert mask.shape == (512, 512)
    assert 0.15 < frac < 0.85
    assert mask.dtype == bool


def test_quality_control():
    # Slide with pen mark and blur
    img_rgb, mask, _ = generate_synthetic_slide_data(
        dimensions=(512, 512),
        add_pen_mark=True,
        add_blur=True,
        seed=42,
    )
    t_mask, t_frac = segment_tissue(img_rgb)
    qc = assess_slide_quality("slide_artifact", img_rgb, t_mask, t_frac)
    assert len(qc.flags) > 0
    assert not qc.passed_qc


def test_slide_tiler():
    slide = SyntheticSlideReader(dimensions=(1024, 1024), mpp=0.50)
    tiler = SlideTiler(target_mpp=0.50, tile_size_px=128, min_tissue_fraction=0.20)
    tiles, qc, thumb, mask = tiler.tile_slide(slide)

    assert len(tiles) > 0
    assert qc.slide_id == slide.slide_id
    assert tiles[0].tile_size_px == 128
    assert tiles[0].target_mpp == 0.50

    # Test patch extraction
    tile_img = tiler.extract_tile_image(slide, tiles[0])
    assert tile_img.size == (128, 128)

