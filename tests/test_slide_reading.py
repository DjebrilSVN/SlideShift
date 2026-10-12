"""Tests for slide reading abstraction and physical MPP scaling."""

import pytest
import numpy as np
from PIL import Image
from slideshift.core.slide import SyntheticSlideReader, SlideFactory


def test_synthetic_slide_reader():
    slide = SyntheticSlideReader(
        path="synthetic_test.svs",
        dimensions=(1024, 1024),
        mpp=0.50,
        has_tumor=True,
        seed=42,
    )

    assert slide.dimensions == (1024, 1024)
    assert slide.mpp == 0.50
    assert slide.level_count == 4

    # Test thumbnail
    thumb = slide.get_thumbnail((256, 256))
    assert isinstance(thumb, Image.Image)
    assert max(thumb.size) <= 256

    # Test read_region
    crop = slide.read_region(location=(100, 100), level=0, size=(128, 128))
    assert crop.size == (128, 128)
    assert isinstance(crop, Image.Image)


def test_mpp_downsample_calculation():
    # If slide is 0.25 MPP (40x scanner) and target is 0.50 MPP (20x)
    slide = SyntheticSlideReader(mpp=0.25)
    best_lvl, extra_scale = slide.compute_patch_extraction_scale(target_mpp=0.50)
    # Target downsample is 0.50 / 0.25 = 2.0x
    assert best_lvl == 0  # Since level 1 is 4.0x, best level <= 2.0 is 0
    assert abs(extra_scale - 2.0) < 1e-4

