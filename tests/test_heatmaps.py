"""Tests for attention heatmaps, ground-truth validation, deletion/insertion tests, and artifact audit."""

import pytest
import numpy as np
from slideshift.heatmaps.generator import generate_attention_heatmap
from slideshift.heatmaps.validation import evaluate_attention_against_tumor_mask
from slideshift.heatmaps.attribution import run_deletion_insertion_test
from slideshift.heatmaps.artifact_audit import audit_artifact_attention
from slideshift.core.types import SlideBag, PredictionOutput


def test_heatmap_generation():
    thumb = np.full((128, 128, 3), 240, dtype=np.uint8)
    coords = np.array([[0, 0], [64, 0], [0, 64], [64, 64]])
    attns = np.array([0.1, 0.4, 0.8, 0.2])

    blended, heat_grid, pct = generate_attention_heatmap(
        thumbnail_rgb=thumb,
        coordinates=coords,
        attention_scores=attns,
        slide_dimensions_l0=(128, 128),
        tile_size_l0=64,
    )

    assert blended.shape == (128, 128, 3)
    assert heat_grid.shape == (128, 128)
    assert len(pct) == 4
    assert np.max(pct) == 1.0


def test_ground_truth_validation():
    tumor_mask = np.zeros((128, 128), dtype=bool)
    tumor_mask[:64, :64] = True  # Top-left quadrant has tumor

    coords = np.array([[0, 0], [64, 0], [0, 64], [64, 64]])
    # Accurate attention: tile 0 has high attention
    attns = np.array([0.95, 0.10, 0.05, 0.02])

    metrics = evaluate_attention_against_tumor_mask(
        attention_scores=attns,
        coordinates=coords,
        tumor_mask=tumor_mask,
        slide_dimensions_l0=(128, 128),
        tile_size_l0=64,
    )

    assert metrics["num_tumor_tiles"] == 1
    assert metrics["attention_tumor_auroc"] == 1.0
    assert metrics["top_20pct_tumor_precision"] == 1.0


def test_artifact_audit():
    pen_mask = np.zeros((128, 128), dtype=bool)
    pen_mask[:32, :32] = True  # pen mark on top-left tile

    coords = np.array([[0, 0], [64, 0], [0, 64], [64, 64]])
    # Model mistakenly assigns huge attention to pen mark tile (shortcut!)
    attns = np.array([0.90, 0.05, 0.03, 0.02])

    audit = audit_artifact_attention(
        attention_scores=attns,
        coordinates=coords,
        pen_mask=pen_mask,
        fold_mask=None,
        bubble_mask=None,
        slide_dimensions_l0=(128, 128),
        tile_size_l0=64,
    )

    assert audit["shortcut_detected"] is True
    assert audit["pen_attention_ratio"] > 2.0

