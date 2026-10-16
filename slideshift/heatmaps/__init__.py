"""Attention heatmap generation, validation, faithfulness tests, and galleries."""

from slideshift.heatmaps.generator import generate_attention_heatmap
from slideshift.heatmaps.validation import evaluate_attention_against_tumor_mask
from slideshift.heatmaps.attribution import run_deletion_insertion_test
from slideshift.heatmaps.artifact_audit import audit_artifact_attention
from slideshift.heatmaps.gallery import (
    select_honest_gallery_slides,
    render_gallery_panel,
)

__all__ = [
    "generate_attention_heatmap",
    "evaluate_attention_against_tumor_mask",
    "run_deletion_insertion_test",
    "audit_artifact_attention",
    "select_honest_gallery_slides",
    "render_gallery_panel",
]

