"""Validation of attention heatmaps against pathologist tumor annotations."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_attention_against_tumor_mask(
    attention_scores: np.ndarray,
    coordinates: np.ndarray,
    tumor_mask: np.ndarray,
    slide_dimensions_l0: Tuple[int, int],
    tile_size_l0: int,
    tumor_threshold_in_tile: float = 0.20,
) -> Dict[str, float]:
    """Validate attention map against pixel-level ground truth tumor annotations.
    
    Args:
        attention_scores: (N,) attention values for each tile.
        coordinates: (N, 2) Level 0 coordinates.
        tumor_mask: 2D boolean or uint8 mask of ground-truth tumor.
        slide_dimensions_l0: (w, h) of slide at Level 0.
        tile_size_l0: tile size at Level 0.
        tumor_threshold_in_tile: Fraction of tile area that must be tumor to label tile as positive.

    Returns:
        Dictionary with:
            - 'attention_tumor_auroc': AUROC of tile attention vs ground-truth tumor
            - 'top_5pct_tumor_precision': % of top 5% attended tiles inside tumor
            - 'top_10pct_tumor_precision': % of top 10% attended tiles inside tumor
            - 'top_20pct_tumor_precision': % of top 20% attended tiles inside tumor
            - 'num_tumor_tiles': Total number of ground-truth tumor tiles
            - 'num_total_tiles': Total tiles evaluated
    """
    n_tiles = len(attention_scores)
    if n_tiles == 0:
        return {
            "attention_tumor_auroc": 0.5,
            "top_5pct_tumor_precision": 0.0,
            "top_10pct_tumor_precision": 0.0,
            "top_20pct_tumor_precision": 0.0,
            "num_tumor_tiles": 0,
            "num_total_tiles": 0,
        }

    mask_h, mask_w = tumor_mask.shape
    l0_w, l0_h = slide_dimensions_l0
    scale_x = mask_w / float(l0_w)
    scale_y = mask_h / float(l0_h)

    # Determine ground-truth tumor label for each tile
    tile_has_tumor = np.zeros(n_tiles, dtype=bool)
    tile_tumor_fraction = np.zeros(n_tiles, dtype=np.float32)

    tile_w_m = max(1, int(round(tile_size_l0 * scale_x)))
    tile_h_m = max(1, int(round(tile_size_l0 * scale_y)))

    for i in range(n_tiles):
        x0_l0, y0_l0 = coordinates[i]
        mx0 = max(0, min(mask_w - 1, int(round(x0_l0 * scale_x))))
        my0 = max(0, min(mask_h - 1, int(round(y0_l0 * scale_y))))
        mx1 = max(mx0 + 1, min(mask_w, mx0 + tile_w_m))
        my1 = max(my0 + 1, min(mask_h, my0 + tile_h_m))

        tile_patch = tumor_mask[my0:my1, mx0:mx1]
        if tile_patch.size > 0:
            frac = float(np.mean(tile_patch > 0))
            tile_tumor_fraction[i] = frac
            tile_has_tumor[i] = frac >= tumor_threshold_in_tile

    num_tumor_tiles = int(np.sum(tile_has_tumor))

    # AUROC of tile attention predicting tumor presence
    if 0 < num_tumor_tiles < n_tiles:
        try:
            auroc = float(roc_auc_score(tile_has_tumor.astype(int), attention_scores))
        except Exception:
            auroc = 0.5
    else:
        # Either all tiles or no tiles have tumor
        auroc = float("nan")

    # Top-k precision
    sorted_indices = np.argsort(attention_scores)[::-1]

    def _precision_at_k(k_pct: float) -> float:
        k = max(1, int(round(n_tiles * k_pct)))
        top_k_idx = sorted_indices[:k]
        return float(np.mean(tile_has_tumor[top_k_idx]))

    return {
        "attention_tumor_auroc": auroc,
        "top_5pct_tumor_precision": _precision_at_k(0.05),
        "top_10pct_tumor_precision": _precision_at_k(0.10),
        "top_20pct_tumor_precision": _precision_at_k(0.20),
        "num_tumor_tiles": num_tumor_tiles,
        "num_total_tiles": n_tiles,
    }

