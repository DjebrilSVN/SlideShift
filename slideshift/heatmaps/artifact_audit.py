"""Artifact audit to uncover spurious shortcuts (pen marks, folds, bubbles, edges)."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from slideshift.core.types import QCReport


def audit_artifact_attention(
    attention_scores: np.ndarray,
    coordinates: np.ndarray,
    pen_mask: Optional[np.ndarray],
    fold_mask: Optional[np.ndarray],
    bubble_mask: Optional[np.ndarray],
    slide_dimensions_l0: Tuple[int, int],
    tile_size_l0: int,
) -> Dict[str, any]:
    """Audit how much attention is inappropriately assigned to histological artifacts.

    Computes:
        - Ratio of mean attention on pen mark tiles vs clean tissue tiles
        - Ratio of mean attention on tissue fold tiles vs clean stroma
        - Percentage of top-5% attended tiles that fall on artifacts (shortcut score)
    """
    n_tiles = len(attention_scores)
    if n_tiles == 0:
        return {"shortcut_detected": False, "artifact_share_top5pct": 0.0}

    l0_w, l0_h = slide_dimensions_l0

    def _get_tile_artifact_status(mask: Optional[np.ndarray], threshold: float = 0.15) -> np.ndarray:
        if mask is None or not np.any(mask):
            return np.zeros(n_tiles, dtype=bool)

        mask_h, mask_w = mask.shape
        scale_x = mask_w / float(l0_w)
        scale_y = mask_h / float(l0_h)
        tw = max(1, int(round(tile_size_l0 * scale_x)))
        th = max(1, int(round(tile_size_l0 * scale_y)))

        is_artifact = np.zeros(n_tiles, dtype=bool)
        for i in range(n_tiles):
            x, y = coordinates[i]
            x0 = max(0, min(mask_w - 1, int(round(x * scale_x))))
            y0 = max(0, min(mask_h - 1, int(round(y * scale_y))))
            patch = mask[y0:min(mask_h, y0 + th), x0:min(mask_w, x0 + tw)]
            if patch.size > 0 and np.mean(patch > 0) >= threshold:
                is_artifact[i] = True
        return is_artifact

    is_pen = _get_tile_artifact_status(pen_mask)
    is_fold = _get_tile_artifact_status(fold_mask)
    is_bubble = _get_tile_artifact_status(bubble_mask)
    is_any_artifact = is_pen | is_fold | is_bubble

    mean_all = float(np.mean(attention_scores))
    mean_clean = float(np.mean(attention_scores[~is_any_artifact])) if np.any(~is_any_artifact) else mean_all

    pen_ratio = float(np.mean(attention_scores[is_pen]) / max(1e-7, mean_clean)) if np.any(is_pen) else 0.0
    fold_ratio = float(np.mean(attention_scores[is_fold]) / max(1e-7, mean_clean)) if np.any(is_fold) else 0.0
    bubble_ratio = float(np.mean(attention_scores[is_bubble]) / max(1e-7, mean_clean)) if np.any(is_bubble) else 0.0

    # Share of top 5% tiles that are artifacts
    sorted_idx = np.argsort(attention_scores)[::-1]
    top_5_k = max(1, int(round(n_tiles * 0.05)))
    top_5_idx = sorted_idx[:top_5_k]
    artifact_share_top5 = float(np.mean(is_any_artifact[top_5_idx]))

    # Shortcut warning if artifacts receive > 2x average attention or > 20% of top-5% tiles
    shortcut_detected = (pen_ratio > 2.0) or (fold_ratio > 2.0) or (artifact_share_top5 > 0.20)

    flags = []
    if pen_ratio > 2.0:
        flags.append(f"HIGH_PEN_MARK_ATTENTION (ratio={pen_ratio:.2f}x)")
    if fold_ratio > 2.0:
        flags.append(f"HIGH_FOLD_ATTENTION (ratio={fold_ratio:.2f}x)")
    if artifact_share_top5 > 0.20:
        flags.append(f"ARTIFACT_DOMINATED_TOP_TILES ({artifact_share_top5*100:.1f}%)")

    return {
        "shortcut_detected": shortcut_detected,
        "pen_attention_ratio": pen_ratio,
        "fold_attention_ratio": fold_ratio,
        "bubble_attention_ratio": bubble_ratio,
        "artifact_share_top5pct": artifact_share_top5,
        "warning_flags": flags,
    }

