"""Attention heatmap generation and thumbnail overlay with percentile scaling."""

from typing import Optional, Tuple
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from scipy.stats import rankdata


def generate_attention_heatmap(
    thumbnail_rgb: np.ndarray,
    coordinates: np.ndarray,
    attention_scores: np.ndarray,
    slide_dimensions_l0: Tuple[int, int],
    tile_size_l0: int,
    alpha: float = 0.45,
    colormap_name: str = "jet",
    use_percentiles: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate smooth attention heatmap overlaid on slide thumbnail.

    Args:
        thumbnail_rgb: (H, W, 3) uint8 slide thumbnail.
        coordinates: (N, 2) array of Level 0 (x, y) coordinates.
        attention_scores: (N,) raw or normalized attention weights.
        slide_dimensions_l0: (width, height) of slide at Level 0.
        tile_size_l0: Tile size in Level 0 pixels.
        alpha: Blending transparency of the heatmap overlay [0.0, 1.0].
        colormap_name: Matplotlib colormap (default: 'jet' or 'inferno').
        use_percentiles: Convert attention to percentile ranks [0, 1] per slide
                         to prevent a single dominant tile from washing out the visual range.

    Returns:
        (blended_rgb, raw_heat_grid, percentile_scores)
    """
    thumb_h, thumb_w, _ = thumbnail_rgb.shape
    l0_w, l0_h = slide_dimensions_l0
    n_tiles = len(attention_scores)

    if n_tiles == 0:
        return thumbnail_rgb.copy(), np.zeros((thumb_h, thumb_w), dtype=np.float32), np.zeros(0)

    # 1. Percentile ranking
    if use_percentiles:
        percentile_scores = rankdata(attention_scores) / float(n_tiles)
    else:
        min_v, max_v = np.min(attention_scores), np.max(attention_scores)
        percentile_scores = (attention_scores - min_v) / max(1e-7, (max_v - min_v))

    scale_x = thumb_w / float(l0_w)
    scale_y = thumb_h / float(l0_h)

    # Accumulate attention on thumbnail grid
    heat_accum = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    count_accum = np.zeros((thumb_h, thumb_w), dtype=np.float32)

    tile_w_thumb = max(1, int(round(tile_size_l0 * scale_x)))
    tile_h_thumb = max(1, int(round(tile_size_l0 * scale_y)))

    for i in range(n_tiles):
        x0_l0, y0_l0 = coordinates[i]
        tx0 = max(0, min(thumb_w - 1, int(round(x0_l0 * scale_x))))
        ty0 = max(0, min(thumb_h - 1, int(round(y0_l0 * scale_y))))
        tx1 = max(tx0 + 1, min(thumb_w, tx0 + tile_w_thumb))
        ty1 = max(ty0 + 1, min(thumb_h, ty0 + tile_h_thumb))

        score = percentile_scores[i]
        heat_accum[ty0:ty1, tx0:tx1] += score
        count_accum[ty0:ty1, tx0:tx1] += 1.0

    # Average overlapping areas
    valid_mask = count_accum > 0
    heat_grid = np.zeros_like(heat_accum)
    heat_grid[valid_mask] = heat_accum[valid_mask] / count_accum[valid_mask]

    # Map normalized heat to RGBA using colormap
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = cm.get_cmap(colormap_name)
    colored_heatmap = cmap(heat_grid)[..., :3]  # RGB float [0, 1]
    colored_heatmap_uint8 = (colored_heatmap * 255.0).astype(np.uint8)

    # Blend with original thumbnail
    blended = thumbnail_rgb.copy().astype(np.float32)
    alpha_mask = (valid_mask * alpha)[..., np.newaxis]
    blended = blended * (1.0 - alpha_mask) + colored_heatmap_uint8 * alpha_mask
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

    return blended_uint8, heat_grid, percentile_scores
