"""Honest heatmap gallery generation showing best, random, and worst failures."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from slideshift.core.types import HeatmapResult, PredictionOutput


def select_honest_gallery_slides(
    predictions: List[PredictionOutput],
    validation_results: Dict[str, Dict[str, float]],
    n_per_category: int = 3,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Select slides across three unbiased bins: best cases, random cases, worst failures."""
    rng = np.random.RandomState(seed)

    best_candidates = []
    worst_candidates = []
    all_slide_ids = [p.slide_id for p in predictions]

    for p in predictions:
        sid = p.slide_id
        val = validation_results.get(sid, {})
        auroc = val.get("attention_tumor_auroc", 0.5)

        # Correct & high overlap
        if p.true_label is not None and p.predicted_class == p.true_label:
            prob_correct = float(p.probabilities[p.true_label])
            score = prob_correct * (auroc if not np.isnan(auroc) else 0.7)
            best_candidates.append((score, sid))
        elif p.true_label is not None and p.predicted_class != p.true_label:
            # False positive or false negative
            conf_wrong = float(p.probabilities[p.predicted_class])
            worst_candidates.append((conf_wrong, sid))

    best_candidates.sort(key=lambda x: x[0], reverse=True)
    worst_candidates.sort(key=lambda x: x[0], reverse=True)

    best_ids = [sid for _, sid in best_candidates[:n_per_category]]
    worst_ids = [sid for _, sid in worst_candidates[:n_per_category]]

    # Random selection excluding already chosen
    remaining = [sid for sid in all_slide_ids if sid not in best_ids and sid not in worst_ids]
    if len(remaining) <= n_per_category:
        random_ids = remaining
    else:
        chosen_rand = rng.choice(remaining, size=n_per_category, replace=False)
        random_ids = list(chosen_rand)

    return {
        "best_cases": best_ids,
        "random_cases": random_ids,
        "worst_failures": worst_ids,
    }


def render_gallery_panel(
    slide_id: str,
    thumbnail_rgb: np.ndarray,
    heatmap_overlay_rgb: np.ndarray,
    tumor_mask: Optional[np.ndarray],
    prediction_info: str,
    output_path: Union[str, Path],
) -> Path:
    """Render a 3-panel comparison: [Original H&E Thumbnail | Pathologist Ground Truth | Attention Heatmap]."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w, _ = thumbnail_rgb.shape

    # Panel 2: Ground truth mask overlay
    if tumor_mask is not None:
        mask_resized = np.array(Image.fromarray(tumor_mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST))
        gt_panel = thumbnail_rgb.copy()
        gt_panel[mask_resized > 0, 0] = np.clip(gt_panel[mask_resized > 0, 0] * 0.4 + 180, 0, 255)
    else:
        gt_panel = thumbnail_rgb.copy()

    # Create composite: 3 panels horizontally with header
    composite_w = w * 3 + 40
    composite_h = h + 80
    composite = Image.new("RGB", (composite_w, composite_h), (250, 250, 250))

    img_orig = Image.fromarray(thumbnail_rgb)
    img_gt = Image.fromarray(gt_panel)
    img_heat = Image.fromarray(heatmap_overlay_rgb)

    draw = ImageDraw.Draw(composite)
    draw.text((20, 10), f"Slide: {slide_id} | {prediction_info}", fill=(20, 20, 20))
    draw.text((20, 35), "1. Original H&E Thumbnail", fill=(60, 60, 60))
    draw.text((w + 35, 35), "2. Pathologist Tumor Mask", fill=(60, 60, 60))
    draw.text((w * 2 + 50, 35), "3. Attention Heatmap (Percentile)", fill=(60, 60, 60))

    composite.paste(img_orig, (15, 60))
    composite.paste(img_gt, (w + 30, 60))
    composite.paste(img_heat, (w * 2 + 45, 60))

    composite.save(output_path)
    return output_path

