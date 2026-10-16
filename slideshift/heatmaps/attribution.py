"""Faithfulness testing via Deletion and Insertion tests and alternative attribution methods."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from slideshift.core.types import SlideBag, PredictionOutput


def run_deletion_insertion_test(
    predict_fn: Callable[[SlideBag], PredictionOutput],
    bag: SlideBag,
    attention_scores: np.ndarray,
    target_class: int = 1,
    fractions: Optional[List[float]] = None,
    n_random_trials: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform deletion and insertion tests on slide bag.

    Deletion:
        Removes the top-k attended tiles. A faithful attention mechanism will cause
        the model's tumor probability to plummet rapidly compared to random tile removal.
    Insertion:
        Starts from an empty bag and adds back only the top-k attended tiles.
        A faithful map should recover the full prediction quickly.

    Returns:
        Dictionary containing deletion curves, insertion curves, random deletion baseline,
        and area-under-curve metrics (AUDC, AUIC).
    """
    fractions = fractions or [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
    n_tiles = bag.num_tiles
    rng = np.random.RandomState(seed)

    sorted_indices = np.argsort(attention_scores)[::-1]  # Descending order

    # Base prediction
    base_pred = predict_fn(bag)
    base_prob = float(base_pred.probabilities[target_class])

    deletion_probs: List[float] = []
    insertion_probs: List[float] = []
    random_deletion_probs: List[float] = []

    for frac in fractions:
        k = int(round(n_tiles * frac))

        # --- Deletion ---
        if k == 0:
            del_prob = base_prob
        elif k >= n_tiles:
            del_prob = 0.5
        else:
            remaining_idx = sorted_indices[k:]
            del_bag = SlideBag(
                slide_id=f"{bag.slide_id}_del_{frac}",
                features=bag.features[remaining_idx],
                coordinates=bag.coordinates[remaining_idx],
                label=bag.label,
            )
            del_out = predict_fn(del_bag)
            del_prob = float(del_out.probabilities[target_class])
        deletion_probs.append(del_prob)

        # --- Insertion ---
        if k == 0:
            ins_prob = 0.5
        else:
            top_k_idx = sorted_indices[:k]
            ins_bag = SlideBag(
                slide_id=f"{bag.slide_id}_ins_{frac}",
                features=bag.features[top_k_idx],
                coordinates=bag.coordinates[top_k_idx],
                label=bag.label,
            )
            ins_out = predict_fn(ins_bag)
            ins_prob = float(ins_out.probabilities[target_class])
        insertion_probs.append(ins_prob)

        # --- Random Deletion Baseline ---
        if k == 0:
            rnd_del_prob = base_prob
        elif k >= n_tiles:
            rnd_del_prob = 0.5
        else:
            trial_probs = []
            for _ in range(n_random_trials):
                rnd_drop = rng.choice(n_tiles, size=k, replace=False)
                keep_mask = np.ones(n_tiles, dtype=bool)
                keep_mask[rnd_drop] = False
                rnd_bag = SlideBag(
                    slide_id=f"{bag.slide_id}_rnd_{frac}",
                    features=bag.features[keep_mask],
                    coordinates=bag.coordinates[keep_mask],
                    label=bag.label,
                )
                rnd_out = predict_fn(rnd_bag)
                trial_probs.append(float(rnd_out.probabilities[target_class]))
            rnd_del_prob = float(np.mean(trial_probs))
        random_deletion_probs.append(rnd_del_prob)

    # Compute trapezoidal Area Under Deletion Curve (lower is better for deletion)
    # and Area Under Insertion Curve (higher is better for insertion)
    audc = float(np.trapz(deletion_probs, fractions))
    auic = float(np.trapz(insertion_probs, fractions))
    random_audc = float(np.trapz(random_deletion_probs, fractions))

    return {
        "fractions": fractions,
        "deletion_curve": deletion_probs,
        "insertion_curve": insertion_probs,
        "random_deletion_curve": random_deletion_probs,
        "audc": audc,
        "auic": auic,
        "random_audc": random_audc,
        "faithfulness_gap": random_audc - audc,  # Positive indicates attention drops faster than random
    }

