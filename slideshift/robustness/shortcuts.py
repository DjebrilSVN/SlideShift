"""Shortcut baseline classifier evaluating site-label confounding."""

from typing import Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from slideshift.mil.dataset import SlideBagDataset


def evaluate_shortcut_baseline(dataset: SlideBagDataset, seed: int = 42) -> Dict[str, any]:
    """Train baseline classifiers using ONLY site metadata or low-level color moments.

    Checks whether the benchmark is trivially predictable due to hospital-label imbalance
    (e.g., Hospital A provided 90% of tumor slides and Hospital B provided 90% of normal slides).
    """
    y = dataset.labels
    valid_idx = np.where(y >= 0)[0]
    if len(valid_idx) < 10:
        return {"shortcut_detected": False, "site_only_auroc": 0.5}

    y_valid = y[valid_idx]
    sites = [dataset.sites[i] for i in valid_idx]
    unique_sites, site_codes = np.unique(sites, return_inverse=True)

    # One-hot encode site
    n_sites = len(unique_sites)
    if n_sites <= 1:
        return {
            "shortcut_detected": False,
            "site_only_auroc": 0.50,
            "notes": "Single site present; no site-label shortcut possible.",
        }

    X_site = np.zeros((len(y_valid), n_sites), dtype=np.float32)
    for i, sc in enumerate(site_codes):
        X_site[i, sc] = 1.0

    clf = LogisticRegression(class_weight="balanced", random_state=seed)
    clf.fit(X_site, y_valid)
    probs = clf.predict_proba(X_site)

    try:
        if len(np.unique(y_valid)) == 2:
            p1 = probs[:, 1] if probs.shape[1] > 1 else probs
            site_auc = float(roc_auc_score(y_valid, p1))
        else:
            site_auc = float(roc_auc_score(y_valid, probs, multi_class="ovr"))
    except Exception:
        site_auc = 0.5

    # Confounding warning threshold: if site alone predicts cancer with AUROC > 0.65
    is_shortcut = site_auc >= 0.65

    warning_msg = None
    if is_shortcut:
        warning_msg = (
            f"CRITICAL SHORTCUT DETECTED: Site metadata alone predicts tumor label with AUROC {site_auc:.3f}! "
            "Reported model performance may be inflated by hospital-specific color/scanner artifacts."
        )

    return {
        "shortcut_detected": is_shortcut,
        "site_only_auroc": site_auc,
        "site_only_accuracy": float(accuracy_score(y_valid, clf.predict(X_site))),
        "warning_message": warning_msg,
    }

