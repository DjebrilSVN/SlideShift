"""Embedding-level robustness: Site linear probe and k-NN site vs biology purity."""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from slideshift.core.types import SlideBag


def evaluate_site_linear_probe(
    slide_embeddings: np.ndarray,
    site_labels: List[str],
    cv_splits: int = 3,
    seed: int = 42,
) -> Dict[str, float]:
    """Train a linear probe to predict scanner / site origin directly from slide embeddings.
    
    If the linear probe achieves high accuracy, the foundation model's representation
    is strongly confounded by hospital/scanner artifacts rather than pure histology.
    """
    unique_sites, site_idx = np.unique(site_labels, return_inverse=True)
    if len(unique_sites) < 2:
        return {"site_probe_accuracy": 0.0, "site_probe_auroc": 0.5, "unique_sites": 1}

    clf = LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)
    clf.fit(slide_embeddings, site_idx)
    preds = clf.predict(slide_embeddings)
    probs = clf.predict_proba(slide_embeddings)

    acc = float(accuracy_score(site_idx, preds))
    try:
        if len(unique_sites) == 2:
            auc = float(roc_auc_score(site_idx, probs[:, 1]))
        else:
            auc = float(roc_auc_score(site_idx, probs, multi_class="ovr"))
    except Exception:
        auc = acc

    return {
        "site_probe_accuracy": acc,
        "site_probe_auroc": auc,
        "unique_sites": len(unique_sites),
    }


def compute_knn_neighborhood_purity(
    embeddings: np.ndarray,
    class_labels: np.ndarray,
    site_labels: List[str],
    k: int = 5,
) -> Dict[str, float]:
    """Calculate k-NN neighborhood purity for biology (class) vs domain (site).
    
    A robust foundation model should have high biology purity and low site purity
    (nearest neighbors in feature space should have the same tumor class, NOT the same hospital).
    """
    n = len(embeddings)
    if n <= k:
        return {"biological_purity": 1.0, "site_purity": 1.0, "entanglement_index": 1.0}

    unique_sites, site_idx = np.unique(site_labels, return_inverse=True)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    indices = nbrs.kneighbors(embeddings, return_distance=False)

    bio_purities = []
    site_purities = []

    for i in range(n):
        # Exclude self (index 0)
        neighbors = indices[i, 1:]

        # Fraction sharing the same class label
        bio_same = np.mean(class_labels[neighbors] == class_labels[i])
        bio_purities.append(bio_same)

        # Fraction sharing the same hospital site
        site_same = np.mean(site_idx[neighbors] == site_idx[i])
        site_purities.append(site_same)

    mean_bio = float(np.mean(bio_purities))
    mean_site = float(np.mean(site_purities))

    # Entanglement index: ratio of site purity to biological purity (>1.0 indicates site dominates biology)
    entanglement = float(mean_site / max(1e-5, mean_bio))

    return {
        "biological_purity": mean_bio,
        "site_purity": mean_site,
        "entanglement_index": entanglement,
    }

