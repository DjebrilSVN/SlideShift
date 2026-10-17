"""Foundation model ensembling across multiple encoder backbones."""

from typing import List, Optional
import numpy as np
from slideshift.core.types import PredictionOutput


def ensemble_predictions(
    predictions_per_model: List[List[PredictionOutput]],
    weights: Optional[List[float]] = None,
) -> List[PredictionOutput]:
    """Average prediction probabilities across multiple foundation model pipelines."""
    n_models = len(predictions_per_model)
    if n_models == 0:
        return []
    if n_models == 1:
        return predictions_per_model[0]

    n_slides = len(predictions_per_model[0])
    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        norm_w = sum(weights)
        weights = [w / norm_w for w in weights]

    ensembled: List[PredictionOutput] = []

    for i in range(n_slides):
        slide_id = predictions_per_model[0][i].slide_id
        true_label = predictions_per_model[0][i].true_label

        avg_prob = np.zeros_like(predictions_per_model[0][i].probabilities)
        for m_idx in range(n_models):
            pred = predictions_per_model[m_idx][i]
            avg_prob += weights[m_idx] * pred.probabilities

        pred_class = int(np.argmax(avg_prob))
        logits = np.log(np.clip(avg_prob, 1e-7, 1.0))

        ensembled.append(
            PredictionOutput(
                slide_id=slide_id,
                logits=logits,
                probabilities=avg_prob,
                predicted_class=pred_class,
                true_label=true_label,
                metadata={"ensemble_size": n_models},
            )
        )

    return ensembled
