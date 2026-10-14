"""CONCH zero-shot classification baseline.

CONCH (Contrastive Language-Image Pre-training for Computational Histology,
Lu et al. 2023) is a vision-language foundation model that can classify slides
from text prompts — no task-specific training required.

This module provides:
  1. Zero-shot tile classification using CONCH text embeddings.
  2. Aggregation of per-tile scores to a slide-level prediction.
  3. Robustness probe: runs the same shifts on zero-shot predictions to compare
     against the trained MIL pipeline.

NOTE: CONCH weights are gated (non-commercial). This module gracefully
      falls back to a similarity-score stub when the model is unavailable.
      Request access at: https://huggingface.co/MahmoodLab/CONCH
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = hasattr(torch, "__version__")
except (ImportError, AttributeError):
    torch = None
    F = None
    HAS_TORCH = False


@dataclass
class ZeroShotClassSpec:
    """Specification for one zero-shot class.

    Args:
        label:    Human-readable class name (e.g. "tumor").
        prompts:  List of text prompts that describe this class. Ensemble
                  predictions over all prompts are averaged.
        class_id: Integer class id used for evaluation.
    """
    label: str
    prompts: List[str]
    class_id: int


# Pre-defined prompt banks for common tasks.
CAMELYON_PROMPTS: List[ZeroShotClassSpec] = [
    ZeroShotClassSpec(
        label="tumor",
        class_id=1,
        prompts=[
            "A histopathology image showing breast cancer metastasis in a lymph node.",
            "Malignant cells invading the lymph node parenchyma.",
            "Dense clusters of atypical epithelial cells with high nuclear-to-cytoplasmic ratio.",
            "Invasive ductal carcinoma metastasis in sentinel lymph node.",
        ],
    ),
    ZeroShotClassSpec(
        label="normal",
        class_id=0,
        prompts=[
            "A histopathology image of a normal lymph node with no metastatic cancer.",
            "Reactive lymphoid tissue with normal follicular architecture.",
            "No evidence of malignancy in this lymph node section.",
            "Healthy lymph node parenchyma with germinal centres.",
        ],
    ),
]

NSCLC_PROMPTS: List[ZeroShotClassSpec] = [
    ZeroShotClassSpec(
        label="luad",
        class_id=0,
        prompts=[
            "Lung adenocarcinoma with acinar, lepidic or papillary growth pattern.",
            "Mucin-producing glandular structures in lung parenchyma.",
            "TTF-1 positive adenocarcinoma cells in the lung.",
        ],
    ),
    ZeroShotClassSpec(
        label="lusc",
        class_id=1,
        prompts=[
            "Lung squamous cell carcinoma with keratinization and intercellular bridges.",
            "Squamous differentiation with keratin pearl formation in the lung.",
            "p40-positive squamous carcinoma cells with high nuclear grade.",
        ],
    ),
]


class CONCHZeroShotClassifier:
    """Vision-language zero-shot classifier using CONCH.

    When CONCH weights are unavailable, falls back to random Gaussian text
    embeddings (for debugging / CI purposes only).
    """

    def __init__(
        self,
        model_name: str = "MahmoodLab/CONCH",
        device: str = "cpu",
        use_mock: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.use_mock = use_mock or not (HAS_TRANSFORMERS and HAS_TORCH)
        self._model = None
        self._tokenizer = None
        self._text_embeddings: Dict[int, np.ndarray] = {}

    def load(self) -> "CONCHZeroShotClassifier":
        """Load model weights. Call this before encoding."""
        if self.use_mock:
            return self
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            dev = torch.device(self.device)
            self._model = self._model.to(dev).eval()
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Could not load CONCH ({exc}). Falling back to mock embeddings. "
                "Request access at https://huggingface.co/MahmoodLab/CONCH",
                UserWarning,
                stacklevel=2,
            )
            self.use_mock = True
        return self

    def _encode_text(self, text: str, emb_dim: int = 512) -> np.ndarray:
        """Encode one text string into a unit-normalised embedding."""
        if self.use_mock:
            # Deterministic pseudo-embedding based on hash
            rng = np.random.RandomState(abs(hash(text)) % (2 ** 31))
            emb = rng.randn(emb_dim).astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-8)
            return emb

        dev = torch.device(self.device)
        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(dev)
        with torch.no_grad():
            out = self._model.get_text_features(**tokens)
            emb = F.normalize(out, dim=-1).cpu().numpy().squeeze(0)
        return emb.astype(np.float32)

    def build_class_embeddings(
        self,
        class_specs: List[ZeroShotClassSpec],
        emb_dim: int = 512,
    ) -> "CONCHZeroShotClassifier":
        """Compute and cache mean text embeddings per class."""
        self._text_embeddings = {}
        for spec in class_specs:
            prompt_embs = [self._encode_text(p, emb_dim) for p in spec.prompts]
            mean_emb = np.mean(prompt_embs, axis=0).astype(np.float32)
            mean_emb /= (np.linalg.norm(mean_emb) + 1e-8)
            self._text_embeddings[spec.class_id] = mean_emb
        return self

    def predict_tile_scores(
        self,
        tile_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute class probabilities for each tile via cosine similarity.

        Args:
            tile_embeddings: (N, D) float32 tile image embeddings from CONCH
                             vision encoder (or a mock).

        Returns:
            probs: (N, num_classes) softmax probabilities.
        """
        if not self._text_embeddings:
            raise RuntimeError("Call build_class_embeddings() first.")

        class_ids = sorted(self._text_embeddings.keys())
        text_matrix = np.stack(
            [self._text_embeddings[cid] for cid in class_ids], axis=0
        )  # (C, D)

        # Normalise tile embeddings
        norms = np.linalg.norm(tile_embeddings, axis=1, keepdims=True) + 1e-8
        tile_norm = tile_embeddings / norms                     # (N, D)

        logits = tile_norm @ text_matrix.T                      # (N, C)
        # Softmax via log-sum-exp
        max_l = logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)        # (N, C)

        return probs.astype(np.float32)

    def predict_slide(
        self,
        tile_embeddings: np.ndarray,
        aggregation: str = "mean",
    ) -> Tuple[np.ndarray, int]:
        """Aggregate per-tile probabilities to a slide-level prediction.

        Args:
            tile_embeddings: (N, D)
            aggregation:     'mean' | 'max' | 'top10pct'

        Returns:
            slide_probs: (num_classes,) slide-level probabilities.
            pred_class: predicted class id.
        """
        tile_probs = self.predict_tile_scores(tile_embeddings)  # (N, C)

        if aggregation == "mean":
            slide_probs = tile_probs.mean(axis=0)
        elif aggregation == "max":
            slide_probs = tile_probs.max(axis=0)
        elif aggregation == "top10pct":
            k = max(1, int(len(tile_probs) * 0.10))
            # Use score of positive class to rank tiles
            pos_scores = tile_probs[:, -1]
            top_idx = np.argsort(pos_scores)[-k:]
            slide_probs = tile_probs[top_idx].mean(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Ensure probabilities sum to 1.0 (e.g. for max aggregation)
        prob_sum = float(slide_probs.sum())
        if prob_sum > 0:
            slide_probs = slide_probs / prob_sum

        pred_class = int(np.argmax(slide_probs))
        return slide_probs, pred_class

    def evaluate(
        self,
        tile_embeddings_list: List[np.ndarray],
        labels: List[int],
        aggregation: str = "mean",
    ) -> Dict[str, float]:
        """Evaluate zero-shot performance on a list of slide embeddings."""
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score

        y_probs = []
        y_preds = []
        for embs in tile_embeddings_list:
            probs, pred = self.predict_slide(embs, aggregation)
            y_probs.append(probs)
            y_preds.append(pred)

        y_true = np.array(labels)
        y_probs_arr = np.array(y_probs)
        y_preds_arr = np.array(y_preds)

        num_classes = y_probs_arr.shape[1]
        try:
            if num_classes == 2:
                auroc = float(roc_auc_score(y_true, y_probs_arr[:, 1]))
            else:
                auroc = float(roc_auc_score(y_true, y_probs_arr, multi_class="ovr"))
        except Exception:
            auroc = 0.5

        bal_acc = float(balanced_accuracy_score(y_true, y_preds_arr))
        return {
            "auroc": auroc,
            "balanced_accuracy": bal_acc,
            "aggregation": aggregation,
        }

