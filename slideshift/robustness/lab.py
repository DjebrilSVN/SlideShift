"""Robustness Lab: Controlled perturbation benchmark runner and reporting."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from slideshift.core.types import SlideBag
from slideshift.core.logging import logger
from slideshift.mil.dataset import SlideBagDataset
from slideshift.mil.trainer import MILTrainer
from slideshift.robustness.perturbations import PERTURBATION_REGISTRY
from slideshift.robustness.metrics import evaluate_shift_degradation
from slideshift.robustness.embedding_shift import evaluate_site_linear_probe, compute_knn_neighborhood_purity
from slideshift.robustness.shortcuts import evaluate_shortcut_baseline


class RobustnessLab:
    """Benchmark suite quantifying model degradation under domain, scanner, and stain shifts."""

    def __init__(
        self,
        trainer: MILTrainer,
        shifts_to_test: Optional[List[str]] = None,
        severity_levels: Optional[List[int]] = None,
    ):
        self.trainer = trainer
        self.shifts_to_test = shifts_to_test or list(PERTURBATION_REGISTRY.keys())
        self.severity_levels = severity_levels or [1, 2, 3]

    def run_benchmark(
        self,
        test_dataset: SlideBagDataset,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Execute the complete robustness experiment matrix."""
        logger.info(f"Running Robustness Lab on {len(test_dataset)} test slides...")

        # 1. Clean in-distribution baseline
        clean_eval = self.trainer.evaluate(test_dataset)
        clean_probs = clean_eval["y_prob"]
        y_true = clean_eval["y_true"]
        clean_auc = clean_eval["auroc"]

        clean_attentions = []
        slide_embeddings = []
        for bag in test_dataset:
            pred = self.trainer.predict_bag(bag)
            clean_attentions.append(pred.attention_scores)
            if pred.slide_embedding is not None:
                slide_embeddings.append(pred.slide_embedding)

        records: List[Dict[str, Any]] = []

        # Baseline entry
        records.append({
            "shift_type": "None (Clean / In-Distribution)",
            "severity": 0,
            "auroc": clean_auc,
            "delta_auroc": 0.0,
            "balanced_accuracy": clean_eval["balanced_accuracy"],
            "delta_balanced_accuracy": 0.0,
            "ece": clean_eval["ece"],
            "delta_ece": 0.0,
            "flip_rate": 0.0,
            "heatmap_rank_correlation": 1.0,
            "heatmap_top10_jaccard": 1.0,
        })

        # 2. Synthetic shift perturbations
        for shift_name in self.shifts_to_test:
            for severity in self.severity_levels:
                logger.info(f"Evaluating shift: {shift_name} (Severity {severity})")

                # Generate perturbed bags
                shifted_probs_list = []
                shifted_attentions = []

                for bag in test_dataset:
                    # In feature-level perturbation simulation:
                    # Stain & quality shifts induce directional drift and variance in feature space
                    drift_scale = severity * 0.15
                    rng = np.random.RandomState(severity * 100 + abs(hash(bag.slide_id)) % 1000)
                    feat_noise = rng.normal(0.0, drift_scale, size=bag.features.shape).astype(np.float32)

                    if "blur" in shift_name:
                        # Blur dampens feature magnitude / texture frequencies
                        pert_features = (bag.features * (1.0 - 0.12 * severity)) + feat_noise * 0.5
                    elif "stain" in shift_name:
                        # Stain shift causes systematic mean shift
                        shift_vector = rng.randn(1, bag.feature_dim).astype(np.float32) * drift_scale
                        pert_features = bag.features + shift_vector + feat_noise * 0.3
                    else:
                        pert_features = bag.features + feat_noise

                    pert_bag = SlideBag(
                        slide_id=f"{bag.slide_id}_{shift_name}_s{severity}",
                        features=pert_features.astype(np.float16),
                        coordinates=bag.coordinates,
                        label=bag.label,
                    )

                    pred = self.trainer.predict_bag(pert_bag)
                    shifted_probs_list.append(pred.probabilities)
                    shifted_attentions.append(pred.attention_scores)

                shifted_probs = np.array(shifted_probs_list)

                deg = evaluate_shift_degradation(
                    y_true=y_true,
                    clean_probs=clean_probs,
                    shifted_probs=shifted_probs,
                    clean_attentions=clean_attentions,
                    shifted_attentions=shifted_attentions,
                )

                records.append({
                    "shift_type": shift_name,
                    "severity": severity,
                    "auroc": deg["shifted_auroc"],
                    "delta_auroc": deg["delta_auroc"],
                    "balanced_accuracy": deg["shifted_balanced_accuracy"],
                    "delta_balanced_accuracy": deg["delta_balanced_accuracy"],
                    "ece": deg["shifted_ece"],
                    "delta_ece": deg["delta_ece"],
                    "flip_rate": deg["flip_rate"],
                    "heatmap_rank_correlation": deg["heatmap_rank_correlation"],
                    "heatmap_top10_jaccard": deg["heatmap_top10_jaccard"],
                })

        df_results = pd.DataFrame(records)

        # 3. Embedding probe & shortcut analysis
        if slide_embeddings:
            slide_embs_np = np.stack(slide_embeddings, axis=0)
            site_probe = evaluate_site_linear_probe(slide_embs_np, test_dataset.sites)
            knn_purity = compute_knn_neighborhood_purity(slide_embs_np, y_true, test_dataset.sites)
            shortcut = evaluate_shortcut_baseline(test_dataset)

            logger.info(f"Site probe AUROC: {site_probe.get('site_probe_auroc', 0.5):.3f}")
            logger.info(f"Entanglement index (site/bio purity): {knn_purity.get('entanglement_index', 1.0):.2f}")
            if shortcut.get("shortcut_detected"):
                logger.warning(shortcut["warning_message"])

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(out_p / "robustness_benchmark.csv", index=False)
            with open(out_p / "robustness_report.md", "w", encoding="utf-8") as f:
                f.write("# SlideShift: Robustness Benchmark Report\n\n")
                try:
                    f.write(df_results.to_markdown(index=False))
                except (ImportError, Exception):
                    headers = list(df_results.columns)
                    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                    for _, row in df_results.iterrows():
                        lines.append("| " + " | ".join(f"{row[h]:.4f}" if isinstance(row[h], (float, np.floating)) else str(row[h]) for h in headers) + " |")
                    f.write("\n".join(lines))
                f.write("\n")

        return df_results
