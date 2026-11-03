"""Experiment 02: MIL aggregator ablation.

For a fixed feature extractor, systematically compares:
  - mean_pool   (baseline)
  - max_pool    (baseline)
  - abmil       (main attention model)
  - clam_sb     (instance-supervised)
  - transmil    (transformer)
  - dtfd_mil    (double-tier distillation)
  - acmil       (multi-branch adaptive)
  - patch_gcn   (spatial graph conv)

Statistical comparison: Wilcoxon signed-rank test vs. abmil baseline.

Outputs:
  - outputs/exp02/aggregator_comparison.csv
  - outputs/exp02/wilcoxon_tests.json
"""

import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from slideshift.mil.dataset import SlideBagDataset
from slideshift.mil.trainer import MILTrainer, wilcoxon_cv_comparison
from slideshift.mil.aggregators import AGGREGATOR_REGISTRY
from slideshift.core.config import MILConfig

OUTPUT_DIR = Path("outputs/exp02")
FEATURE_DIR = Path("outputs/exp01/features_mock")
ARCHITECTURES = ["mean_pool", "max_pool", "abmil", "clam_sb", "transmil", "dtfd_mil", "acmil"]
N_FOLDS = 5
N_SEEDS = 3


def evaluate_aggregator(
    bags: SlideBagDataset,
    arch: str,
    n_folds: int = 5,
    seeds: list = None,
) -> dict:
    if seeds is None:
        seeds = list(range(3))

    all_fold_aucs = []
    for seed in seeds:
        config = MILConfig(architecture=arch, epochs=15)
        trainer = MILTrainer(config=config)
        splits = bags.create_splits(n_splits=n_folds, strategy="patient_stratified")
        for train_ds, val_ds, test_ds in splits:
            try:
                trainer.train_fold(train_ds, val_ds, seed=seed)
                metrics = trainer.evaluate(test_ds if len(test_ds) > 0 else val_ds)
                all_fold_aucs.append(metrics["auroc"])
            except Exception as e:
                print(f"  Fold failed: {e}")

    if not all_fold_aucs:
        return {"mean_auroc": 0.5, "std_auroc": 0.0, "arch": arch, "fold_aucs": []}

    return {
        "arch": arch,
        "mean_auroc": float(np.mean(all_fold_aucs)),
        "std_auroc": float(np.std(all_fold_aucs)),
        "fold_aucs": [float(x) for x in all_fold_aucs],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_DIR.exists():
        print(f"Feature directory {FEATURE_DIR} not found.")
        print("Run experiment 01 first: python experiments/01_foundation_model_comparison.py")
        return

    bags = SlideBagDataset.from_directory(FEATURE_DIR)
    if len(bags) < 4:
        print(f"Not enough slides in {FEATURE_DIR}. Need ≥4, found {len(bags)}.")
        return

    print(f"Loaded {len(bags)} slide bags from {FEATURE_DIR}")
    results = {}

    for arch in ARCHITECTURES:
        print(f"\nEvaluating {arch}...")
        try:
            results[arch] = evaluate_aggregator(bags, arch)
            print(f"  AUROC: {results[arch]['mean_auroc']:.4f} ± {results[arch]['std_auroc']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[arch] = {"error": str(e)}

    # Wilcoxon signed-rank tests vs. ABMIL baseline
    wilcoxon_results = {}
    baseline_aucs = results.get("abmil", {}).get("fold_aucs", [])
    if baseline_aucs:
        for arch in ARCHITECTURES:
            if arch == "abmil":
                continue
            challenger_aucs = results.get(arch, {}).get("fold_aucs", [])
            if challenger_aucs and len(challenger_aucs) == len(baseline_aucs):
                wtest = wilcoxon_cv_comparison(baseline_aucs, challenger_aucs)
                wilcoxon_results[f"abmil_vs_{arch}"] = wtest

    # Save
    with open(OUTPUT_DIR / "aggregator_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUTPUT_DIR / "wilcoxon_tests.json", "w") as f:
        json.dump(wilcoxon_results, f, indent=2)

    # Print table
    print("\n" + "=" * 60)
    print(f"{'Aggregator':<15} {'AUROC Mean':>12} {'AUROC Std':>12}")
    print("-" * 40)
    for arch, res in sorted(results.items(), key=lambda x: -x[1].get("mean_auroc", 0)):
        if "error" not in res:
            print(f"{arch:<15} {res['mean_auroc']:>12.4f} {res['std_auroc']:>12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

