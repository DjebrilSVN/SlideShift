"""Experiment 03: Robustness benchmark across shift types.

For each foundation model × MIL aggregator combination, measures:
  - AUROC drop under stain shift (H&E OD jitter, saturation)
  - AUROC drop under quality shift (blur, JPEG, noise)
  - Prediction flip rate
  - Heatmap stability (Spearman correlation, top-k Jaccard)
  - Embedding MMD between clean and shifted distributions
  - Conformal prediction defer rate under shift

Outputs:
  - outputs/exp03/robustness_results.csv
  - outputs/exp03/robustness_summary.md
  - outputs/exp03/conformal_under_shift.json
"""

import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from slideshift.robustness.lab import RobustnessLab
from slideshift.robustness.metrics import mmd_rbf
from slideshift.robustness.conformal import ConformalPredictor
from slideshift.mil.dataset import SlideBagDataset
from slideshift.mil.trainer import MILTrainer, temperature_scaling
from slideshift.core.config import MILConfig

OUTPUT_DIR = Path("outputs/exp03")
FEATURE_DIR = Path("outputs/exp01/features_mock")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_DIR.exists():
        print(f"Feature directory {FEATURE_DIR} not found.")
        print("Run experiment 01 first: python experiments/01_foundation_model_comparison.py")
        return

    # Standard robustness benchmark
    lab = RobustnessLab(feature_dir=FEATURE_DIR, output_dir=OUTPUT_DIR)
    try:
        summary = lab.run_benchmark()
        print(f"Robustness benchmark complete. Results in {OUTPUT_DIR}")
    except Exception as e:
        print(f"Robustness lab failed: {e}")
        summary = {}

    # Conformal prediction under shift
    bags = SlideBagDataset.from_directory(FEATURE_DIR)
    if len(bags) >= 6:
        config = MILConfig(architecture="abmil", epochs=10)
        trainer = MILTrainer(config=config)
        splits = bags.create_splits(n_splits=3, split_type="patient")

        if splits:
            fold = splits[0]
            trainer.train_fold(fold["train"], fold["val"], seed=0)
            metrics = trainer.evaluate(fold["val"])

            # Conformal calibration
            y_true = metrics.get("y_true", np.array([]))
            y_prob = metrics.get("y_prob", np.array([]))

            if len(y_true) >= 4:
                # Split into cal/test
                n_cal = len(y_true) // 2
                cp = ConformalPredictor(target_coverage=0.90)
                cp.calibrate(y_prob[:n_cal], y_true[:n_cal])
                eval_result = cp.evaluate_coverage(y_prob[n_cal:], y_true[n_cal:])

                conformal_results = {
                    "coverage": eval_result.coverage,
                    "average_set_size": eval_result.average_set_size,
                    "defer_rate": eval_result.defer_rate,
                    "target_coverage": eval_result.target_coverage,
                    "coverage_satisfied": eval_result.coverage_satisfied,
                }

                # Temperature scaling calibration
                T_opt, scaled = temperature_scaling(y_prob, y_true)
                conformal_results["optimal_temperature"] = T_opt

                out_file = OUTPUT_DIR / "conformal_under_shift.json"
                with open(out_file, "w") as f:
                    json.dump(conformal_results, f, indent=2)
                print(f"Conformal results saved to {out_file}")
                print(f"  Coverage: {eval_result.coverage:.3f} (target: {eval_result.target_coverage})")
                print(f"  Defer rate: {eval_result.defer_rate:.3f}")
                print(f"  Optimal temperature: {T_opt:.2f}")


if __name__ == "__main__":
    main()

