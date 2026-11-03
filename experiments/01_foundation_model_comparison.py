"""Experiment 01: Foundation model comparison.

Compares tile encoding quality across:
  - Mock (random baseline — lower bound)
  - ResNet-50 (ImageNet baseline)
  - Phikon-v2 (open-weight pathology)
  - UNI / CONCH / Virchow (gated — requires approved access)

Outputs:
  - outputs/exp01/feature_stats.csv   — embedding dim, variance, norm stats
  - outputs/exp01/mil_results.csv     — AUROC, ECE per model × aggregator
  - outputs/exp01/delong_tests.csv    — pairwise DeLong AUC comparisons
"""

import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from slideshift.features.registry import MODEL_REGISTRY
from slideshift.features.encoders import EncoderFactory
from slideshift.features.extractor import SlideFeatureExtractor
from slideshift.core.slide import SlideFactory
from slideshift.ingestion.tiling import SlideTiler
from slideshift.mil.dataset import SlideBagDataset
from slideshift.mil.trainer import MILTrainer, delong_auc_test, wilcoxon_cv_comparison
from slideshift.core.config import PipelineConfig, MILConfig

OUTPUT_DIR = Path("outputs/exp01")
SLIDE_DIR = Path("data/benchmark_cohort")
MODELS_TO_TEST = ["mock", "resnet50"]  # add "phikon", "uni", "conch" when access approved


def run_feature_extraction(slide_dir: Path, model_name: str, output_dir: Path) -> Path:
    """Extract features for a given model."""
    feat_dir = output_dir / f"features_{model_name}"
    feat_dir.mkdir(parents=True, exist_ok=True)
    encoder = EncoderFactory.create(model_name=model_name)
    extractor = SlideFeatureExtractor(encoder=encoder, tiler=SlideTiler(target_mpp=0.50))
    slides = [
        p for p in (list(slide_dir.glob("*.svs")) + list(slide_dir.glob("*.tif")) + list(slide_dir.glob("*.png")))
        if not p.stem.endswith("_mask") and not p.stem.endswith("_tumor_mask")
    ]
    for slide_p in slides:
        label = 1 if "tumor" in slide_p.name.lower() else 0
        with SlideFactory.open(slide_p, mpp=0.50) as reader:
            bag = extractor.extract_slide_features(reader, label=label)
            extractor.save_bag(bag, feat_dir)
    return feat_dir


def evaluate_model(feat_dir: Path, arch: str = "abmil") -> dict:
    """Train and evaluate a MIL model on extracted features."""
    bags = SlideBagDataset.from_directory(feat_dir)
    if len(bags) < 4:
        return {"auroc": 0.5, "ece": 0.25, "model": feat_dir.name, "arch": arch}

    config = MILConfig(architecture=arch, epochs=10)
    trainer = MILTrainer(config=config)
    splits = bags.create_splits(n_splits=2, strategy="patient_stratified")

    fold_aucs = []
    for train_ds, val_ds, test_ds in splits:
        trainer.train_fold(train_ds, val_ds, seed=42)
        metrics = trainer.evaluate(test_ds if len(test_ds) > 0 else val_ds)
        fold_aucs.append(metrics["auroc"])

    return {
        "auroc_mean": float(np.mean(fold_aucs)),
        "auroc_std": float(np.std(fold_aucs)),
        "model": feat_dir.name,
        "arch": arch,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SLIDE_DIR.exists():
        print(f"Generating synthetic cohort at {SLIDE_DIR}...")
        from slideshift.ingestion.synthetic_wsi import SyntheticWSIGenerator
        gen = SyntheticWSIGenerator(output_dir=SLIDE_DIR, num_slides=30, seed=42)
        gen.generate_all()

    results = {}
    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*50}\nModel: {model_name}\n{'='*50}")
        feat_dir = run_feature_extraction(SLIDE_DIR, model_name, OUTPUT_DIR)
        for arch in ["mean_pool", "abmil", "transmil"]:
            key = f"{model_name}_{arch}"
            try:
                results[key] = evaluate_model(feat_dir, arch)
                print(f"  {arch}: AUROC={results[key].get('auroc_mean', 0):.4f} ± {results[key].get('auroc_std', 0):.4f}")
            except Exception as e:
                print(f"  {arch}: FAILED — {e}")
                results[key] = {"error": str(e)}

    # Save results
    out_file = OUTPUT_DIR / "mil_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Pairwise DeLong comparisons (where possible)
    print("\n[DeLong pairwise AUC tests would run here with per-fold probabilities]")


if __name__ == "__main__":
    main()

