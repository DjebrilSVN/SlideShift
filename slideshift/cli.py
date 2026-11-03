"""SlideShift command line interface."""

import os
import sys
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table

from slideshift.core.config import PipelineConfig
from slideshift.core.logging import logger
from slideshift.core.slide import SlideFactory
from slideshift.ingestion.tiling import SlideTiler
from slideshift.ingestion.synthetic_wsi import save_synthetic_slide
from slideshift.features.encoders import EncoderFactory
from slideshift.features.extractor import SlideFeatureExtractor
from slideshift.features.registry import MODEL_REGISTRY
from slideshift.mil.dataset import SlideBagDataset
from slideshift.mil.trainer import MILTrainer
from slideshift.robustness.lab import RobustnessLab
from slideshift.monitor.extractor import SlideSignatureExtractor
from slideshift.monitor.detector import SlideShiftDetector

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """SlideShift: Pathology whole-slide pipeline with foundation models, MIL, robustness benchmarking, and shift monitoring."""
    pass


@main.command()
@click.option("--output-dir", default="data/synthetic_cohort", help="Directory to save synthetic slides")
@click.option("--num-slides", default=10, help="Number of synthetic slides to generate")
def synthetic_data(output_dir: str, num_slides: int):
    """Generate synthetic WSI cohort for end-to-end testing and CI."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold green]Generating {num_slides} synthetic whole-slide images in {output_dir}...[/bold green]")

    for i in range(num_slides):
        has_tumor = (i % 2 == 1)
        site = "site_B" if i >= (num_slides - 2) else "site_A"
        slide_name = f"slide_{i:03d}_{site}_{'tumor' if has_tumor else 'normal'}.png"
        save_synthetic_slide(
            output_path=out_p / slide_name,
            dimensions=(1024, 1024),
            has_tumor=has_tumor,
            site_id=site,
            add_pen_mark=(i == 0),
            seed=42 + i,
        )
    console.print(f"[bold green]Successfully generated {num_slides} synthetic slides.[/bold green]")


@main.command()
@click.option("--slide-dir", required=True, help="Directory containing whole slide images")
@click.option("--output-dir", default="outputs/features", help="Directory to save extracted feature bags")
@click.option("--model", default="resnet50", help="Foundation model name (resnet50, phikon, uni, conch, virchow, mock)")
@click.option("--target-mpp", default=0.50, help="Target physical resolution in microns/pixel")
@click.option("--batch-size", default=64, help="Tile batch size for feature extractor")
def extract(slide_dir: str, output_dir: str, model: str, target_mpp: float, batch_size: int):
    """Extract foundation model tile embeddings from whole-slide images."""
    s_dir = Path(slide_dir)
    slides = [
        p for p in (list(s_dir.glob("*.svs")) + list(s_dir.glob("*.tif")) + list(s_dir.glob("*.tiff")) + list(s_dir.glob("*.png")))
        if not p.stem.endswith("_mask") and not p.stem.endswith("_tumor_mask")
    ]
    if not slides:
        console.print(f"[bold red]No slide files found in {slide_dir}[/bold red]")
        sys.exit(1)

    console.print(f"[bold cyan]Initializing foundation model: {model}...[/bold cyan]")
    encoder = EncoderFactory.create(model_name=model)
    tiler = SlideTiler(target_mpp=target_mpp)
    extractor = SlideFeatureExtractor(encoder=encoder, tiler=tiler, batch_size=batch_size)

    console.print(f"[bold green]Found {len(slides)} slides. Starting feature extraction...[/bold green]")
    for slide_p in slides:
        try:
            with SlideFactory.open(slide_p, mpp=target_mpp) as reader:
                label = 1 if "tumor" in slide_p.name.lower() else (0 if "normal" in slide_p.name.lower() else None)
                bag = extractor.extract_slide_features(reader, label=label)
                extractor.save_bag(bag, output_dir)
                console.print(f"  Extracted: {bag.slide_id} ({bag.num_tiles} tiles, dim: {bag.feature_dim})")
        except Exception as e:
            console.print(f"[bold red]Error processing {slide_p.name}: {e}[/bold red]")


@main.command()
@click.option("--feature-dir", required=True, help="Directory containing extracted feature bags (.h5, .pt, .npz)")
@click.option("--arch", default="abmil", help="MIL aggregator (mean_pool, max_pool, abmil, clam_sb, clam_mb, transmil)")
@click.option("--epochs", default=30, help="Number of training epochs")
@click.option("--output-dir", default="outputs/checkpoints", help="Output directory")
def train(feature_dir: str, arch: str, epochs: int, output_dir: str):
    """Train Multiple-Instance Learning aggregator with cross-validation."""
    dataset = SlideBagDataset.from_directory(feature_dir)
    console.print(f"[bold green]Loaded {len(dataset)} slide bags from {feature_dir}.[/bold green]")

    from slideshift.core.config import MILConfig
    config = MILConfig(architecture=arch, epochs=epochs)
    trainer = MILTrainer(config=config)

    splits = dataset.create_splits(strategy="patient_stratified", n_splits=3)
    train_ds, val_ds, test_ds = splits[0]

    console.print(f"[bold cyan]Training {arch.upper()} on Fold 1 ({len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test)...[/bold cyan]")
    res = trainer.train_fold(train_ds, val_ds)

    eval_test = trainer.evaluate(test_ds)
    table = Table(title=f"Evaluation Results ({arch.upper()})")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_row("AUROC", f"{eval_test['auroc']:.4f}")
    table.add_row("Balanced Accuracy", f"{eval_test['balanced_accuracy']:.4f}")
    table.add_row("Expected Calibration Error (ECE)", f"{eval_test['ece']:.4f}")
    console.print(table)


@main.command()
@click.option("--feature-dir", required=True, help="Directory containing extracted test feature bags")
@click.option("--output-dir", default="outputs/robustness", help="Directory to save robustness report")
def robustness(feature_dir: str, output_dir: str):
    """Run Robustness Lab benchmark across natural and synthetic shifts."""
    dataset = SlideBagDataset.from_directory(feature_dir)
    from slideshift.core.config import MILConfig
    trainer = MILTrainer(config=MILConfig(architecture="abmil"))
    # Fast fit on available dataset
    trainer.train_fold(dataset, dataset)

    lab = RobustnessLab(trainer=trainer)
    df_results = lab.run_benchmark(dataset, output_dir=output_dir)

    console.print("[bold green]Robustness Benchmark Complete![/bold green]")
    console.print(df_results[["shift_type", "severity", "auroc", "delta_auroc", "flip_rate", "ece"]].to_string())


@main.command()
@click.option("--port", default=8050, help="Web dashboard port (default: 8050)")
@click.option("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
def demo(port: int, host: str):
    """Launch the interactive SlideShift clinical & research dashboard."""
    from slideshift.demo.app import run_app
    console.print(f"[bold cyan]Launching SlideShift interactive UI at http://{host}:{port}...[/bold cyan]")
    run_app(port=port, host=host)


if __name__ == "__main__":
    main()
