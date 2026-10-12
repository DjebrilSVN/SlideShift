"""Configuration system for SlideShift pipelines."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class TilingConfig:
    """Configuration for slide segmentation and physical tiling."""
    target_mpp: float = 0.50  # ~20x magnification (0.5 microns/pixel)
    tile_size_px: int = 256  # Tile width/height at target MPP
    min_tissue_fraction: float = 0.20  # Minimum tissue coverage to keep a tile
    overlap_px: int = 0  # Overlap between adjacent tiles (e.g., 0 or 32 for smooth heatmaps)
    segmentation_downsample: int = 32  # Downsample factor for tissue thresholding
    blur_threshold: float = 100.0  # Laplacian variance threshold below which a slide is blurry
    save_thumbnails: bool = True


@dataclass
class FoundationModelConfig:
    """Configuration for tile feature extraction foundation models."""
    model_name: str = "resnet50"  # 'resnet50', 'phikon', 'uni', 'conch', 'virchow', 'mock'
    checkpoint_path: Optional[str] = None
    batch_size: int = 64
    num_workers: int = 4
    use_fp16: bool = True
    device: str = "cuda"  # "cuda" or "cpu"
    cache_format: str = "h5"  # "h5", "pt", or "parquet"
    hf_token: Optional[str] = None


@dataclass
class MILConfig:
    """Configuration for Multiple-Instance Learning aggregation."""
    architecture: str = "abmil"  # 'mean_pool', 'max_pool', 'abmil', 'clam_sb', 'clam_mb', 'transmil'
    num_classes: int = 2
    hidden_dim: int = 256
    dropout: float = 0.25
    instance_dropout: float = 0.10  # Randomly drop 10% of tiles during training
    feature_noise_std: float = 0.02  # Gaussian noise added to embeddings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10
    batch_size: int = 1  # 1 slide bag per optimization step
    class_weighting: bool = True
    seeds: List[int] = field(default_factory=lambda: [42, 101, 2024])
    n_splits: int = 5
    split_strategy: str = "patient_stratified"  # "patient_stratified", "site_held_out"


@dataclass
class RobustnessConfig:
    """Configuration for robustness lab perturbations and metrics."""
    test_subset_size: int = 200
    shifts_to_test: List[str] = field(default_factory=lambda: [
        "stain_he_jitter",
        "stain_saturation_shift",
        "gaussian_blur",
        "jpeg_compression",
        "additive_noise",
        "resolution_mismatch",
        "scanner_difference",
    ])
    severity_levels: List[int] = field(default_factory=lambda: [1, 2, 3])
    evaluate_calibration: bool = True
    evaluate_flip_rate: bool = True
    evaluate_heatmap_stability: bool = True
    evaluate_embedding_shortcuts: bool = True


@dataclass
class MitigationConfig:
    """Configuration for shift mitigations."""
    stain_normalization: str = "none"  # "none", "macenko", "reinhard", "vahadane"
    per_slide_standardization: bool = False  # Zero-mean unit-variance on tile features
    domain_adversarial: bool = False  # DANN site gradient reversal
    domain_weight: float = 0.1
    ensemble_models: List[str] = field(default_factory=list)


@dataclass
class ShiftMonitorConfig:
    """Configuration for the real-time slide shift detector."""
    method: str = "combined"  # "mahalanobis", "mmd", "isolation_forest", "combined"
    target_false_alarm_rate: float = 0.05
    stain_weight: float = 0.4
    embedding_weight: float = 0.6
    cache_reference_stats: bool = True


@dataclass
class PipelineConfig:
    """Master configuration for the SlideShift pipeline."""
    project_name: str = "SlideShift"
    data_dir: str = "data"
    output_dir: str = "outputs"
    task_name: str = "camelyon16"  # "camelyon16", "tcga_nsclc", "panda"
    tiling: TilingConfig = field(default_factory=TilingConfig)
    encoder: FoundationModelConfig = field(default_factory=FoundationModelConfig)
    mil: MILConfig = field(default_factory=MILConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    monitor: ShiftMonitorConfig = field(default_factory=ShiftMonitorConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        tiling_data = data.get("tiling", {})
        encoder_data = data.get("encoder", {})
        mil_data = data.get("mil", {})
        robustness_data = data.get("robustness", {})
        mitigation_data = data.get("mitigation", {})
        monitor_data = data.get("monitor", {})

        return cls(
            project_name=data.get("project_name", "SlideShift"),
            data_dir=data.get("data_dir", "data"),
            output_dir=data.get("output_dir", "outputs"),
            task_name=data.get("task_name", "camelyon16"),
            tiling=TilingConfig(**tiling_data),
            encoder=FoundationModelConfig(**encoder_data),
            mil=MILConfig(**mil_data),
            robustness=RobustnessConfig(**robustness_data),
            mitigation=MitigationConfig(**mitigation_data),
            monitor=ShiftMonitorConfig(**monitor_data),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

