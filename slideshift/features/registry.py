"""Foundation model metadata, registry, and licensing requirements."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ModelInfo:
    """Metadata specification for a pathology or vision foundation model."""
    name: str
    architecture: str
    embedding_dim: int
    input_size: Tuple[int, int]
    is_gated: bool
    hf_repo_id: Optional[str]
    license_type: str
    pretraining_data_notes: str
    contamination_risk: Dict[str, str]  # dataset_name -> 'Pretrained' / 'Unseen' / 'Partial'
    description: str


MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "resnet50": ModelInfo(
        name="resnet50",
        architecture="ResNet-50 (ImageNet-1k)",
        embedding_dim=2048,
        input_size=(224, 224),
        is_gated=False,
        hf_repo_id="torchvision/resnet50",
        license_type="Apache-2.0 / PyTorch BSD",
        pretraining_data_notes="Pretrained on standard ImageNet-1K natural images only.",
        contamination_risk={
            "camelyon16": "Unseen (zero pathology pretraining)",
            "camelyon17": "Unseen",
            "tcga_nsclc": "Unseen",
            "panda": "Unseen",
            "cptac": "Unseen",
        },
        description="Standard computer vision baseline. Shows what pathology foundation pretraining actually buys.",
    ),
    "phikon": ModelInfo(
        name="phikon",
        architecture="ViT-B/16 (iBOT)",
        embedding_dim=768,
        input_size=(224, 224),
        is_gated=False,
        hf_repo_id="owkin/phikon",
        license_type="Open (CC-BY-NC-4.0)",
        pretraining_data_notes="Trained on TCGA slides (~40M pan-cancer tiles) with iBOT SSL.",
        contamination_risk={
            "camelyon16": "Unseen",
            "camelyon17": "Unseen",
            "tcga_nsclc": "Pretrained (TCGA was used during iBOT training)",
            "panda": "Unseen",
            "cptac": "Unseen",
        },
        description="Strong open-weights baseline from Owkin with straightforward licensing.",
    ),
    "uni": ModelInfo(
        name="uni",
        architecture="ViT-L/16",
        embedding_dim=1024,
        input_size=(224, 224),
        is_gated=True,
        hf_repo_id="mahmoodlab/UNI",
        license_type="Non-commercial, gated (Mahmood Lab / Harvard BWH)",
        pretraining_data_notes="Trained on Mass-100K (>100M tiles from 100k slides across 20 hospitals).",
        contamination_risk={
            "camelyon16": "Unseen",
            "camelyon17": "Unseen",
            "tcga_nsclc": "Unseen in Mass-100k pretraining set",
            "panda": "Unseen",
            "cptac": "Unseen",
        },
        description="Widely cited standard pathology foundation model benchmark.",
    ),
    "conch": ModelInfo(
        name="conch",
        architecture="CoCa-ViT-B / Vision-Language",
        embedding_dim=512,
        input_size=(448, 448),
        is_gated=True,
        hf_repo_id="mahmoodlab/CONCH",
        license_type="Non-commercial, gated (Mahmood Lab / Harvard BWH)",
        pretraining_data_notes="Multimodal vision-language training on 1.17M histology image-caption pairs.",
        contamination_risk={
            "camelyon16": "Unseen",
            "camelyon17": "Unseen",
            "tcga_nsclc": "Unseen",
            "panda": "Unseen",
            "cptac": "Unseen",
        },
        description="Vision-language foundation model allowing both tile features and zero-shot text-prompted classification.",
    ),
    "virchow": ModelInfo(
        name="virchow",
        architecture="ViT-H/14",
        embedding_dim=1280,
        input_size=(224, 224),
        is_gated=True,
        hf_repo_id="paige-ai/Virchow",
        license_type="Non-commercial, gated (Paige AI / MSKCC)",
        pretraining_data_notes="Trained on 1.5 million WSIs from MSKCC on Paige clinical archive.",
        contamination_risk={
            "camelyon16": "Unseen",
            "camelyon17": "Unseen",
            "tcga_nsclc": "Unseen (trained on proprietary MSK clinical cohort)",
            "panda": "Unseen",
            "cptac": "Unseen",
        },
        description="Massive scale ViT-H/14 with large embedding dimension and strong benchmark performance.",
    ),
    "mock": ModelInfo(
        name="mock",
        architecture="Synthetic Feature Extractor",
        embedding_dim=512,
        input_size=(224, 224),
        is_gated=False,
        hf_repo_id=None,
        license_type="MIT",
        pretraining_data_notes="In-memory synthetic feature projection for testing and CI.",
        contamination_risk={
            "camelyon16": "Unseen",
            "camelyon17": "Unseen",
            "tcga_nsclc": "Unseen",
            "panda": "Unseen",
            "cptac": "Unseen",
        },
        description="Fast offline mock encoder for unit tests, CI, and local verification.",
    ),
}


def get_model_info(model_name: str) -> ModelInfo:
    """Retrieve metadata for a specified foundation model."""
    name_clean = model_name.lower().strip()
    if name_clean not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown foundation model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name_clean]

