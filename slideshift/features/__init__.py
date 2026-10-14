"""Foundation model feature extraction and caching."""

from slideshift.features.registry import MODEL_REGISTRY, ModelInfo, get_model_info
from slideshift.features.base import BaseFoundationEncoder
from slideshift.features.encoders import (
    MockEncoder,
    ResNet50Encoder,
    HuggingFacePathologyEncoder,
    EncoderFactory,
)
from slideshift.features.extractor import SlideFeatureExtractor

__all__ = [
    "MODEL_REGISTRY",
    "ModelInfo",
    "get_model_info",
    "BaseFoundationEncoder",
    "MockEncoder",
    "ResNet50Encoder",
    "HuggingFacePathologyEncoder",
    "EncoderFactory",
    "SlideFeatureExtractor",
]

