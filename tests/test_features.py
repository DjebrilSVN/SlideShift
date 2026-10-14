"""Tests for foundation model registry, mock encoder, and feature caching."""

import pytest
import numpy as np
from PIL import Image
from slideshift.features.registry import get_model_info, MODEL_REGISTRY
from slideshift.features.encoders import MockEncoder, EncoderFactory
from slideshift.features.extractor import SlideFeatureExtractor
from slideshift.core.types import SlideBag


def test_model_registry():
    assert "resnet50" in MODEL_REGISTRY
    assert "uni" in MODEL_REGISTRY
    assert "conch" in MODEL_REGISTRY
    assert "virchow" in MODEL_REGISTRY
    assert "phikon" in MODEL_REGISTRY

    info = get_model_info("uni")
    assert info.embedding_dim == 1024
    assert info.is_gated is True


def test_mock_encoder():
    encoder = MockEncoder(embedding_dim=512)
    assert encoder.embedding_dim == 512
    assert encoder.input_size == (224, 224)

    tiles = [Image.new("RGB", (256, 256), (200, 150, 180)) for _ in range(5)]
    embs = encoder.encode_tiles(tiles)
    assert embs.shape == (5, 512)
    assert embs.dtype == np.float16


def test_feature_bag_caching(tmp_path):
    from slideshift.features.extractor import HAS_H5PY
    fmt = "h5" if HAS_H5PY else "npz"
    bag = SlideBag(
        slide_id="slide_test_01",
        features=np.random.randn(20, 128).astype(np.float16),
        coordinates=np.zeros((20, 2), dtype=np.int32),
        label=1,
        patient_id="patient_A",
        site="hospital_1",
    )

    extractor = SlideFeatureExtractor(encoder=MockEncoder(embedding_dim=128), cache_format=fmt)
    saved_path = extractor.save_bag(bag, tmp_path)
    assert saved_path.exists()

    loaded_bag = SlideFeatureExtractor.load_bag(saved_path)
    assert loaded_bag.slide_id == bag.slide_id
    assert loaded_bag.label == 1
    assert loaded_bag.patient_id == "patient_A"
    assert loaded_bag.features.shape == (20, 128)
