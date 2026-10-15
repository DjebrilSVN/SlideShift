"""Tests for MIL aggregators, patient grouping splits, and training."""

import pytest
import numpy as np
from slideshift.core.types import SlideBag
from slideshift.core.config import MILConfig
from slideshift.mil.dataset import SlideBagDataset
from slideshift.mil.trainer import MILTrainer, compute_expected_calibration_error, compute_bootstrap_ci
from slideshift.mil.regularization import apply_instance_dropout, apply_feature_noise


def test_patient_stratified_split():
    bags = []
    # 10 patients, 2 slides each = 20 slides
    for p in range(10):
        label = p % 2
        for s in range(2):
            bag = SlideBag(
                slide_id=f"p{p}_s{s}",
                features=np.random.randn(15, 64).astype(np.float32),
                coordinates=np.zeros((15, 2), dtype=int),
                label=label,
                patient_id=f"patient_{p}",
                site=f"site_{p % 3}",
            )
            bags.append(bag)

    dataset = SlideBagDataset(bags)
    splits = dataset.create_splits(strategy="patient_stratified", n_splits=3)
    assert len(splits) == 3

    train_ds, val_ds, test_ds = splits[0]
    # Verify NO patient leakage between train and test
    train_patients = set(train_ds.patient_ids)
    test_patients = set(test_ds.patient_ids)
    assert len(train_patients.intersection(test_patients)) == 0


def test_regularization():
    feats = np.random.randn(50, 32).astype(np.float32)
    coords = np.zeros((50, 2), dtype=int)

    # Instance dropout
    dr_f, dr_c = apply_instance_dropout(feats, coords, dropout_rate=0.20)
    assert len(dr_f) < 50
    assert len(dr_f) >= 35

    # Feature noise
    noisy = apply_feature_noise(feats, noise_std=0.05)
    assert not np.array_equal(feats, noisy)
    assert abs(np.mean(noisy - feats)) < 0.05


def test_mil_training_and_eval():
    bags = []
    # Create distinguishable classes for testing
    for i in range(12):
        label = i % 2
        feat = np.random.randn(20, 32).astype(np.float32)
        if label == 1:
            feat[:, 0] += 2.0  # distinctive signal
        bag = SlideBag(
            slide_id=f"slide_{i}",
            features=feat,
            coordinates=np.zeros((20, 2), dtype=int),
            label=label,
            patient_id=f"pat_{i}",
            site="site_A",
        )
        bags.append(bag)

    ds = SlideBagDataset(bags)
    trainer = MILTrainer(config=MILConfig(architecture="abmil", epochs=5, num_classes=2))
    res = trainer.train_fold(ds, ds)

    eval_out = trainer.evaluate(ds)
    assert "auroc" in eval_out
    assert "balanced_accuracy" in eval_out
    assert "ece" in eval_out
    assert 0.0 <= eval_out["ece"] <= 1.0


def test_calibration_and_bootstrap():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    ece = compute_expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0

    mean_auc, low, high = compute_bootstrap_ci(labels, probs, metric_name="auroc", n_bootstraps=50)
    assert 0.0 <= low <= high <= 1.0

