"""Tests for advanced robustness and monitor modules."""

import numpy as np
import pytest

from slideshift.robustness.metrics import mmd_rbf, frechet_embedding_distance
from slideshift.robustness.conformal import ConformalPredictor, ConformalResult
from slideshift.monitor.fingerprint import (
    ScannerFingerprintExtractor,
    ScannerFingerprint,
    ScannerClusterer,
    rgb_to_od,
    deconvolve_he,
)
from slideshift.monitor.detector import hotelling_t2, energy_ood_score


# ─────────────────────────────────────────────
# MMD
# ─────────────────────────────────────────────

def test_mmd_identical_distributions():
    """MMD of a distribution with itself must be near 0."""
    rng = np.random.RandomState(0)
    X = rng.randn(50, 16).astype(np.float32)
    mmd = mmd_rbf(X, X)
    assert mmd < 0.05, f"MMD of identical distributions should be ~0, got {mmd}"


def test_mmd_different_distributions():
    """MMD of well-separated distributions must be substantially positive."""
    rng = np.random.RandomState(1)
    X = rng.randn(50, 16).astype(np.float32)
    Y = rng.randn(50, 16).astype(np.float32) + 5.0  # large offset
    mmd = mmd_rbf(X, Y)
    assert mmd > 0.1, f"MMD of shifted distributions should be > 0.1, got {mmd}"


def test_mmd_non_negative():
    """MMD must always be non-negative."""
    rng = np.random.RandomState(2)
    for _ in range(10):
        X = rng.randn(rng.randint(5, 30), 8).astype(np.float32)
        Y = rng.randn(rng.randint(5, 30), 8).astype(np.float32) + rng.randn()
        assert mmd_rbf(X, Y) >= 0.0


# ─────────────────────────────────────────────
# Fréchet Embedding Distance
# ─────────────────────────────────────────────

def test_fed_identical():
    """FED of a distribution with itself must be near 0."""
    rng = np.random.RandomState(3)
    X = rng.randn(40, 12).astype(np.float32)
    fed = frechet_embedding_distance(X, X)
    assert fed < 1.0, f"FED of identical distributions should be small, got {fed}"


def test_fed_shifted():
    """FED must increase for shifted distributions."""
    rng = np.random.RandomState(4)
    X = rng.randn(40, 8).astype(np.float32)
    Y = X + 3.0
    fed_close = frechet_embedding_distance(X, X + 0.1)
    fed_far = frechet_embedding_distance(X, Y)
    assert fed_far > fed_close, "FED must increase with shift magnitude"


# ─────────────────────────────────────────────
# Hotelling T² test
# ─────────────────────────────────────────────

def test_hotelling_inlier_high_pval():
    """An inlier sample should produce a high (non-significant) p-value."""
    rng = np.random.RandomState(5)
    reference = rng.randn(50, 4).astype(np.float32)
    query = rng.randn(4).astype(np.float32)  # same distribution
    t2, p = hotelling_t2(query, reference)
    assert p > 0.01, f"Inlier should have high p-value, got {p}"


def test_hotelling_outlier_low_pval():
    """A far outlier should produce a low p-value."""
    rng = np.random.RandomState(6)
    reference = rng.randn(60, 3).astype(np.float32)
    query = np.array([20.0, 20.0, 20.0], dtype=np.float32)  # extreme outlier
    t2, p = hotelling_t2(query, reference)
    assert p < 0.05, f"Extreme outlier should have low p-value, got {p}"


# ─────────────────────────────────────────────
# Energy OOD score
# ─────────────────────────────────────────────

def test_energy_ood_inlier_vs_outlier():
    """Inlier must have a lower energy score than a far OOD point."""
    rng = np.random.RandomState(7)
    reference = rng.randn(60, 8).astype(np.float32)
    inlier = rng.randn(8).astype(np.float32)         # near reference
    outlier = rng.randn(8).astype(np.float32) + 10.0 # far from reference
    e_in = energy_ood_score(inlier, reference)
    e_out = energy_ood_score(outlier, reference)
    assert e_out > e_in, f"OOD energy ({e_out}) should exceed inlier ({e_in})"


# ─────────────────────────────────────────────
# Scanner fingerprinting
# ─────────────────────────────────────────────

def _make_fake_he_tile(rng, brightness: float = 0.5, size: int = 64) -> np.ndarray:
    """Generate a fake H&E-like tile (pink/purple on white background)."""
    tile = np.ones((size, size, 3), dtype=np.uint8) * 240
    # Add purple (haematoxylin) blobs
    n_cells = rng.randint(3, 10)
    for _ in range(n_cells):
        r, c = rng.randint(5, size - 5, 2)
        rad = rng.randint(3, 8)
        rr, cc = np.ogrid[-rad:rad+1, -rad:rad+1]
        mask = rr**2 + cc**2 <= rad**2
        y_sl = slice(max(0, r - rad), min(size, r + rad + 1))
        x_sl = slice(max(0, c - rad), min(size, c + rad + 1))
        m = mask[:y_sl.stop - y_sl.start, :x_sl.stop - x_sl.start]
        sub = tile[y_sl, x_sl]
        sub[m] = np.array([140, 100, 180], dtype=np.uint8)  # purple
    # Scale by brightness
    tile = np.clip(tile.astype(float) * brightness, 0, 255).astype(np.uint8)
    return tile


def test_fingerprint_extraction():
    """ScannerFingerprintExtractor must produce a valid fingerprint."""
    rng = np.random.RandomState(8)
    tiles = [_make_fake_he_tile(rng) for _ in range(20)]
    extractor = ScannerFingerprintExtractor()
    fp = extractor.extract(tiles, slide_id="test_slide")

    assert fp.slide_id == "test_slide"
    assert fp.glass_mean_rgb.shape == (3,)
    assert fp.dynamic_range >= 0
    assert fp.colour_temperature > 0


def test_fingerprint_to_vector():
    """Fingerprint vector must be 1-D and contain the expected number of features."""
    rng = np.random.RandomState(9)
    tiles = [_make_fake_he_tile(rng) for _ in range(10)]
    extractor = ScannerFingerprintExtractor()
    fp = extractor.extract(tiles, slide_id="s0")
    vec = extractor.fingerprint_to_vector(fp)
    assert vec.ndim == 1
    # 3 (glass_mean) + 3 (glass_std) + 11 (h/e/sharpness/etc) = 17
    assert len(vec) == 17


def test_scanner_clusterer():
    """ScannerClusterer must assign cluster labels to fingerprints."""
    rng = np.random.RandomState(10)
    extractor = ScannerFingerprintExtractor()

    fingerprints = []
    for i in range(12):
        tiles = [_make_fake_he_tile(rng, brightness=0.5 + 0.5 * (i % 2)) for _ in range(8)]
        fp = extractor.extract(tiles, slide_id=f"slide_{i}")
        fingerprints.append(fp)

    clusterer = ScannerClusterer(n_clusters=2)
    labels = clusterer.fit_predict(fingerprints)
    assert len(labels) == 12
    assert set(labels).issubset({0, 1})


def test_od_deconvolution():
    """H&E OD deconvolution must produce non-negative concentrations."""
    rng = np.random.RandomState(11)
    tile = _make_fake_he_tile(rng)
    od = rgb_to_od(tile)
    h_ch, e_ch = deconvolve_he(od)
    assert h_ch.min() >= 0.0
    assert e_ch.min() >= 0.0
    assert h_ch.shape == tile.shape[:2]


# ─────────────────────────────────────────────
# Conformal prediction (advanced)
# ─────────────────────────────────────────────

def test_conformal_temperature_then_calibrate():
    """Temperature-scaled conformal calibration must not crash."""
    rng = np.random.RandomState(12)
    N, C = 100, 2
    labels = rng.randint(0, C, N)
    probs = rng.dirichlet(np.ones(C), size=N).astype(np.float32)

    cp = ConformalPredictor(target_coverage=0.90)
    cp.temperature_scale_and_calibrate(probs[:60], labels[:60])
    eval_result = cp.evaluate_coverage(probs[60:], labels[60:])
    assert eval_result.num_samples == 40


def test_conformal_defer_rate():
    """Defer rate must be 0 when predictions are very confident."""
    N, C = 50, 2
    labels = np.zeros(N, dtype=int)
    # Perfect confident predictions
    probs = np.zeros((N, C), dtype=np.float32)
    probs[:, 0] = 0.999
    probs[:, 1] = 0.001

    cp = ConformalPredictor(target_coverage=0.80)
    cp.calibrate(probs, labels)
    result = cp.predict(probs[0], slide_id="confident_slide")
    # A very confident prediction set should have size 1
    assert len(result.prediction_set) == 1
    assert not result.defer_to_expert

