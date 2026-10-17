"""Scanner fingerprinting: profiling low-level optical density statistics
to detect scanner / staining protocol changes without a trained model.

A scanner fingerprint captures signal-level properties that are unlikely to
change due to biological variation but will change when the scanner, stain
protocol, or stain vendor changes.

Features captured per slide:
  - Background glass RGB statistics (empty regions ≈ 255 on bright-field)
  - Haematoxylin and eosin OD means and variances (Ruifrok deconvolution)
  - Sharpness distribution (Laplacian variance per tile)
  - Colour temperature (R / B ratio in tissue regions)
  - Slide dynamic range and 5th/95th percentile intensity

Reference implementation inspired by:
  Tellez et al., "Quantifying the effects of data augmentation and stain
  color normalization in convolutional neural networks", MedIA 2019.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


# Ruifrok & Johnston (2001) stain OD matrix for H&E.
# Rows: [Haematoxylin, Eosin]
_HE_OD_MATRIX = np.array(
    [[0.6500286, 0.7044536, 0.2860126],
     [0.2760799, 0.8014827, 0.5327570]],
    dtype=np.float32,
)


def rgb_to_od(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert uint8 RGB image to optical density."""
    rgb_f = rgb.astype(np.float32) / 255.0 + eps
    od = -np.log(rgb_f)
    return od


def deconvolve_he(od: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Separate H&E channels using Ruifrok matrix.

    Args:
        od: (..., 3) optical density array.

    Returns:
        h_channel, e_channel — concentrations ∈ [0, ∞).
    """
    shape = od.shape
    od_flat = od.reshape(-1, 3)
    # Least-squares: concentrations = OD @ pinv(M^T)
    m_pinv = np.linalg.pinv(_HE_OD_MATRIX)             # (3, 2)
    concs = od_flat @ m_pinv                             # (N, 2)
    concs = np.clip(concs, 0.0, None)
    h_ch = concs[:, 0].reshape(shape[:-1])
    e_ch = concs[:, 1].reshape(shape[:-1])
    return h_ch, e_ch


@dataclass
class ScannerFingerprint:
    """Low-level scanner/stain fingerprint of one slide."""
    slide_id: str
    # Background glass
    glass_mean_rgb: np.ndarray          # (3,)
    glass_std_rgb: np.ndarray           # (3,)
    # H&E OD statistics
    h_od_mean: float
    h_od_std: float
    e_od_mean: float
    e_od_std: float
    # Sharpness
    sharpness_mean: float
    sharpness_p10: float
    sharpness_p90: float
    # Colour temperature (R/B ratio)
    colour_temperature: float
    # Dynamic range
    intensity_p5: float
    intensity_p95: float
    dynamic_range: float


class ScannerFingerprintExtractor:
    """Extract scanner fingerprints from slide image patches.

    Usage::

        extractor = ScannerFingerprintExtractor()
        fp = extractor.extract(tile_list, slide_id)
    """

    def __init__(self, background_threshold: int = 220):
        """
        Args:
            background_threshold: Tiles with mean brightness above this value
                                  are treated as glass/background.
        """
        self.bg_thresh = background_threshold

    def _laplacian_variance(self, gray: np.ndarray) -> float:
        """Estimate sharpness via Laplacian variance (no OpenCV needed)."""
        # 2-D Laplacian via finite differences
        gy = np.diff(gray.astype(np.float32), n=2, axis=0)
        gx = np.diff(gray.astype(np.float32), n=2, axis=1)
        lap = gy[:, 1:-1] if gy.shape[1] > 2 else gy
        lap_x = gx[1:-1, :] if gx.shape[0] > 2 else gx
        min_rows = min(lap.shape[0], lap_x.shape[0])
        min_cols = min(lap.shape[1], lap_x.shape[1])
        if min_rows == 0 or min_cols == 0:
            return float(np.var(gray.astype(np.float32)))
        combined = lap[:min_rows, :min_cols] + lap_x[:min_rows, :min_cols]
        return float(np.var(combined))

    def extract(
        self,
        tiles: List[np.ndarray],
        slide_id: str,
    ) -> ScannerFingerprint:
        """Extract fingerprint from a list of RGB tile arrays.

        Args:
            tiles:    List of (H, W, 3) uint8 RGB numpy arrays.
            slide_id: Slide identifier.

        Returns:
            ScannerFingerprint dataclass.
        """
        if not tiles:
            return self._empty_fingerprint(slide_id)

        bg_tiles = []
        tissue_tiles = []
        sharpness_vals = []
        all_intensities = []

        for tile in tiles:
            gray = tile.mean(axis=-1)
            brightness = float(gray.mean())
            sharpness = self._laplacian_variance(gray)
            sharpness_vals.append(sharpness)
            all_intensities.append(gray.ravel())

            if brightness >= self.bg_thresh:
                bg_tiles.append(tile)
            else:
                tissue_tiles.append(tile)

        # Background statistics
        if bg_tiles:
            bg_stack = np.concatenate(
                [t.reshape(-1, 3) for t in bg_tiles], axis=0
            ).astype(np.float32)
            glass_mean = bg_stack.mean(axis=0)
            glass_std = bg_stack.std(axis=0)
        else:
            glass_mean = np.array([240.0, 240.0, 240.0], dtype=np.float32)
            glass_std = np.ones(3, dtype=np.float32)

        # H&E statistics from tissue tiles
        h_means, e_means, h_stds, e_stds, cr_temps = [], [], [], [], []
        for tile in tissue_tiles[:500]:                        # cap for speed
            od = rgb_to_od(tile)
            h_ch, e_ch = deconvolve_he(od)
            h_means.append(float(h_ch.mean()))
            h_stds.append(float(h_ch.std()))
            e_means.append(float(e_ch.mean()))
            e_stds.append(float(e_ch.std()))
            r = float(tile[:, :, 0].mean())
            b = float(tile[:, :, 2].mean())
            cr_temps.append(r / max(b, 1.0))

        h_od_mean = float(np.mean(h_means)) if h_means else 0.0
        h_od_std = float(np.mean(h_stds)) if h_stds else 0.0
        e_od_mean = float(np.mean(e_means)) if e_means else 0.0
        e_od_std = float(np.mean(e_stds)) if e_stds else 0.0
        colour_temp = float(np.mean(cr_temps)) if cr_temps else 1.0

        # Intensity distribution
        intensity_all = np.concatenate(all_intensities) if all_intensities else np.array([128.0])
        i_p5 = float(np.percentile(intensity_all, 5))
        i_p95 = float(np.percentile(intensity_all, 95))
        dyn_range = i_p95 - i_p5

        # Sharpness
        sharp_arr = np.array(sharpness_vals, dtype=np.float32)
        sharp_mean = float(sharp_arr.mean())
        sharp_p10 = float(np.percentile(sharp_arr, 10))
        sharp_p90 = float(np.percentile(sharp_arr, 90))

        return ScannerFingerprint(
            slide_id=slide_id,
            glass_mean_rgb=glass_mean,
            glass_std_rgb=glass_std,
            h_od_mean=h_od_mean,
            h_od_std=h_od_std,
            e_od_mean=e_od_mean,
            e_od_std=e_od_std,
            sharpness_mean=sharp_mean,
            sharpness_p10=sharp_p10,
            sharpness_p90=sharp_p90,
            colour_temperature=colour_temp,
            intensity_p5=i_p5,
            intensity_p95=i_p95,
            dynamic_range=dyn_range,
        )

    def fingerprint_to_vector(self, fp: ScannerFingerprint) -> np.ndarray:
        """Flatten all fingerprint fields into a 1-D feature vector."""
        return np.concatenate([
            fp.glass_mean_rgb,
            fp.glass_std_rgb,
            [fp.h_od_mean, fp.h_od_std,
             fp.e_od_mean, fp.e_od_std,
             fp.sharpness_mean, fp.sharpness_p10, fp.sharpness_p90,
             fp.colour_temperature,
             fp.intensity_p5, fp.intensity_p95, fp.dynamic_range],
        ]).astype(np.float32)

    def _empty_fingerprint(self, slide_id: str) -> ScannerFingerprint:
        return ScannerFingerprint(
            slide_id=slide_id,
            glass_mean_rgb=np.array([240.0, 240.0, 240.0], dtype=np.float32),
            glass_std_rgb=np.ones(3, dtype=np.float32),
            h_od_mean=0.0, h_od_std=0.0,
            e_od_mean=0.0, e_od_std=0.0,
            sharpness_mean=0.0, sharpness_p10=0.0, sharpness_p90=0.0,
            colour_temperature=1.0,
            intensity_p5=0.0, intensity_p95=255.0, dynamic_range=255.0,
        )


class ScannerClusterer:
    """Cluster slides by fingerprint to identify scanner groups.

    Uses K-means or hierarchical clustering on the fingerprint vectors.
    Useful for:
      1. Verifying site/scanner labels in the dataset.
      2. Detecting when a new scanner is introduced without a label change.
      3. Exploring whether embeddings cluster by biology or scanner.
    """

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self._labels: Optional[np.ndarray] = None
        self._km = None

    def fit_predict(self, fingerprints: List[ScannerFingerprint]) -> np.ndarray:
        """Cluster fingerprints and return cluster labels."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        extractor = ScannerFingerprintExtractor()
        vectors = np.stack([extractor.fingerprint_to_vector(fp) for fp in fingerprints])
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)

        km = KMeans(n_clusters=min(self.n_clusters, len(fingerprints)), n_init="auto", random_state=42)
        self._labels = km.fit_predict(vectors_scaled)
        self._km = km
        self._scaler = scaler
        return self._labels

    def compare_against_site_labels(
        self,
        fingerprints: List[ScannerFingerprint],
        site_labels: List[str],
    ) -> Dict[str, float]:
        """Measure how well fingerprint clusters align with known site labels.

        Returns:
            dict with 'ami' (Adjusted Mutual Information) and 'ari'
            (Adjusted Rand Index) scores.
        """
        from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
        from sklearn.preprocessing import LabelEncoder

        cluster_labels = self.fit_predict(fingerprints)
        site_int = LabelEncoder().fit_transform(site_labels)
        ami = float(adjusted_mutual_info_score(site_int, cluster_labels))
        ari = float(adjusted_rand_score(site_int, cluster_labels))
        return {"ami": ami, "ari": ari}

