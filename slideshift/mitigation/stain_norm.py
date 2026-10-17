"""Stain normalization techniques: Reinhard and Macenko methods."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from PIL import Image


class BaseStainNormalizer(ABC):
    """Abstract base class for stain normalization algorithms."""

    @abstractmethod
    def fit(self, target_image: np.ndarray) -> "BaseStainNormalizer":
        """Fit stain reference statistics on a target reference image."""
        pass

    @abstractmethod
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image to match the target reference stain distribution."""
        pass


class ReinhardNormalizer(BaseStainNormalizer):
    """Reinhard stain normalizer using statistical alignment in CIELAB color space.
    
    Reference:
        Reinhard et al., 'Color transfer between images', IEEE CGA, 2001.
    """

    def __init__(self):
        # Default reference values (means and stds in LAB space from standard H&E)
        self.target_mean = np.array([148.60, 168.33, 110.35], dtype=np.float32)
        self.target_std = np.array([32.49, 9.15, 9.41], dtype=np.float32)

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        # Fast perceptual RGB to LAB approximation
        r = rgb[..., 0].astype(np.float32)
        g = rgb[..., 1].astype(np.float32)
        b = rgb[..., 2].astype(np.float32)

        # Log color transformation
        l_channel = 0.3811 * r + 0.5783 * g + 0.0402 * b
        m_channel = 0.1967 * r + 0.7244 * g + 0.0782 * b
        s_channel = 0.0241 * r + 0.1288 * g + 0.8444 * b

        l_channel = np.log(np.clip(l_channel, 1.0, 255.0))
        m_channel = np.log(np.clip(m_channel, 1.0, 255.0))
        s_channel = np.log(np.clip(s_channel, 1.0, 255.0))

        lab_l = (l_channel + m_channel + s_channel) / np.sqrt(3)
        lab_a = (l_channel + m_channel - 2 * s_channel) / np.sqrt(6)
        lab_b = (l_channel - m_channel) / np.sqrt(2)

        return np.stack([lab_l, lab_a, lab_b], axis=-1)

    def _lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        lab_l = lab[..., 0]
        lab_a = lab[..., 1]
        lab_b = lab[..., 2]

        l_c = lab_l / np.sqrt(3) + lab_a / np.sqrt(6) + lab_b / np.sqrt(2)
        m_c = lab_l / np.sqrt(3) + lab_a / np.sqrt(6) - lab_b / np.sqrt(2)
        s_c = lab_l / np.sqrt(3) - 2 * lab_a / np.sqrt(6)

        r = np.exp(l_c)
        g = np.exp(m_c)
        b = np.exp(s_c)

        return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

    def fit(self, target_image: np.ndarray) -> "ReinhardNormalizer":
        lab = self._rgb_to_lab(target_image)
        # Exclude pure white background (>90% intensity)
        mask = target_image.mean(axis=-1) < 225
        if np.any(mask):
            self.target_mean = np.mean(lab[mask], axis=0)
            self.target_std = np.std(lab[mask], axis=0) + 1e-5
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        lab = self._rgb_to_lab(image)
        mask = image.mean(axis=-1) < 225
        if not np.any(mask):
            return image.copy()

        src_mean = np.mean(lab[mask], axis=0)
        src_std = np.std(lab[mask], axis=0) + 1e-5

        norm_lab = lab.copy()
        for c in range(3):
            norm_lab[..., c] = ((lab[..., c] - src_mean[c]) / src_std[c]) * self.target_std[c] + self.target_mean[c]

        return self._lab_to_rgb(norm_lab)


class MacenkoNormalizer(BaseStainNormalizer):
    """Macenko optical density SVD stain normalizer.
    
    Reference:
        Macenko et al., 'A method for normalizing histology slides for quantitative analysis', ISBI 2009.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.15):
        self.alpha = alpha
        self.beta = beta
        # Reference H&E stain vectors
        self.he_ref = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581],
        ], dtype=np.float32)
        self.max_c_ref = np.array([1.9705, 1.0308], dtype=np.float32)

    def fit(self, target_image: np.ndarray) -> "MacenkoNormalizer":
        # Target reference vector can be extracted from reference tissue
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = np.clip(arr, 1e-4, 1.0)
        od = -np.log10(arr)
        orig_shape = od.shape
        od_flat = od.reshape(-1, 3)

        # Remove background (low OD pixels)
        od_thresh = od_flat[np.all(od_flat > self.beta, axis=1)]
        if len(od_thresh) < 100:
            return image.copy()

        try:
            # SVD on OD covariance
            _, _, vh = np.linalg.svd(od_thresh, full_matrices=False)
            phi = np.dot(od_thresh, vh[:2].T)
            angles = np.arctan2(phi[:, 1], phi[:, 0])

            # Robust min/max angles via percentiles (alpha and 100-alpha)
            min_phi = np.percentile(angles, self.alpha)
            max_phi = np.percentile(angles, 100 - self.alpha)

            v_min = np.dot(vh[:2].T, np.array([np.cos(min_phi), np.sin(min_phi)]))
            v_max = np.dot(vh[:2].T, np.array([np.cos(max_phi), np.sin(max_phi)]))

            if v_min[0] > v_max[0]:
                he_stains = np.stack([v_min, v_max], axis=1)
            else:
                he_stains = np.stack([v_max, v_min], axis=1)

            # Stain concentrations
            he_inv = np.linalg.pinv(he_stains)
            c = np.dot(od_flat, he_inv.T)

            # Normalize concentrations to target reference maximums
            max_c = np.percentile(c, 99, axis=0)
            max_c = np.clip(max_c, 1e-5, None)
            c_norm = c * (self.max_c_ref / max_c)

            # Reconstruct RGB
            od_norm = np.dot(c_norm, self.he_ref.T).reshape(orig_shape)
            rgb_norm = 10.0 ** (-od_norm) * 255.0
            return np.clip(rgb_norm, 0, 255).astype(np.uint8)

        except Exception:
            return image.copy()

