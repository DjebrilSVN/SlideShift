"""Synthetic and natural perturbation generators for histology whole-slide images."""

import io
from typing import Callable, Dict, List, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter


# Standard Optical Density matrix for Hematoxylin and Eosin (Ruifrok and Johnston, 2001)
DEFAULT_HE_MATRIX = np.array([
    [0.644, 0.717, 0.267],  # Hematoxylin
    [0.093, 0.954, 0.283],  # Eosin
    [0.636, 0.001, 0.771],  # Residual / DAB
], dtype=np.float32)


def apply_he_stain_jitter(image_pil: Image.Image, severity: int = 1, seed: int = 42) -> Image.Image:
    """Apply Ruifrok H&E optical density deconvolution and perturb stain concentration matrices.
    
    Severity 1: slight protocol variance (5-10% variation)
    Severity 2: moderate stain shift (15-25% variation)
    Severity 3: severe over/under staining (35-50% variation)
    """
    rng = np.random.RandomState(seed + severity)
    jitter_scale = {1: 0.10, 2: 0.25, 3: 0.45}.get(severity, 0.20)

    arr = np.array(image_pil, dtype=np.float32) / 255.0
    arr = np.clip(arr, 1e-4, 1.0)

    # Convert to Optical Density (Beer-Lambert: OD = -log10(I))
    od = -np.log10(arr)  # (H, W, 3)
    orig_shape = od.shape
    od_flat = od.reshape(-1, 3)

    # Deconvolve into stain concentrations C = OD * M^-1
    he_matrix = DEFAULT_HE_MATRIX.copy()
    try:
        inv_he = np.linalg.pinv(he_matrix)
        concentrations = np.dot(od_flat, inv_he.T)  # (pixels, 3)

        # Multiplicative jitter on Hematoxylin (0) and Eosin (1)
        h_factor = 1.0 + rng.uniform(-jitter_scale, jitter_scale)
        e_factor = 1.0 + rng.uniform(-jitter_scale, jitter_scale)
        concentrations[:, 0] *= h_factor
        concentrations[:, 1] *= e_factor
        concentrations = np.clip(concentrations, 0, None)

        # Reconstruct OD
        od_reconstructed = np.dot(concentrations, he_matrix)
        od_reconstructed = od_reconstructed.reshape(orig_shape)

        # Reconstruct RGB
        rgb_reconstructed = 10.0 ** (-od_reconstructed) * 255.0
        out_uint8 = np.clip(rgb_reconstructed, 0, 255).astype(np.uint8)
        return Image.fromarray(out_uint8)
    except Exception:
        # Fallback to color balance enhancement
        factor = 1.0 + jitter_scale * 0.7
        enhancer = ImageEnhance.Color(image_pil)
        return enhancer.enhance(factor)


def apply_hue_saturation_shift(image_pil: Image.Image, severity: int = 1) -> Image.Image:
    """Shift saturation and contrast to simulate different laboratory chemical batches."""
    sat_factors = {1: 0.85, 2: 0.65, 3: 0.40}
    contrast_factors = {1: 1.15, 2: 1.30, 3: 1.55}

    img = ImageEnhance.Color(image_pil).enhance(sat_factors.get(severity, 0.7))
    img = ImageEnhance.Contrast(img).enhance(contrast_factors.get(severity, 1.2))
    return img


def apply_gaussian_blur(image_pil: Image.Image, severity: int = 1) -> Image.Image:
    """Simulate optical defocus blur caused by z-axis focal plane drift."""
    radii = {1: 1.2, 2: 2.5, 3: 4.5}
    radius = radii.get(severity, 2.0)
    return image_pil.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_jpeg_compression(image_pil: Image.Image, severity: int = 1) -> Image.Image:
    """Simulate lossy PACS compression and network transmission artifacts."""
    qualities = {1: 65, 2: 40, 3: 18}
    quality = qualities.get(severity, 40)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_additive_noise(image_pil: Image.Image, severity: int = 1, seed: int = 42) -> Image.Image:
    """Simulate sensor shot noise from low-exposure camera sensors."""
    sigmas = {1: 8.0, 2: 18.0, 3: 35.0}
    sigma = sigmas.get(severity, 15.0)
    rng = np.random.RandomState(seed + severity)
    arr = np.array(image_pil, dtype=np.float32)
    noise = rng.normal(0, sigma, arr.shape)
    arr_noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_noisy)


def apply_resolution_mismatch(image_pil: Image.Image, severity: int = 1) -> Image.Image:
    """Simulate feeding 40x data where 20x is expected (downsample mismatch bug)."""
    scale_factors = {1: 0.75, 2: 0.50, 3: 0.35}
    scale = scale_factors.get(severity, 0.50)
    w, h = image_pil.size
    smaller_w, smaller_h = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
    # Downsample then upsample back to original dimensions
    low_res = image_pil.resize((smaller_w, smaller_h), Image.Resampling.BILINEAR)
    return low_res.resize((w, h), Image.Resampling.BILINEAR)


PERTURBATION_REGISTRY: Dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "stain_he_jitter": apply_he_stain_jitter,
    "stain_saturation_shift": apply_hue_saturation_shift,
    "gaussian_blur": apply_gaussian_blur,
    "jpeg_compression": apply_jpeg_compression,
    "additive_noise": apply_additive_noise,
    "resolution_mismatch": apply_resolution_mismatch,
}

