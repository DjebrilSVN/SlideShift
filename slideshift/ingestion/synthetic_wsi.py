"""Synthetic whole-slide image generator for tests and reproducible benchmarking.

Generates realistic histopathological tissue morphology including:
- Multi-lobed organic tissue contours with realistic crevices and margins
- Eosinophilic fibrous stroma with collagenous micro-textures
- Glandular and ductal structures with clear optical lumens
- Normal cellularity with vesicular basophilic nuclei
- Invasive carcinoma regions with nuclear pleomorphism, hyperchromasia, and cribriform nests
- Scanner-specific optical density profiles (Aperio, Philips, Hamamatsu, Leica)
- Controlled histological artifacts (surgical pen marks, tissue folds, defocus blur)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter


def generate_synthetic_slide_data(
    dimensions: Tuple[int, int] = (2048, 2048),
    has_tumor: bool = True,
    site_id: str = "site_A",
    scanner_id: str = "scanner_aperio",
    add_pen_mark: bool = False,
    add_blur: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """Generate high-fidelity in-memory synthetic histology slide image, tumor mask, and metadata.

    Args:
        dimensions: (width, height) of generated image.
        has_tumor: Whether to include invasive metastatic carcinoma lesion.
        site_id: Clinical site identifier for domain shift.
        scanner_id: Scanner model ('scanner_aperio', 'scanner_philips', 'scanner_hamamatsu', etc.)
        add_pen_mark: Whether to superimpose a surgical pen marker artifact.
        add_blur: Whether to simulate optical defocus blur.
        seed: Random seed for deterministic reproducibility.

    Returns:
        (image_rgb_uint8, tumor_mask_bool, metadata)
    """
    w, h = dimensions
    rng = np.random.RandomState(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Scanner Optical Density & Color Profiles
    # ─────────────────────────────────────────────────────────────────────────
    if scanner_id == "scanner_philips" or site_id == "site_B":
        # Philips Ultra Fast: warmer tone, higher brightness, softer pink eosin
        bg_rgb = np.array([247, 245, 242], dtype=np.float32)
        stroma_rgb = np.array([230, 152, 186], dtype=np.float32)
        nuclei_rgb = np.array([84, 46, 134], dtype=np.float32)
        tumor_nuclei_rgb = np.array([50, 18, 92], dtype=np.float32)
        desmoplasia_rgb = np.array([215, 172, 198], dtype=np.float32)
    elif scanner_id == "scanner_hamamatsu" or site_id == "site_C":
        # Hamamatsu NanoZoomer: crisp neutral white balance, deeper violet-blue
        bg_rgb = np.array([241, 242, 244], dtype=np.float32)
        stroma_rgb = np.array([214, 140, 172], dtype=np.float32)
        nuclei_rgb = np.array([62, 34, 126], dtype=np.float32)
        tumor_nuclei_rgb = np.array([36, 12, 86], dtype=np.float32)
        desmoplasia_rgb = np.array([198, 160, 188], dtype=np.float32)
    elif scanner_id == "scanner_leica":
        # Leica Aperio GT450: saturated eosin, distinct warm hue
        bg_rgb = np.array([245, 242, 243], dtype=np.float32)
        stroma_rgb = np.array([225, 134, 170], dtype=np.float32)
        nuclei_rgb = np.array([72, 38, 120], dtype=np.float32)
        tumor_nuclei_rgb = np.array([42, 14, 84], dtype=np.float32)
        desmoplasia_rgb = np.array([205, 162, 192], dtype=np.float32)
    else:
        # Standard Site A / Aperio ScanScope CS2: classic high-contrast H&E
        bg_rgb = np.array([243, 243, 246], dtype=np.float32)
        stroma_rgb = np.array([218, 138, 178], dtype=np.float32)
        nuclei_rgb = np.array([76, 36, 122], dtype=np.float32)
        tumor_nuclei_rgb = np.array([44, 14, 84], dtype=np.float32)
        desmoplasia_rgb = np.array([202, 164, 192], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Glass Background with Realistic Microscope Vignetting
    # ─────────────────────────────────────────────────────────────────────────
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0

    # Slight condenser lens illumination gradient
    dist_center_sq = ((xx - cx) ** 2 + (yy - cy) ** 2) / (w * h / 2.0)
    vignette = 1.0 - 0.035 * dist_center_sq
    img = np.tile(bg_rgb.reshape(1, 1, 3), (h, w, 1)) * vignette[..., None]
    # Subtle sensor noise on glass
    img += rng.normal(0, 1.2, (h, w, 3))

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Organic Multi-Lobed Tissue Contours
    # ─────────────────────────────────────────────────────────────────────────
    theta = np.arctan2(yy - cy, xx - cx)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Multi-harmonic polar curve creates natural lobular tissue boundaries
    r_max = (min(h, w) * 0.38) * (
        1.0
        + 0.14 * np.sin(2.0 * theta + 0.4)
        + 0.09 * np.cos(3.0 * theta - 0.3)
        + 0.05 * np.sin(5.0 * theta + 1.2)
        + 0.03 * np.cos(7.0 * theta)
    )
    tissue_mask = r < r_max

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Fibrous Collagenous Stroma Matrix
    # ─────────────────────────────────────────────────────────────────────────
    # Directional collagen bundle micro-texture (striations in eosin)
    col_wave1 = np.sin(0.025 * xx + 0.018 * yy)
    col_wave2 = np.cos(0.014 * xx - 0.028 * yy)
    stroma_texture = 0.88 + 0.12 * (col_wave1 * col_wave2)
    stroma_pixels = stroma_rgb.reshape(1, 1, 3) * stroma_texture[..., None]
    img[tissue_mask] = np.clip(stroma_pixels[tissue_mask], 0, 255)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Histological Structures: Glands, Ducts, Lumens, and Adipose Vacuoles
    # ─────────────────────────────────────────────────────────────────────────
    t_idx = np.where(tissue_mask)
    if len(t_idx[0]) > 0:
        # Glandular duct lumens: circular clear spaces
        num_glands = max(8, int(min(w, h) * 0.015))
        gland_seed_idx = rng.choice(len(t_idx[0]), size=num_glands, replace=False)
        gland_y = t_idx[0][gland_seed_idx]
        gland_x = t_idx[1][gland_seed_idx]
        gland_radii = rng.uniform(min(w, h) * 0.012, min(w, h) * 0.028, size=num_glands)

        for gy, gx, gr in zip(gland_y, gland_x, gland_radii):
            g_dist = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
            # Empty lumen space (optically clear)
            lumen_mask = (g_dist < gr) & tissue_mask
            img[lumen_mask] = np.array([244, 243, 245], dtype=np.float32)
            # Palisading epithelial lining along lumen border
            lining_mask = (g_dist >= gr) & (g_dist < gr * 1.35) & tissue_mask
            img[lining_mask] = (stroma_rgb * 0.4 + nuclei_rgb * 0.6).astype(np.float32)

        # Adipose tissue: clusters of lipid vacuoles
        num_adipocytes = max(12, int(min(w, h) * 0.02))
        adipo_seed_idx = rng.choice(len(t_idx[0]), size=num_adipocytes, replace=False)
        for ay, ax in zip(t_idx[0][adipo_seed_idx], t_idx[1][adipo_seed_idx]):
            ar = rng.uniform(min(w, h) * 0.008, min(w, h) * 0.016)
            a_dist = np.sqrt((xx - ax) ** 2 + (yy - ay) ** 2)
            vacuole = (a_dist < ar) & tissue_mask
            img[vacuole] = np.array([246, 245, 244], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Normal Cellularity: Vesicular Basophilic Nuclei
    # ─────────────────────────────────────────────────────────────────────────
    if len(t_idx[0]) > 0:
        # Scatter normal stromal and lymphoid nuclei
        n_nuclei = min(25000, max(2000, len(t_idx[0]) // 4))
        chosen_n = rng.choice(len(t_idx[0]), size=n_nuclei, replace=False)
        ny = t_idx[0][chosen_n]
        nx = t_idx[1][chosen_n]

        # Nuclei vary in radius (1 to 2 pixels) with realistic chromatin density
        for y, x in zip(ny, nx):
            rad = 1 if rng.rand() < 0.7 else 2
            y_min, y_max = max(0, y - rad), min(h, y + rad + 1)
            x_min, x_max = max(0, x - rad), min(w, x + rad + 1)
            # Basophilic nucleus
            img[y_min:y_max, x_min:x_max] = nuclei_rgb

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Ground-Truth Invasive Carcinoma Lesion
    # ─────────────────────────────────────────────────────────────────────────
    tumor_mask = np.zeros((h, w), dtype=bool)
    if has_tumor:
        # Invasive tumor center located off-center in the tissue lobe
        t_cy, t_cx = h * 0.42, w * 0.42
        t_theta = np.arctan2(yy - t_cy, xx - t_cx)
        t_r = np.sqrt((xx - t_cx) ** 2 + (yy - t_cy) ** 2)

        # Infiltrative cribriform margin: irregular lobular tumor nest
        t_rmax = (min(h, w) * 0.17) * (
            1.0
            + 0.20 * np.sin(3.0 * t_theta + 0.5)
            + 0.15 * np.cos(4.0 * t_theta - 0.8)
            + 0.08 * np.sin(6.0 * t_theta)
        )
        tumor_mask = (t_r < t_rmax) & tissue_mask

        # Surrounding desmoplastic reactive stroma rim
        desmo_mask = (t_r >= t_rmax) & (t_r < t_rmax * 1.30) & tissue_mask
        img[desmo_mask] = desmoplasia_rgb

        # Inside tumor bed: hypercellular, hyperchromatic sheets of carcinoma cells
        img[tumor_mask] = (stroma_rgb * 0.35 + tumor_nuclei_rgb * 0.65)

        tumor_pts = np.where(tumor_mask)
        if len(tumor_pts[0]) > 0:
            # Dense pleomorphic, atypical carcinoma nuclei (dark violet, enlarged)
            n_tumor_cells = min(22000, len(tumor_pts[0]))
            chosen_tum = rng.choice(len(tumor_pts[0]), size=n_tumor_cells, replace=False)
            ty = tumor_pts[0][chosen_tum]
            tx = tumor_pts[1][chosen_tum]

            for y, x in zip(ty, tx):
                # Carcinoma nuclei are larger (2 to 3 px) and more pleomorphic
                t_rad = rng.choice([2, 3], p=[0.6, 0.4])
                y_min, y_max = max(0, y - t_rad), min(h, y + t_rad + 1)
                x_min, x_max = max(0, x - t_rad), min(w, x + t_rad + 1)
                img[y_min:y_max, x_min:x_max] = tumor_nuclei_rgb
                # Occasional prominent cherry nucleolus (reddish spot in center)
                if rng.rand() < 0.25:
                    img[y, x] = np.array([140, 25, 45], dtype=np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Controlled Histopathological Artifacts
    # ─────────────────────────────────────────────────────────────────────────
    # Surgical pen marker shortcut artifact
    if add_pen_mark:
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        # Curved surgical marker stroke with ink opacity gradient
        pen_color = (25, 55, 215)  # Surgical blue ink
        x_start, y_start = int(w * 0.18), int(h * 0.22)
        x_mid, y_mid = int(w * 0.28), int(h * 0.38)
        x_end, y_end = int(w * 0.38), int(h * 0.48)
        draw.line([(x_start, y_start), (x_mid, y_mid), (x_end, y_end)], fill=pen_color, width=14)
        img = np.array(pil_img)

    # Defocus blur artifact (simulating z-axis optical focal drift)
    if add_blur:
        img = gaussian_filter(img, sigma=[3.5, 3.5, 0]).astype(np.uint8)

    metadata = {
        "dimensions": dimensions,
        "mpp": 0.50 if scanner_id != "scanner_40x" else 0.25,
        "has_tumor": has_tumor,
        "site_id": site_id,
        "scanner_id": scanner_id,
        "has_pen_mark": add_pen_mark,
        "has_blur": add_blur,
        "seed": seed,
    }

    return img, tumor_mask, metadata


def save_synthetic_slide(
    output_path: Union[str, Path],
    dimensions: Tuple[int, int] = (2048, 2048),
    has_tumor: bool = True,
    site_id: str = "site_A",
    scanner_id: str = "scanner_aperio",
    add_pen_mark: bool = False,
    add_blur: bool = False,
    seed: int = 42,
) -> Path:
    """Generate and save a synthetic slide image to disk as TIFF or PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img, tumor_mask, meta = generate_synthetic_slide_data(
        dimensions=dimensions,
        has_tumor=has_tumor,
        site_id=site_id,
        scanner_id=scanner_id,
        add_pen_mark=add_pen_mark,
        add_blur=add_blur,
        seed=seed,
    )

    pil_img = Image.fromarray(img)
    pil_img.save(output_path)

    # Also save corresponding ground truth tumor mask if tumor exists
    mask_path = output_path.with_name(f"{output_path.stem}_tumor_mask.png")
    mask_pil = Image.fromarray((tumor_mask * 255).astype(np.uint8))
    mask_pil.save(mask_path)

    return output_path


class SyntheticWSIGenerator:
    """Batch generator for synthetic whole-slide cohorts across multiple sites."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        num_slides: int = 20,
        dimensions: Tuple[int, int] = (1024, 1024),
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.num_slides = num_slides
        self.dimensions = dimensions
        self.seed = seed

    def generate_all(self) -> List[Path]:
        """Generate cohort of synthetic slides with balanced classes and multi-site shifts."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        sites = ["site_A", "site_B", "site_C"]
        scanners = {
            "site_A": "scanner_aperio",
            "site_B": "scanner_philips",
            "site_C": "scanner_hamamatsu",
        }

        for i in range(self.num_slides):
            site = sites[i % len(sites)]
            scanner = scanners[site]
            has_tumor = (i % 2 == 1)
            add_pen = (i == 0)  # first slide has a pen mark artifact
            add_blur = (i == 1) # second slide has a blur artifact

            p = self.output_dir / f"slide_{i:03d}_{site}.png"
            save_synthetic_slide(
                output_path=p,
                dimensions=self.dimensions,
                has_tumor=has_tumor,
                site_id=site,
                scanner_id=scanner,
                add_pen_mark=add_pen,
                add_blur=add_blur,
                seed=self.seed + i,
            )
            paths.append(p)

        return paths
