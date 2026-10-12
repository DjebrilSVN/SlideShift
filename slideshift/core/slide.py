"""Whole-Slide Image (WSI) reading abstraction with robust MPP calibration."""

from abc import ABC, abstractmethod
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

# Graceful OpenSlide import (Windows safe)
try:
    import openslide
    HAS_OPENSLIDE = True
except (ImportError, OSError):
    openslide = None
    HAS_OPENSLIDE = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    tifffile = None
    HAS_TIFFFILE = False


class BaseSlideReader(ABC):
    """Abstract base class for whole-slide image readers."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.slide_id = self.path.stem
        self._dimensions: Tuple[int, int] = (0, 0)
        self._mpp: Optional[float] = None
        self._level_downsamples: Tuple[float, ...] = (1.0,)
        self._level_dimensions: Tuple[Tuple[int, int], ...] = ((0, 0),)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """(width, height) at baseline Level 0."""
        return self._dimensions

    @property
    def level_count(self) -> int:
        return len(self._level_dimensions)

    @property
    def level_dimensions(self) -> Tuple[Tuple[int, int], ...]:
        return self._level_dimensions

    @property
    def level_downsamples(self) -> Tuple[float, ...]:
        return self._level_downsamples

    @property
    def mpp(self) -> float:
        """Physical resolution in microns per pixel (defaulting to 0.5 if metadata missing)."""
        if self._mpp is not None and self._mpp > 0:
            return self._mpp
        return 0.50

    @abstractmethod
    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        """Read a rectangular region from the slide as an RGB PIL Image.
        
        Args:
            location: (x, y) tuple giving top-left pixel coordinates at Level 0.
            level: The pyramid level to sample from.
            size: (width, height) tuple of the region at the requested level.
        """
        pass

    @abstractmethod
    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        """Return a thumbnail RGB PIL Image fitting within `size`."""
        pass

    def compute_patch_extraction_scale(self, target_mpp: float) -> Tuple[int, float]:
        """Compute the best pyramid level and downsample factor for a target physical MPP.
        
        This prevents the common bug where scanner magnification is assumed to be 20x
        without verifying microns-per-pixel.
        
        Returns:
            (best_level, extra_scaling_factor)
        """
        current_mpp = self.mpp
        target_downsample = target_mpp / current_mpp

        best_level = 0
        for lvl, ds in enumerate(self.level_downsamples):
            if ds <= target_downsample:
                best_level = lvl
            else:
                break

        actual_level_ds = self.level_downsamples[best_level]
        remaining_scale = target_downsample / actual_level_ds
        return best_level, remaining_scale

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class OpenSlideReader(BaseSlideReader):
    """Slide reader powered by OpenSlide C-library."""

    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        if not HAS_OPENSLIDE:
            raise RuntimeError(
                "openslide-python is not installed or OpenSlide C libraries are missing."
            )
        self._osr = openslide.OpenSlide(str(self.path))
        self._dimensions = self._osr.dimensions
        self._level_dimensions = self._osr.level_dimensions
        self._level_downsamples = self._osr.level_downsamples
        self._mpp = self._extract_mpp()

    def _extract_mpp(self) -> Optional[float]:
        props = self._osr.properties
        # Check standard OpenSlide MPP keys
        for key in ["openslide.mpp-x", "tiff.XResolution", "aperio.MPP"]:
            if key in props:
                try:
                    val = float(props[key])
                    if key == "tiff.XResolution" and val > 0:
                        # Convert resolution (pixels per cm/inch) to microns per pixel
                        unit = props.get("tiff.ResolutionUnit", "centimeter").lower()
                        if "inch" in unit:
                            return 25400.0 / val
                        elif "centimeter" in unit or "cm" in unit:
                            return 10000.0 / val
                    return val
                except (ValueError, ZeroDivisionError):
                    continue
        return None

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        rgba = self._osr.read_region(location, level, size)
        return rgba.convert("RGB")

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        thumb = self._osr.get_thumbnail(size)
        return thumb.convert("RGB")

    def close(self) -> None:
        if hasattr(self, "_osr") and self._osr:
            self._osr.close()


class SyntheticSlideReader(BaseSlideReader):
    """In-memory synthetic WSI reader for tests, CI, and local experimentation.
    
    Generates realistic H&E stained synthetic histology patterns with tissue regions,
    optional tumor foci, background glass, and artifacts.
    """

    def __init__(
        self,
        path: Union[str, Path] = "synthetic_slide.svs",
        dimensions: Tuple[int, int] = (4096, 4096),
        mpp: float = 0.50,
        has_tumor: bool = True,
        seed: int = 42,
    ):
        super().__init__(path)
        self._dimensions = dimensions
        self._mpp = mpp
        self.has_tumor = has_tumor
        self.seed = seed
        self._level_dimensions = (
            dimensions,
            (dimensions[0] // 4, dimensions[1] // 4),
            (dimensions[0] // 16, dimensions[1] // 16),
            (dimensions[0] // 32, dimensions[1] // 32),
        )
        self._level_downsamples = (1.0, 4.0, 16.0, 32.0)
        self._cached_base_image: Optional[np.ndarray] = None

    def _generate_synthetic_image(self) -> np.ndarray:
        """Create a synthetic histology image in RGB uint8."""
        if self._cached_base_image is not None:
            return self._cached_base_image

        w, h = self._dimensions
        rng = np.random.RandomState(self.seed)

        # Background: glass slide (almost white with slight noise)
        img = np.full((h, w, 3), 242, dtype=np.uint8)
        img += rng.randint(-3, 4, size=(h, w, 3), dtype=np.int16).clip(-5, 5).astype(np.uint8)

        # Tissue ellipse in center
        yy, xx = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius_y, radius_x = int(h * 0.38), int(w * 0.38)
        tissue_mask = ((xx - center_x) ** 2 / radius_x ** 2 + (yy - center_y) ** 2 / radius_y ** 2) <= 1.0

        # Base Stroma/Eosin color: Pinkish (~220, 150, 180)
        eosin = np.array([215, 145, 175], dtype=np.float32)
        # Nuclei/Hematoxylin color: Deep Purple/Violet (~80, 50, 130)
        hematoxylin = np.array([80, 45, 125], dtype=np.float32)

        # Fill normal tissue
        img[tissue_mask] = eosin.astype(np.uint8)

        # Add stroma texture and simulated cell nuclei
        normal_idx = np.where(tissue_mask)
        if len(normal_idx[0]) > 0:
            num_cells = min(20000, len(normal_idx[0]) // 4)
            chosen = rng.choice(len(normal_idx[0]), size=num_cells, replace=False)
            cy = normal_idx[0][chosen]
            cx = normal_idx[1][chosen]
            # Stamp nuclei
            for y, x in zip(cy, cx):
                y_min, y_max = max(0, y - 1), min(h, y + 2)
                x_min, x_max = max(0, x - 1), min(w, x + 2)
                img[y_min:y_max, x_min:x_max] = hematoxylin.astype(np.uint8)

        # Add tumor island if specified (denser, darker nuclei, atypical morphology)
        if self.has_tumor:
            tumor_cy, tumor_cx = int(h * 0.45), int(w * 0.45)
            tumor_ry, tumor_rx = int(h * 0.15), int(w * 0.15)
            tumor_mask = ((xx - tumor_cx) ** 2 / tumor_rx ** 2 + (yy - tumor_cy) ** 2 / tumor_ry ** 2) <= 1.0
            tumor_idx = np.where(tumor_mask)
            if len(tumor_idx[0]) > 0:
                # Darker baseline
                img[tumor_mask] = (eosin * 0.75 + hematoxylin * 0.25).astype(np.uint8)
                # Dense hyperchromatic tumor nuclei
                num_tumor_cells = min(15000, len(tumor_idx[0]))
                chosen_t = rng.choice(len(tumor_idx[0]), size=num_tumor_cells, replace=False)
                for y, x in zip(tumor_idx[0][chosen_t], tumor_idx[1][chosen_t]):
                    y_min, y_max = max(0, y - 2), min(h, y + 3)
                    x_min, x_max = max(0, x - 2), min(w, x + 3)
                    img[y_min:y_max, x_min:x_max] = np.array([55, 30, 95], dtype=np.uint8)

        self._cached_base_image = img
        return self._cached_base_image

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        full_img = self._generate_synthetic_image()
        x0, y0 = location
        w, h = size
        ds = self.level_downsamples[level]
        x1 = int(x0 + w * ds)
        y1 = int(y0 + h * ds)

        # Crop from level 0
        img_h, img_w, _ = full_img.shape
        x0_c = max(0, min(img_w, x0))
        y0_c = max(0, min(img_h, y0))
        x1_c = max(0, min(img_w, x1))
        y1_c = max(0, min(img_h, y1))

        crop = full_img[y0_c:y1_c, x0_c:x1_c]
        if crop.size == 0:
            pil_img = Image.new("RGB", size, (242, 242, 242))
        else:
            pil_img = Image.fromarray(crop).resize(size, Image.Resampling.BILINEAR)
        return pil_img

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        full_img = self._generate_synthetic_image()
        pil_img = Image.fromarray(full_img)
        pil_img.thumbnail(size, Image.Resampling.BILINEAR)
        return pil_img

    def close(self) -> None:
        self._cached_base_image = None


class GenericImageReader(BaseSlideReader):
    """Fallback reader for standard image formats (TIFF, PNG, JPEG) via Pillow / Tifffile."""

    def __init__(self, path: Union[str, Path], mpp: float = 0.50):
        super().__init__(path)
        self._mpp = mpp
        if HAS_TIFFFILE and self.path.suffix.lower() in [".tif", ".tiff"]:
            with tifffile.TiffFile(str(self.path)) as tif:
                page = tif.pages[0]
                h, w = page.shape[:2]
                self._dimensions = (w, h)
        else:
            with Image.open(self.path) as img:
                self._dimensions = img.size

        w, h = self._dimensions
        self._level_dimensions = ((w, h), (w // 4, h // 4), (w // 16, h // 16))
        self._level_downsamples = (1.0, 4.0, 16.0)
        self._pil_img: Optional[Image.Image] = None

    def _get_image(self) -> Image.Image:
        if self._pil_img is None:
            self._pil_img = Image.open(self.path).convert("RGB")
        return self._pil_img

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        img = self._get_image()
        x0, y0 = location
        w, h = size
        ds = self.level_downsamples[level]
        x1 = int(x0 + w * ds)
        y1 = int(y0 + h * ds)
        crop = img.crop((x0, y0, x1, y1))
        return crop.resize(size, Image.Resampling.BILINEAR)

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        img = self._get_image().copy()
        img.thumbnail(size, Image.Resampling.BILINEAR)
        return img

    def close(self) -> None:
        if self._pil_img:
            self._pil_img.close()
            self._pil_img = None


class SlideFactory:
    """Factory for instantiating the appropriate slide reader."""

    @staticmethod
    def open(
        path_or_spec: Union[str, Path],
        mpp: Optional[float] = None,
        synthetic_dims: Tuple[int, int] = (4096, 4096),
        has_tumor: bool = True,
        seed: int = 42,
    ) -> BaseSlideReader:
        str_path = str(path_or_spec)
        if str_path.startswith("synthetic:") or str_path == "synthetic":
            return SyntheticSlideReader(
                path=str_path,
                dimensions=synthetic_dims,
                mpp=mpp or 0.50,
                has_tumor=has_tumor,
                seed=seed,
            )

        path = Path(path_or_spec)
        suffix = path.suffix.lower()

        # Try OpenSlide first if available and file format is a pyramid
        if HAS_OPENSLIDE and suffix in [".svs", ".ndpi", ".mrxs", ".vms", ".vmu", ".scn", ".tif", ".tiff"]:
            try:
                reader = OpenSlideReader(path)
                if mpp is not None:
                    reader._mpp = mpp
                return reader
            except Exception:
                pass

        # Fallback to generic PIL / Tiff reader
        return GenericImageReader(path, mpp=mpp or 0.50)

