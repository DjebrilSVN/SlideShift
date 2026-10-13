"""Physical resolution tiling engine for whole slide images."""

from typing import Generator, List, Optional, Tuple
import numpy as np
from PIL import Image
from slideshift.core.slide import BaseSlideReader
from slideshift.core.types import TileMetadata, QCReport
from slideshift.ingestion.tissue_mask import segment_tissue
from slideshift.ingestion.quality_control import assess_slide_quality


class SlideTiler:
    """Extracts non-overlapping or overlapping tiles at calibrated physical resolution (µm/pixel)."""

    def __init__(
        self,
        target_mpp: float = 0.50,
        tile_size_px: int = 256,
        min_tissue_fraction: float = 0.20,
        overlap_px: int = 0,
        thumbnail_size: Tuple[int, int] = (1024, 1024),
    ):
        self.target_mpp = target_mpp
        self.tile_size_px = tile_size_px
        self.min_tissue_fraction = min_tissue_fraction
        self.overlap_px = overlap_px
        self.thumbnail_size = thumbnail_size

    def tile_slide(
        self,
        slide: BaseSlideReader,
    ) -> Tuple[List[TileMetadata], QCReport, np.ndarray, np.ndarray]:
        """Process slide, perform quality control, and determine valid tile coordinates.

        Returns:
            (valid_tiles_metadata, qc_report, thumbnail_rgb, tissue_mask)
        """
        # 1. Generate thumbnail
        thumb_pil = slide.get_thumbnail(self.thumbnail_size)
        thumb_rgb = np.array(thumb_pil)

        # 2. Segment tissue
        tissue_mask, tissue_fraction = segment_tissue(thumb_rgb)

        # 3. Quality control check
        qc_report = assess_slide_quality(
            slide_id=slide.slide_id,
            thumbnail_rgb=thumb_rgb,
            tissue_mask=tissue_mask,
            tissue_fraction=tissue_fraction,
        )

        # 4. Compute physical tile size in Level 0 native pixels
        # target_mpp / native_mpp gives the ratio of pixel spacing
        native_mpp = slide.mpp
        ratio = self.target_mpp / native_mpp
        # Physical width/height in Level 0 pixels needed to produce tile_size_px at target_mpp
        step_l0 = int(round((self.tile_size_px - self.overlap_px) * ratio))
        tile_w_l0 = int(round(self.tile_size_px * ratio))
        tile_h_l0 = tile_w_l0

        w_l0, h_l0 = slide.dimensions
        thumb_w, thumb_h = thumb_pil.size
        scale_x = thumb_w / w_l0
        scale_y = thumb_h / h_l0

        tiles: List[TileMetadata] = []
        tile_idx = 0

        # Grid iteration over Level 0 coordinates
        for y0 in range(0, max(1, h_l0 - tile_h_l0 + 1), max(1, step_l0)):
            for x0 in range(0, max(1, w_l0 - tile_w_l0 + 1), max(1, step_l0)):
                # Map Level 0 tile rectangle to thumbnail mask coordinates
                tx0 = int(round(x0 * scale_x))
                ty0 = int(round(y0 * scale_y))
                tx1 = max(tx0 + 1, int(round((x0 + tile_w_l0) * scale_x)))
                ty1 = max(ty0 + 1, int(round((y0 + tile_h_l0) * scale_y)))

                # Clamp to thumbnail dimensions
                tx0_c = max(0, min(thumb_w, tx0))
                ty0_c = max(0, min(thumb_h, ty0))
                tx1_c = max(0, min(thumb_w, tx1))
                ty1_c = max(0, min(thumb_h, ty1))

                mask_patch = tissue_mask[ty0_c:ty1_c, tx0_c:tx1_c]
                if mask_patch.size == 0:
                    continue

                tissue_coverage = float(np.mean(mask_patch))
                if tissue_coverage >= self.min_tissue_fraction:
                    tile_meta = TileMetadata(
                        tile_id=f"{slide.slide_id}_t{tile_idx}_{x0}_{y0}",
                        slide_id=slide.slide_id,
                        coord_x=x0,
                        coord_y=y0,
                        tile_size_px=self.tile_size_px,
                        target_mpp=self.target_mpp,
                        native_mpp=native_mpp,
                        tissue_content=tissue_coverage,
                    )
                    tiles.append(tile_meta)
                    tile_idx += 1

        return tiles, qc_report, thumb_rgb, tissue_mask

    def extract_tile_image(
        self,
        slide: BaseSlideReader,
        tile_meta: TileMetadata,
    ) -> Image.Image:
        """Extract a single patch from the slide at calibrated physical resolution."""
        best_level, extra_scaling = slide.compute_patch_extraction_scale(self.target_mpp)
        ds = slide.level_downsamples[best_level]

        # Region size at best_level
        # Size at level 0: tile_size_px * (target_mpp / native_mpp)
        l0_size = int(round(self.tile_size_px * (self.target_mpp / slide.mpp)))
        region_w = max(1, int(round(l0_size / ds)))
        region_h = region_w

        img = slide.read_region(
            location=(tile_meta.coord_x, tile_meta.coord_y),
            level=best_level,
            size=(region_w, region_h),
        )

        # Final resize to standard patch size (e.g. 256x256)
        if img.size != (self.tile_size_px, self.tile_size_px):
            img = img.resize((self.tile_size_px, self.tile_size_px), Image.Resampling.BILINEAR)

        return img

    def stream_tiles(
        self,
        slide: BaseSlideReader,
        tiles: List[TileMetadata],
    ) -> Generator[Tuple[TileMetadata, Image.Image], None, None]:
        """Generator yielding (TileMetadata, PIL.Image) one at a time."""
        for meta in tiles:
            img = self.extract_tile_image(slide, meta)
            yield meta, img

