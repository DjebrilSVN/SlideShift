"""Slide feature extraction pipeline with streaming and HDF5 / PT caching."""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    h5py = None
    HAS_H5PY = False

from slideshift.core.slide import BaseSlideReader, SlideFactory
from slideshift.core.types import SlideBag, TileMetadata, QCReport
from slideshift.core.logging import logger
from slideshift.features.base import BaseFoundationEncoder
from slideshift.features.encoders import EncoderFactory
from slideshift.ingestion.tiling import SlideTiler


class SlideFeatureExtractor:
    """Streams tiles from a whole-slide image, encodes with foundation model, and saves feature bag."""

    def __init__(
        self,
        encoder: BaseFoundationEncoder,
        tiler: Optional[SlideTiler] = None,
        batch_size: int = 64,
        cache_format: str = "h5",
    ):
        self.encoder = encoder
        self.tiler = tiler or SlideTiler()
        self.batch_size = batch_size
        self.cache_format = cache_format.lower()

    def extract_slide_features(
        self,
        slide: BaseSlideReader,
        label: Optional[int] = None,
        patient_id: Optional[str] = None,
        site: Optional[str] = None,
        scanner: Optional[str] = None,
        show_progress: bool = True,
    ) -> SlideBag:
        """Tile the slide, run foundation model inference in batches, and construct SlideBag."""
        # 1. Tile slide
        tiles_meta, qc_report, thumb_rgb, tissue_mask = self.tiler.tile_slide(slide)
        logger.info(
            f"Slide {slide.slide_id}: {len(tiles_meta)} valid tissue tiles identified (tissue fraction: {qc_report.tissue_fraction*100:.1f}%)."
        )

        if not tiles_meta:
            logger.warning(f"No valid tissue patches found for slide {slide.slide_id}!")
            return SlideBag(
                slide_id=slide.slide_id,
                features=np.zeros((0, self.encoder.embedding_dim), dtype=np.float16),
                coordinates=np.zeros((0, 2), dtype=np.int32),
                label=label,
                patient_id=patient_id,
                site=site,
                scanner=scanner,
                qc_report=qc_report,
                tile_mpp=self.tiler.target_mpp,
            )

        # 2. Extract and encode in batches
        all_embeddings: List[np.ndarray] = []
        all_coords: List[Tuple[int, int]] = []

        batch_images = []
        batch_coords = []

        iterator = tiles_meta
        if show_progress:
            iterator = tqdm(tiles_meta, desc=f"Extracting {slide.slide_id}", leave=False)

        for meta in iterator:
            img = self.tiler.extract_tile_image(slide, meta)
            batch_images.append(img)
            batch_coords.append((meta.coord_x, meta.coord_y))

            if len(batch_images) >= self.batch_size:
                embs = self.encoder.encode_tiles(batch_images)
                all_embeddings.append(embs)
                all_coords.extend(batch_coords)
                batch_images = []
                batch_coords = []

        # Remaining batch
        if batch_images:
            embs = self.encoder.encode_tiles(batch_images)
            all_embeddings.append(embs)
            all_coords.extend(batch_coords)

        features = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, self.encoder.embedding_dim), dtype=np.float16)
        coordinates = np.array(all_coords, dtype=np.int32)

        return SlideBag(
            slide_id=slide.slide_id,
            features=features,
            coordinates=coordinates,
            label=label,
            patient_id=patient_id,
            site=site,
            scanner=scanner,
            qc_report=qc_report,
            tile_mpp=self.tiler.target_mpp,
            metadata={"model_name": self.encoder.model_name},
        )

    def save_bag(self, bag: SlideBag, output_dir: Union[str, Path]) -> Path:
        """Save extracted features, coordinates, and QC metadata in half-precision."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_format == "h5" and HAS_H5PY:
            out_file = out_dir / f"{bag.slide_id}.h5"
            with h5py.File(out_file, "w") as f:
                f.create_dataset("features", data=bag.features, compression="gzip", dtype="float16")
                f.create_dataset("coordinates", data=bag.coordinates, compression="gzip", dtype="int32")
                f.attrs["slide_id"] = bag.slide_id
                f.attrs["tile_mpp"] = bag.tile_mpp
                if bag.label is not None:
                    f.attrs["label"] = bag.label
                if bag.patient_id is not None:
                    f.attrs["patient_id"] = bag.patient_id
                if bag.site is not None:
                    f.attrs["site"] = bag.site
                if bag.scanner is not None:
                    f.attrs["scanner"] = bag.scanner
                if bag.qc_report is not None:
                    f.attrs["qc_json"] = json.dumps(bag.qc_report.to_dict())
                f.attrs["metadata_json"] = json.dumps(bag.metadata)
            return out_file

        elif self.cache_format == "pt":
            try:
                import torch
                HAS_TORCH = hasattr(torch, "__version__")
            except Exception:
                HAS_TORCH = False

            if HAS_TORCH:
                out_file = out_dir / f"{bag.slide_id}.pt"
                torch.save(
                    {
                        "features": torch.from_numpy(bag.features),
                        "coordinates": torch.from_numpy(bag.coordinates),
                        "slide_id": bag.slide_id,
                        "label": bag.label,
                        "patient_id": bag.patient_id,
                        "site": bag.site,
                        "scanner": bag.scanner,
                        "qc_report": bag.qc_report.to_dict() if bag.qc_report else None,
                        "metadata": bag.metadata,
                    },
                    out_file,
                )
                return out_file
            else:
                out_file = out_dir / f"{bag.slide_id}.npz"
                np.savez_compressed(
                    out_file,
                    features=bag.features,
                    coordinates=bag.coordinates,
                    slide_id=bag.slide_id,
                    label=bag.label if bag.label is not None else -1,
                    site=bag.site or "",
                    scanner=bag.scanner or "",
                )
                return out_file
        else:
            # Fallback numpy npz
            out_file = out_dir / f"{bag.slide_id}.npz"
            np.savez_compressed(
                out_file,
                features=bag.features,
                coordinates=bag.coordinates,
                slide_id=bag.slide_id,
                label=bag.label if bag.label is not None else -1,
                patient_id=bag.patient_id or "",
                site=bag.site or "",
                scanner=bag.scanner or "",
            )
            return out_file

    @staticmethod
    def load_bag(file_path: Union[str, Path]) -> SlideBag:
        """Load an extracted feature bag from HDF5, PT, or NPZ."""
        path = Path(file_path)
        if path.suffix.lower() in [".h5", ".hdf5"]:
            with h5py.File(path, "r") as f:
                features = f["features"][:]
                coordinates = f["coordinates"][:]
                slide_id = str(f.attrs.get("slide_id", path.stem))
                tile_mpp = float(f.attrs.get("tile_mpp", 0.50))
                label = int(f.attrs["label"]) if "label" in f.attrs else None
                patient_id = str(f.attrs["patient_id"]) if "patient_id" in f.attrs else None
                site = str(f.attrs["site"]) if "site" in f.attrs else None
                scanner = str(f.attrs["scanner"]) if "scanner" in f.attrs else None
                qc_report = None
                if "qc_json" in f.attrs:
                    q_data = json.loads(f.attrs["qc_json"])
                    qc_report = QCReport(**q_data)
                metadata = {}
                if "metadata_json" in f.attrs:
                    metadata = json.loads(f.attrs["metadata_json"])

                return SlideBag(
                    slide_id=slide_id,
                    features=features,
                    coordinates=coordinates,
                    label=label,
                    patient_id=patient_id,
                    site=site,
                    scanner=scanner,
                    qc_report=qc_report,
                    tile_mpp=tile_mpp,
                    metadata=metadata,
                )
        elif path.suffix.lower() == ".pt":
            import torch
            data = torch.load(path, map_location="cpu")
            qc_dict = data.get("qc_report")
            qc_rep = QCReport(**qc_dict) if qc_dict else None
            return SlideBag(
                slide_id=data["slide_id"],
                features=data["features"].numpy(),
                coordinates=data["coordinates"].numpy(),
                label=data.get("label"),
                patient_id=data.get("patient_id"),
                site=data.get("site"),
                scanner=data.get("scanner"),
                qc_report=qc_rep,
                metadata=data.get("metadata", {}),
            )
        else:
            data = np.load(path)
            return SlideBag(
                slide_id=str(data["slide_id"]),
                features=data["features"],
                coordinates=data["coordinates"],
                label=int(data["label"]) if data["label"] != -1 else None,
                patient_id=str(data["patient_id"]) if "patient_id" in data and str(data["patient_id"]) else None,
                site=str(data["site"]) if data["site"] else None,
                scanner=str(data["scanner"]) if data["scanner"] else None,
            )
