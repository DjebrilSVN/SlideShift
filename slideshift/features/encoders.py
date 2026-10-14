"""Implementations of foundation model tile encoders."""

import os
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from slideshift.features.base import BaseFoundationEncoder
from slideshift.features.registry import get_model_info
from slideshift.core.logging import logger


class MockEncoder(BaseFoundationEncoder):
    """Deterministic mock encoder for offline testing, CI, and local demos."""

    def __init__(
        self,
        model_name: str = "mock",
        embedding_dim: int = 512,
        device: str = "cpu",
        use_fp16: bool = True,
    ):
        super().__init__(model_name=model_name, device=device, use_fp16=use_fp16)
        self._dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode_tiles(self, tiles: List[Image.Image]) -> np.ndarray:
        if not tiles:
            return np.zeros((0, self.embedding_dim), dtype=np.float16)

        embeddings = []
        for img in tiles:
            # Resize to standardized input size
            img_resized = img.resize(self.input_size, Image.Resampling.BILINEAR)
            arr = np.array(img_resized, dtype=np.float32) / 255.0

            # Color statistics
            mean_rgb = np.mean(arr, axis=(0, 1))
            std_rgb = np.std(arr, axis=(0, 1))

            # Nuclear density heuristic (dark violet pixels)
            dark_mask = (arr[..., 0] < 0.4) & (arr[..., 2] > arr[..., 1])
            nuclear_fraction = float(np.mean(dark_mask))

            # Deterministic projection vector
            # Seed pseudo-random basis using spatial hash of mean colors
            h_val = int(abs(mean_rgb[0] * 1000 + mean_rgb[2] * 500 + nuclear_fraction * 2000)) % 100000
            rng = np.random.RandomState(h_val)
            proj = rng.randn(self.embedding_dim).astype(np.float32)

            # Signal component: distinguish tumor-like tiles
            if nuclear_fraction > 0.15:
                proj[:64] += 2.5
            else:
                proj[:64] -= 1.0

            # Normalize to unit sphere (typical for SSL representations)
            norm = np.linalg.norm(proj) + 1e-7
            feat = proj / norm
            embeddings.append(feat)

        out = np.stack(embeddings, axis=0)
        return out.astype(np.float16 if self.use_fp16 else np.float32)


class ResNet50Encoder(BaseFoundationEncoder):
    """ImageNet-pretrained ResNet-50 baseline feature extractor."""

    def __init__(self, device: str = "cuda", use_fp16: bool = True):
        super().__init__(model_name="resnet50", device=device, use_fp16=use_fp16)
        self.model = None
        self.transform = None
        self._torch = None
        self._device_obj = None
        self._init_model()

    def _init_model(self):
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            weights = models.ResNet50_Weights.DEFAULT
            base_model = models.resnet50(weights=weights)
            # Remove the final classification layer, keeping the 2048-dim avgpool
            modules = list(base_model.children())[:-1]
            self.model = torch.nn.Sequential(*modules)
            self.model.eval()

            dev = torch.device(self.device if torch.cuda.is_available() and "cuda" in self.device else "cpu")
            self.model = self.model.to(dev)

            self.transform = T.Compose([
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._torch = torch
            self._device_obj = dev
        except Exception as e:
            logger.warning(f"Could not load PyTorch ResNet50: {e}. Falling back to MockEncoder.")
            self.model = None

    def encode_tiles(self, tiles: List[Image.Image]) -> np.ndarray:
        if self.model is None:
            mock = MockEncoder(embedding_dim=2048, use_fp16=self.use_fp16)
            return mock.encode_tiles(tiles)

        tensors = [self.transform(img) for img in tiles]
        batch = self._torch.stack(tensors, dim=0).to(self._device_obj)

        with self._torch.no_grad():
            if self.use_fp16 and self._device_obj.type == "cuda":
                with self._torch.amp.autocast("cuda"):
                    feats = self.model(batch).flatten(start_dim=1)
            else:
                feats = self.model(batch).flatten(start_dim=1)

            feats_np = feats.cpu().numpy()

        return feats_np.astype(np.float16 if self.use_fp16 else np.float32)


class HuggingFacePathologyEncoder(BaseFoundationEncoder):
    """Pathology foundation model loader (Phikon, UNI, CONCH, Virchow via Hugging Face/timm)."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_fp16: bool = True,
        hf_token: Optional[str] = None,
    ):
        super().__init__(model_name=model_name, device=device, use_fp16=use_fp16)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.model = None
        self.transform = None
        self._init_model()

    def _init_model(self):
        try:
            import torch
            import timm
            from transformers import AutoImageProcessor, AutoModel

            repo_id = self.info.hf_repo_id
            dev = torch.device(self.device if torch.cuda.is_available() and "cuda" in self.device else "cpu")
            self._device_obj = dev
            self._torch = torch

            if self.info.is_gated and not self.hf_token:
                logger.warning(
                    f"Model '{self.model_name}' is gated ({self.info.license_type}). "
                    f"Please provide an authorized Hugging Face token via `HF_TOKEN` env var or config."
                )

            # Phikon (Owkin ViT-B)
            if self.model_name == "phikon":
                self.processor = AutoImageProcessor.from_pretrained(repo_id, token=self.hf_token)
                self.model = AutoModel.from_pretrained(repo_id, token=self.hf_token).to(dev)
                self.model.eval()
            else:
                # Gated models (UNI, Virchow) loaded via timm or transformers
                try:
                    self.model = timm.create_model(
                        f"hf-hub:{repo_id}",
                        pretrained=True,
                        num_classes=0,
                        token=self.hf_token,
                    ).to(dev)
                    self.model.eval()
                except Exception as inner_e:
                    logger.warning(f"Failed to load {self.model_name} from Hugging Face hub: {inner_e}")
                    self.model = None

        except Exception as e:
            logger.warning(f"Could not initialize {self.model_name}: {e}. Mock fallback will be used.")
            self.model = None

    def encode_tiles(self, tiles: List[Image.Image]) -> np.ndarray:
        if self.model is None:
            mock = MockEncoder(embedding_dim=self.embedding_dim, use_fp16=self.use_fp16)
            return mock.encode_tiles(tiles)

        try:
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensors = [transform(img) for img in tiles]
            batch = self._torch.stack(tensors, dim=0).to(self._device_obj)

            with self._torch.no_grad():
                if hasattr(self.model, "forward_features"):
                    feats = self.model.forward_features(batch)
                    if hasattr(self.model, "global_pool"):
                        feats = self.model.global_pool(feats)
                    elif feats.ndim == 3:
                        feats = feats[:, 0]  # CLS token
                else:
                    feats = self.model(batch)
                    if hasattr(feats, "last_hidden_state"):
                        feats = feats.last_hidden_state[:, 0]

                feats_np = feats.cpu().numpy()
            return feats_np.astype(np.float16 if self.use_fp16 else np.float32)

        except Exception as e:
            logger.error(f"Inference error with {self.model_name}: {e}. Falling back to mock output.")
            mock = MockEncoder(embedding_dim=self.embedding_dim, use_fp16=self.use_fp16)
            return mock.encode_tiles(tiles)


class EncoderFactory:
    """Factory to instantiate foundation encoders with automatic fallback."""

    @staticmethod
    def create(
        model_name: str,
        device: str = "cuda",
        use_fp16: bool = True,
        hf_token: Optional[str] = None,
    ) -> BaseFoundationEncoder:
        clean_name = model_name.lower().strip()
        if clean_name == "mock":
            return MockEncoder(device=device, use_fp16=use_fp16)
        elif clean_name == "resnet50":
            return ResNet50Encoder(device=device, use_fp16=use_fp16)
        elif clean_name in ["phikon", "uni", "conch", "virchow"]:
            return HuggingFacePathologyEncoder(
                model_name=clean_name,
                device=device,
                use_fp16=use_fp16,
                hf_token=hf_token,
            )
        else:
            raise ValueError(f"Unknown foundation model name: {model_name}")

