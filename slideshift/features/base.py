"""Abstract base class for foundation model tile encoders."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import numpy as np
from PIL import Image
from slideshift.features.registry import ModelInfo, get_model_info


class BaseFoundationEncoder(ABC):
    """Base class for pathology foundation model feature extractors."""

    def __init__(self, model_name: str, device: str = "cuda", use_fp16: bool = True):
        self.model_name = model_name.lower().strip()
        self.info: ModelInfo = get_model_info(self.model_name)
        self.device = device
        self.use_fp16 = use_fp16

    @property
    def embedding_dim(self) -> int:
        return self.info.embedding_dim

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.info.input_size

    @abstractmethod
    def encode_tiles(self, tiles: List[Image.Image]) -> np.ndarray:
        """Encode a batch of PIL Image tiles into float16 / float32 embedding vectors.

        Args:
            tiles: List of RGB PIL Images.

        Returns:
            np.ndarray of shape (len(tiles), embedding_dim) in float16.
        """
        pass

