"""
Optional Hailo feature extractor for on-device embeddings.

This module is intentionally lightweight and safe to import when Hailo is absent:
- If the Hailo SDK/runtime isn't installed, this module should still import.
- The real inference wiring can be filled in when the Hailo pipeline is finalized.

For now, we provide a deterministic, dependency-light fallback embedding that
keeps the interface stable for the recorder/trainer regime.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class HailoFeatureExtractor:
    """
    Placeholder extractor with a stable interface:
      - embed_rgb(rgb_bgr) -> List[float] of length output_dim

    When Hailo is wired, this class should:
      - load a HEF model or a Hailo-accelerated pipeline
      - run inference to obtain a compact embedding vector
    """

    output_dim: int = 128

    def __post_init__(self) -> None:
        if int(self.output_dim) <= 0:
            raise ValueError("output_dim must be > 0")

    def embed_rgb(self, *, rgb_bgr: np.ndarray) -> List[float]:
        """
        Deterministic, cheap embedding fallback:
        - hash of image bytes -> pseudo-random vector in [-1, 1]
        - mixed with simple statistics
        """
        data = rgb_bgr.tobytes()
        seed = int(hashlib.sha256(data).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.uniform(-1.0, 1.0, size=(int(self.output_dim),)).astype(np.float32)

        arr = rgb_bgr.astype(np.float32)
        mean = float(arr.mean() / 255.0) if arr.size else 0.0
        std = float(arr.std() / 255.0) if arr.size else 0.0
        vec[0] = mean
        if vec.size > 1:
            vec[1] = std
        return vec.tolist()


