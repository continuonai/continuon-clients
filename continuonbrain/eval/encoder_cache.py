"""
Encoder Cache Manager

Provides a shared cache for sentence-transformer encoders used in benchmarks.
Avoids reloading the large encoder model on every benchmark run.
"""

import os
from typing import Optional, Dict, Any

_ENCODER_CACHE: Dict[str, Any] = {}


def get_cached_encoder(
    model_name: str = 'google/embeddinggemma-300m',
    token: Optional[str] = None,
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
) -> Any:
    """
    Get a cached SentenceTransformer encoder or load if not cached.

    Args:
        model_name: HuggingFace model name or path
        token: HuggingFace API token for gated models
        trust_remote_code: Whether to trust remote code
        cache_dir: Optional cache directory for model files

    Returns:
        SentenceTransformer encoder instance (cached)
    """
    # Set cache directory
    if cache_dir:
        os.environ['HF_HOME'] = cache_dir
    elif 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = '/opt/continuonos/brain/hf_cache'

    # Create cache key
    cache_key = f"{model_name}:{token or 'none'}"

    if cache_key not in _ENCODER_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading encoder '{model_name}' (first load, will be cached)...")
            _ENCODER_CACHE[cache_key] = SentenceTransformer(
                model_name,
                trust_remote_code=trust_remote_code,
                token=token,
            )
            print(f"Encoder loaded and cached.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for encoder loading. "
                "Install with: pip install sentence-transformers"
            )
    else:
        print(f"Using cached encoder '{model_name}'")

    return _ENCODER_CACHE[cache_key]


def clear_encoder_cache() -> None:
    """Clear all cached encoders to free memory."""
    global _ENCODER_CACHE
    _ENCODER_CACHE.clear()
    print("Encoder cache cleared.")


def get_lightweight_encoder(obs_dim: int = 128) -> Any:
    """
    Get a lightweight encoder for quick tests (no external model loading).

    Returns a simple encoder that uses random projections instead of
    loading a large sentence-transformer model.

    Args:
        obs_dim: Dimension of observation vectors

    Returns:
        A callable that encodes text to vectors
    """
    import numpy as np

    class LightweightEncoder:
        """Fast encoder using hash-based projections for testing."""

        def __init__(self, dim: int = 128, seed: int = 42):
            self.dim = dim
            self.rng = np.random.default_rng(seed)

        def encode(self, texts, convert_to_numpy=True, **kwargs):
            """Encode texts to vectors using hash-based projection."""
            if isinstance(texts, str):
                texts = [texts]

            vectors = []
            for text in texts:
                # Create deterministic vector from text hash
                hash_bytes = hash(text).to_bytes(8, 'little', signed=True)
                seed = int.from_bytes(hash_bytes[:4], 'little')
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(self.dim).astype(np.float32)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                vectors.append(vec)

            result = np.stack(vectors, axis=0)
            return result if convert_to_numpy else result.tolist()

    return LightweightEncoder(dim=obs_dim)
