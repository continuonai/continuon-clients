from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class WaveCoreConfig:
    """Hyperparameters for WaveCore spectral models."""
    vocab_size: int = 512
    d_model: int = 128
    seq_len: int = 64
    n_layers: int = 2
    n_heads: int = 4
    decay_init: float = 0.01
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 16
    device: str = "cpu"
    
    # HOPE Loop variants
    loop_type: str = "seed" # "fast", "mid", "slow", "seed"

    @classmethod
    def fast_loop(cls) -> WaveCoreConfig:
        """Latency-optimized variant."""
        return cls(d_model=64, n_layers=1, n_heads=2, loop_type="fast")

    @classmethod
    def mid_loop(cls) -> WaveCoreConfig:
        """Context-optimized variant."""
        return cls(d_model=128, n_layers=4, n_heads=4, loop_type="mid")

    @classmethod
    def slow_loop(cls) -> WaveCoreConfig:
        """Capacity-optimized variant."""
        return cls(d_model=256, n_layers=8, n_heads=8, loop_type="slow")