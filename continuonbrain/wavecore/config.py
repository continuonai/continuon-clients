"""WaveCore Configuration."""
from dataclasses import dataclass
from typing import Optional

@dataclass
class WaveCoreConfig:
    """Hyperparameters for WaveCore models."""
    vocab_size: int = 512
    seq_len: int = 64
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    
    # Loop specific settings (Fast, Mid, Slow)
    loop_type: str = "mid"  # fast, mid, slow

    @staticmethod
    def fast_loop():
        """Latency-optimized config for reflex loops (50-100ms)."""
        return WaveCoreConfig(
            vocab_size=256,
            seq_len=32,
            d_model=64,
            n_layers=1,
            loop_type="fast"
        )

    @staticmethod
    def mid_loop():
        """Context-optimized config for tactical planning (1-10s)."""
        return WaveCoreConfig(
            vocab_size=1024,
            seq_len=128,
            d_model=128,
            n_layers=4,
            loop_type="mid"
        )

    @staticmethod
    def slow_loop():
        """Capacity-optimized config for strategic reasoning (Cloud/Offline)."""
        return WaveCoreConfig(
            vocab_size=4096,
            seq_len=512,
            d_model=256,
            n_layers=8,
            loop_type="slow"
        )
