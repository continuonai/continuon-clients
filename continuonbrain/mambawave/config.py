"""MambaWave Configuration.

Combines WaveCore spectral settings with Mamba SSM parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class MambaWaveConfig:
    """Unified configuration for MambaWave architecture.

    Combines:
    - WaveCore spectral parameters (FFT, decay, filters)
    - Mamba SSM parameters (state dim, expand, dt)
    - Training loop variants (fast/mid/slow)
    """

    # Model dimensions
    vocab_size: int = 512
    d_model: int = 128
    seq_len: int = 64
    n_layers: int = 4

    # Spectral (WaveCore heritage)
    use_spectral: bool = True
    spectral_decay_init: float = 0.01
    n_fft_bins: Optional[int] = None  # None = auto from seq_len

    # SSM (Mamba heritage)
    use_ssm: bool = True
    ssm_state_dim: int = 16  # N in Mamba
    ssm_expand: int = 2  # E expansion factor
    ssm_dt_rank: str = "auto"  # or int
    ssm_dt_min: float = 0.001
    ssm_dt_max: float = 0.1
    ssm_dt_init: str = "random"  # "random" or "constant"
    ssm_conv_dim: int = 4  # Local convolution kernel size

    # Attention (optional hybrid)
    use_attention: bool = False
    n_heads: int = 4
    window_size: int = 0  # 0 = full attention, >0 = sliding window

    # Training
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 16
    device: str = "cpu"

    # Loop type for HOPE architecture
    loop_type: Literal["fast", "mid", "slow", "seed"] = "seed"

    # World model settings
    joint_dim: int = 6  # Robot arm joints
    joint_limit: float = 1.0

    @classmethod
    def fast_loop(cls) -> "MambaWaveConfig":
        """Latency-optimized variant for real-time inference.

        - Smaller model (64-dim)
        - Single layer
        - SSM-only (no attention)
        - Spectral for efficient filtering
        """
        return cls(
            d_model=64,
            n_layers=1,
            use_spectral=True,
            use_ssm=True,
            use_attention=False,
            ssm_state_dim=8,
            ssm_expand=1,
            loop_type="fast",
        )

    @classmethod
    def mid_loop(cls) -> "MambaWaveConfig":
        """Context-optimized variant for learning loops.

        - Medium model (128-dim)
        - 4 layers
        - Full MambaWave (SSM + Spectral)
        - Optional attention
        """
        return cls(
            d_model=128,
            n_layers=4,
            use_spectral=True,
            use_ssm=True,
            use_attention=False,
            ssm_state_dim=16,
            ssm_expand=2,
            loop_type="mid",
        )

    @classmethod
    def slow_loop(cls) -> "MambaWaveConfig":
        """Capacity-optimized variant for batch training.

        - Larger model (256-dim)
        - 8 layers
        - Full hybrid (SSM + Spectral + Attention)
        """
        return cls(
            d_model=256,
            n_layers=8,
            use_spectral=True,
            use_ssm=True,
            use_attention=True,
            n_heads=8,
            ssm_state_dim=32,
            ssm_expand=2,
            loop_type="slow",
        )

    @classmethod
    def default(cls) -> "MambaWaveConfig":
        """Default balanced configuration."""
        return cls.mid_loop()

    def get_ssm_dt_rank(self) -> int:
        """Compute dt_rank based on config."""
        if self.ssm_dt_rank == "auto":
            return max(1, self.d_model // 16)
        return int(self.ssm_dt_rank)

    def get_ssm_inner_dim(self) -> int:
        """Compute inner dimension for SSM."""
        return self.d_model * self.ssm_expand
