from __future__ import annotations

from typing import Optional

from .config import CoreModelConfig


def get_config_for_preset(name: Optional[str]) -> CoreModelConfig:
    """
    Return a CoreModelConfig for the requested preset.

    Presets:
      - pi5 (default): small, Pi-friendly dims.
      - seed_local_2050: GPU-friendly local seed run (short seq, modest dims).
      - columnar_small: narrow columns / low-rank style for tiny footprints.
      - wave_only: emphasize wave/fast, shrink particle.
      - hybrid: balanced wave/particle with slightly wider context.
    """
    if not name:
        return CoreModelConfig.pi5_optimized()

    key = name.lower()
    if key in ("pi5", "pi5_optimized", "default"):
        return CoreModelConfig.pi5_optimized()

    if key in ("seed_local_2050", "rtx2050", "local_2050"):
        # Local GPU seed config: keep dims modest for small VRAM and quick iteration.
        return CoreModelConfig(
            d_s=256,
            d_w=256,
            d_p=128,
            d_e=256,
            d_k=64,
            d_c=256,
            num_levels=3,
            cms_sizes=[64, 128, 256],
            cms_dims=[128, 256, 512],
            cms_decays=[0.08, 0.05, 0.02],
            learning_rate=3e-4,
            gradient_clip=6.0,
            # Explicitly ensure Mamba-like wave is enabled for seed training.
            use_mamba_wave=True,
            mamba_state_dim=1,
        )

    if key == "columnar_small":
        return CoreModelConfig(
            d_s=96,
            d_w=96,
            d_p=32,
            d_e=96,
            d_k=32,
            d_c=96,
            num_levels=3,
            cms_sizes=[24, 48, 96],
            cms_dims=[64, 96, 128],
            cms_decays=[0.08, 0.05, 0.02],
            learning_rate=8e-4,
            gradient_clip=4.0,
        )

    if key == "wave_only":
        return CoreModelConfig(
            d_s=96,
            d_w=128,
            d_p=16,
            d_e=96,
            d_k=24,
            d_c=96,
            num_levels=3,
            cms_sizes=[24, 48, 96],
            cms_dims=[64, 96, 160],
            cms_decays=[0.08, 0.05, 0.02],
            learning_rate=1e-3,
            gradient_clip=4.0,
        )

    if key == "hybrid":
        return CoreModelConfig(
            d_s=144,
            d_w=160,
            d_p=64,
            d_e=144,
            d_k=40,
            d_c=160,
            num_levels=3,
            cms_sizes=[32, 64, 128],
            cms_dims=[96, 160, 224],
            cms_decays=[0.06, 0.04, 0.015],
            learning_rate=7e-4,
            gradient_clip=6.0,
        )

    # Fallback
    return CoreModelConfig.pi5_optimized()
