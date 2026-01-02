"""
Seed Model Scaling Configurations

Golden Rule: Must run on devices with <8GB RAM

Memory Budget (8GB device):
- Total RAM:        8.0 GB
- OS overhead:      1.5 GB
- Embeddings:       0.5 GB (EmbeddingGemma-300m)
- CMS memory:       0.3 GB
- JAX runtime:      0.5 GB
- Safety margin:    0.5 GB
- Available:        4.7 GB (~1.2B params float32, ~2.4B params float16)
"""

from dataclasses import dataclass
from typing import List, Tuple
from .config import CoreModelConfig


@dataclass
class ScalingTier:
    """Defines a scaling tier for the seed model."""
    version: str
    params_target: int
    d_s: int
    d_w: int
    d_p: int
    d_e: int
    d_k: int
    d_c: int
    cms_sizes: Tuple[int, ...]
    cms_dims: Tuple[int, ...]
    action_dim: int
    output_dim: int
    precision: str = "float32"
    target_device: str = "Pi 5 (8GB)"
    
    def to_config(self) -> CoreModelConfig:
        """Convert to CoreModelConfig."""
        return CoreModelConfig(
            d_s=self.d_s,
            d_w=self.d_w,
            d_p=self.d_p,
            d_e=self.d_e,
            d_k=self.d_k,
            d_c=self.d_c,
            num_levels=len(self.cms_sizes),
            cms_sizes=list(self.cms_sizes),
            cms_dims=list(self.cms_dims),
            cms_decays=[0.05, 0.03, 0.01][:len(self.cms_sizes)],
        )
    
    @property
    def memory_gb(self) -> float:
        """Estimated memory usage in GB."""
        bytes_per_param = 4 if self.precision == "float32" else 2
        return self.params_target * bytes_per_param / 1e9


# Scaling tiers - grow model over time while staying under 8GB
SCALING_TIERS = {
    "v2.0": ScalingTier(
        version="2.0.0",
        params_target=1_000_000,
        d_s=128, d_w=128, d_p=64, d_e=128, d_k=32, d_c=128,
        cms_sizes=(32, 64, 128),
        cms_dims=(64, 128, 256),
        action_dim=64,
        output_dim=64,
    ),
    "v3.0": ScalingTier(
        version="3.0.0",
        params_target=5_000_000,
        d_s=256, d_w=256, d_p=128, d_e=256, d_k=64, d_c=256,
        cms_sizes=(64, 128, 256),
        cms_dims=(128, 256, 512),
        action_dim=128,
        output_dim=128,
    ),
    "v4.0": ScalingTier(
        version="4.0.0",
        params_target=25_000_000,
        d_s=512, d_w=512, d_p=256, d_e=512, d_k=128, d_c=512,
        cms_sizes=(128, 256, 512),
        cms_dims=(256, 512, 1024),
        action_dim=256,
        output_dim=256,
    ),
    "v5.0": ScalingTier(
        version="5.0.0",
        params_target=100_000_000,
        d_s=1024, d_w=1024, d_p=512, d_e=1024, d_k=256, d_c=1024,
        cms_sizes=(256, 512, 1024),
        cms_dims=(512, 1024, 2048),
        action_dim=512,
        output_dim=512,
        precision="float16",
    ),
    "v6.0": ScalingTier(
        version="6.0.0",
        params_target=500_000_000,
        d_s=2048, d_w=2048, d_p=1024, d_e=2048, d_k=512, d_c=2048,
        cms_sizes=(512, 1024, 2048),
        cms_dims=(1024, 2048, 4096),
        action_dim=1024,
        output_dim=1024,
        precision="int8",
        target_device="8GB + Quantization",
    ),
}


def get_tier(version: str) -> ScalingTier:
    """Get scaling tier by version."""
    return SCALING_TIERS.get(version, SCALING_TIERS["v3.0"])


def get_next_tier(current_version: str) -> ScalingTier:
    """Get the next scaling tier after current version."""
    versions = list(SCALING_TIERS.keys())
    try:
        idx = versions.index(current_version)
        if idx < len(versions) - 1:
            return SCALING_TIERS[versions[idx + 1]]
    except ValueError:
        pass
    return SCALING_TIERS[versions[-1]]


def estimate_memory(param_count: int, precision: str = "float32") -> float:
    """Estimate memory usage in GB."""
    bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}
    return param_count * bytes_per_param.get(precision, 4) / 1e9


def fits_device(param_count: int, device_ram_gb: float = 8.0, precision: str = "float32") -> bool:
    """Check if model fits on device."""
    overhead_gb = 3.3  # OS + embeddings + CMS + JAX + margin
    available_gb = device_ram_gb - overhead_gb
    model_gb = estimate_memory(param_count, precision)
    return model_gb <= available_gb

