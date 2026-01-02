"""
Seed Model Configuration

Hardware-aware configuration that adapts to any platform.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging

from .hardware import HardwareProfile, Architecture, Accelerator, detect_hardware

logger = logging.getLogger(__name__)


@dataclass
class SeedConfig:
    """
    Configuration for the universal seed model.
    
    Automatically adapts to available hardware resources.
    """
    # Model dimensions (scaled by hardware)
    d_s: int = 64        # Fast state dimension
    d_w: int = 64        # Wave state dimension
    d_p: int = 32        # Particle state dimension
    d_e: int = 64        # Encoded input dimension
    d_k: int = 32        # Key dimension for CMS
    d_c: int = 64        # Context dimension
    
    # CMS hierarchy
    num_levels: int = 3
    cms_sizes: List[int] = field(default_factory=lambda: [64, 128, 256])
    cms_dims: List[int] = field(default_factory=lambda: [32, 64, 128])
    cms_decays: List[float] = field(default_factory=lambda: [0.9, 0.99, 0.999])
    
    # Mamba SSM
    use_mamba_wave: bool = True
    mamba_state_dim: int = 1
    
    # Training
    learning_rate: float = 1e-3
    gradient_clip: float = 10.0
    
    # Inference
    batch_size: int = 1
    use_jit: bool = True
    
    # Hardware target
    target_device: str = "auto"
    use_accelerator: bool = True
    
    @classmethod
    def for_hardware(cls, profile: HardwareProfile) -> "SeedConfig":
        """
        Create config optimized for specific hardware profile.
        
        Args:
            profile: Hardware profile from detection
        
        Returns:
            SeedConfig tuned for the hardware
        """
        # Base config
        config = cls()
        
        # Scale based on RAM
        if profile.ram_mb < 2000:
            # Very constrained (embedded)
            config = cls.embedded()
        elif profile.ram_mb < 4000:
            # Constrained (Pi 4, low-end edge)
            config = cls.minimal()
        elif profile.ram_mb < 16000:
            # Standard edge (Pi 5, Jetson Nano)
            config = cls.edge()
        elif profile.ram_mb < 64000:
            # Workstation / Jetson Orin
            config = cls.workstation()
        else:
            # Cloud / TPU
            config = cls.cloud()
        
        # Adjust for accelerators
        if Accelerator.TPU in profile.accelerators:
            config.batch_size = 8
            config.use_jit = True
        elif Accelerator.CUDA in profile.accelerators:
            config.batch_size = 4
            config.use_jit = True
        elif Accelerator.HAILO in profile.accelerators:
            config.use_accelerator = True
            config.batch_size = 1
        
        config.target_device = profile.device_class
        
        return config
    
    @classmethod
    def auto(cls) -> "SeedConfig":
        """Auto-detect hardware and return optimized config."""
        profile = detect_hardware()
        return cls.for_hardware(profile)
    
    @classmethod
    def embedded(cls) -> "SeedConfig":
        """Minimal config for very constrained devices (<2GB RAM)."""
        return cls(
            d_s=32, d_w=32, d_p=16, d_e=32, d_k=16, d_c=32,
            num_levels=2,
            cms_sizes=[16, 32],
            cms_dims=[16, 32],
            cms_decays=[0.9, 0.999],
            use_mamba_wave=False,  # Use simpler dynamics
            batch_size=1,
            use_jit=False,  # JIT can be slow to compile
            target_device="embedded",
        )
    
    @classmethod
    def minimal(cls) -> "SeedConfig":
        """Minimal config for constrained devices (2-4GB RAM)."""
        return cls(
            d_s=48, d_w=48, d_p=24, d_e=48, d_k=24, d_c=48,
            num_levels=3,
            cms_sizes=[32, 64, 128],
            cms_dims=[24, 48, 96],
            cms_decays=[0.9, 0.99, 0.999],
            batch_size=1,
            target_device="edge",
        )
    
    @classmethod
    def edge(cls) -> "SeedConfig":
        """Standard config for edge devices (Pi5, Jetson Nano, 4-16GB)."""
        return cls(
            d_s=64, d_w=64, d_p=32, d_e=64, d_k=32, d_c=64,
            num_levels=3,
            cms_sizes=[64, 128, 256],
            cms_dims=[32, 64, 128],
            cms_decays=[0.9, 0.99, 0.999],
            batch_size=1,
            target_device="edge",
        )
    
    @classmethod
    def workstation(cls) -> "SeedConfig":
        """Config for workstations / Jetson Orin (16-64GB)."""
        return cls(
            d_s=128, d_w=128, d_p=64, d_e=128, d_k=64, d_c=128,
            num_levels=3,
            cms_sizes=[128, 256, 512],
            cms_dims=[64, 128, 256],
            cms_decays=[0.9, 0.99, 0.999],
            batch_size=4,
            target_device="workstation",
        )
    
    @classmethod
    def cloud(cls) -> "SeedConfig":
        """Full config for cloud/TPU (64GB+)."""
        return cls(
            d_s=256, d_w=256, d_p=128, d_e=256, d_k=64, d_c=256,
            num_levels=4,
            cms_sizes=[256, 512, 1024, 2048],
            cms_dims=[128, 256, 512, 1024],
            cms_decays=[0.9, 0.95, 0.99, 0.999],
            batch_size=8,
            target_device="cloud",
        )
    
    # Convenience aliases
    @classmethod
    def pi5(cls) -> "SeedConfig":
        """Alias for edge() - Pi5 optimized."""
        return cls.edge()
    
    @classmethod
    def jetson(cls) -> "SeedConfig":
        """Alias for workstation() - Jetson Orin optimized."""
        return cls.workstation()
    
    @classmethod
    def tpu(cls) -> "SeedConfig":
        """Alias for cloud() - TPU optimized."""
        return cls.cloud()
    
    def param_count_estimate(self) -> int:
        """Estimate total parameter count for this config."""
        # Rough estimate based on dimensions
        encoder_params = self.d_e * 128 * 2  # Input encoder
        cms_params = sum(s * d for s, d in zip(self.cms_sizes, self.cms_dims))
        core_params = (self.d_s + self.d_w + self.d_p) * self.d_e * 4
        decoder_params = self.d_c * 128
        
        return encoder_params + cms_params + core_params + decoder_params
    
    def memory_estimate_mb(self) -> int:
        """Estimate memory usage in MB."""
        params = self.param_count_estimate()
        # Params (float32) + gradients + optimizer state
        param_memory = params * 4 * 3  # bytes
        
        # CMS memory
        cms_memory = sum(
            self.batch_size * s * d * 4 
            for s, d in zip(self.cms_sizes, self.cms_dims)
        )
        
        return (param_memory + cms_memory) // (1024 * 1024)

