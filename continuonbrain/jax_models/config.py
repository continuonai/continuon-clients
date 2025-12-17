"""
CoreModel Configuration

Configuration management for JAX/Flax CoreModel with sensible defaults
optimized for both Pi development and cloud TPU training.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CoreModelConfig:
    """
    Configuration for CoreModel (JAX/Flax implementation).
    
    Architecture inspired by HOPE looping system:
    - Fast state (s_t): low-level reactive state
    - Wave state (w_t): SSM-like global coordination  
    - Particle state (p_t): local nonlinear dynamics
    - CMS memory integration (hierarchical memory levels)
    
    Dimensions:
        d_s: Fast state dimension
        d_w: Wave state dimension
        d_p: Particle state dimension
        d_e: Encoded input dimension
        d_k: Key dimension for CMS attention
        d_c: Mixed context dimension
    
    CMS Hierarchy:
        num_levels: Number of CMS levels (0=episodic, L=semantic)
        cms_sizes: Number of slots per level [N_0, N_1, ..., N_L]
        cms_dims: Dimension per level [d_0, d_1, ..., d_L]
        cms_decays: Decay coefficient per level [d_0, d_1, ..., d_L]
    
    Training:
        learning_rate: Base learning rate
        gradient_clip: Gradient clipping threshold
    """
    
    # Dimensions
    d_s: int = 256
    d_w: int = 256
    d_p: int = 128
    d_e: int = 256
    d_k: int = 64
    d_c: int = 256
    
    # CMS hierarchy (3 levels: episodic, working, semantic)
    num_levels: int = 3
    cms_sizes: List[int] = field(default_factory=lambda: [64, 128, 256])
    cms_dims: List[int] = field(default_factory=lambda: [128, 256, 512])
    cms_decays: List[float] = field(default_factory=lambda: [0.1, 0.05, 0.01])
    
    # Training
    learning_rate: float = 1e-3
    gradient_clip: float = 10.0
    
    # Stability
    use_layer_norm: bool = True
    state_saturation_limit: float = 10.0
    
    # Architecture options
    obs_type: str = "vector"  # "vector" or "image"
    output_type: str = "continuous"  # "continuous" or "discrete"

    # Mamba-like selective SSM options (seed-safe defaults)
    use_mamba_wave: bool = True
    mamba_state_dim: int = 1
    mamba_dt_min: float = 1e-4
    mamba_dt_scale: float = 1.0
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.num_levels == len(self.cms_sizes), \
            f"num_levels ({self.num_levels}) must match cms_sizes length ({len(self.cms_sizes)})"
        assert self.num_levels == len(self.cms_dims), \
            f"num_levels ({self.num_levels}) must match cms_dims length ({len(self.cms_dims)})"
        assert self.num_levels == len(self.cms_decays), \
            f"num_levels ({self.num_levels}) must match cms_dims length ({len(self.cms_decays)})"
        
        # Validate decay coefficients
        for i, d in enumerate(self.cms_decays):
            assert 0 < d < 1, f"cms_decays[{i}] = {d} must be in (0, 1)"

        # Normalize mutable lists to tuples for hashability and safety.
        self.cms_sizes = tuple(self.cms_sizes)
        self.cms_dims = tuple(self.cms_dims)
        self.cms_decays = tuple(self.cms_decays)

    def __hash__(self) -> int:
        """Enable use as a static argument in JAX transforms."""
        return hash(
            (
                self.d_s,
                self.d_w,
                self.d_p,
                self.d_e,
                self.d_k,
                self.d_c,
                self.num_levels,
                self.cms_sizes,
                self.cms_dims,
                self.cms_decays,
                self.learning_rate,
                self.gradient_clip,
                self.use_layer_norm,
                self.state_saturation_limit,
                self.obs_type,
                self.output_type,
                self.use_mamba_wave,
                self.mamba_state_dim,
                self.mamba_dt_min,
                self.mamba_dt_scale,
            )
        )
        
        # Validate timescale separation (faster levels decay more)
        for i in range(len(self.cms_decays) - 1):
            assert self.cms_decays[i] >= self.cms_decays[i+1], \
                f"Decay rates must be non-increasing: cms_decays[{i}] = {self.cms_decays[i]} < cms_decays[{i+1}] = {self.cms_decays[i+1]}"
    
    @classmethod
    def pi5_optimized(cls) -> "CoreModelConfig":
        """
        Create configuration optimized for Raspberry Pi 5.
        
        Targets:
        - Memory usage < 2GB
        - Inference speed > 10 steps/sec
        - Model size < 500MB
        """
        return cls(
            # Smaller dimensions for memory efficiency
            d_s=128,
            d_w=128,
            d_p=64,
            d_e=128,
            d_k=32,
            d_c=128,
            
            # Smaller CMS for memory efficiency
            num_levels=3,
            cms_sizes=[32, 64, 128],
            cms_dims=[64, 128, 256],
            cms_decays=[0.05, 0.03, 0.01],  # Slower decay for better retention
            
            learning_rate=1e-3,
            gradient_clip=5.0,
        )
    
    @classmethod
    def development(cls) -> "CoreModelConfig":
        """
        Create configuration for development/debugging.
        
        Smaller sizes for faster iteration.
        """
        return cls(
            d_s=64,
            d_w=64,
            d_p=32,
            d_e=64,
            d_k=16,
            d_c=64,
            
            num_levels=2,
            cms_sizes=[16, 32],
            cms_dims=[32, 64],
            cms_decays=[0.1, 0.05],
            
            learning_rate=1e-3,
            gradient_clip=5.0,
        )
    
    @classmethod
    def tpu_optimized(cls) -> "CoreModelConfig":
        """
        Create configuration optimized for Google Cloud TPU training.
        
        Larger dimensions for better model capacity.
        """
        return cls(
            d_s=512,
            d_w=512,
            d_p=256,
            d_e=512,
            d_k=128,
            d_c=512,
            
            num_levels=4,
            cms_sizes=[128, 256, 512, 1024],
            cms_dims=[256, 512, 1024, 2048],
            cms_decays=[0.1, 0.05, 0.02, 0.01],
            
            learning_rate=1e-4,
            gradient_clip=10.0,
        )

