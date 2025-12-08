"""
HOPE Configuration

Configuration management for HOPE architecture with sensible defaults
optimized for Raspberry Pi 5 deployment.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HOPEConfig:
    """
    Configuration for HOPE architecture.
    
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
    
    Optimization:
        use_quantization: Enable INT8/FP16 quantization
        quantization_dtype: "int8" or "fp16"
        device: Target device ("cpu", "cuda", etc.)
        dtype: Default tensor dtype
    
    Training:
        learning_rate: Base learning rate
        eta_init: Initial nested learning rate
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
    
    # Optimization
    quantization: str = "none"  # "none", "int8", "fp16"
    
    # Hybrid "Thousand Brains" Mode
    use_hybrid_mode: bool = False
    num_columns: int = 4
    device: str = "cpu"
    dtype: str = "float32"
    
    # Training
    learning_rate: float = 2e-3  # Increased for aggressive self-learning
    eta_init: float = 0.05       # Stronger initial plasticity
    gradient_clip: float = 10.0  # Allow larger updates for rapid adaptation
    
    # Stability
    use_layer_norm: bool = True
    lyapunov_weight: float = 1e-3
    lyapunov_threshold: float = 1e6  # Absolute energy threshold before flagging instability
    dissipation_floor: float = -20.0  # Minimum acceptable dissipation; negative implies energy increasing
    
    # Stability Constraints
    param_clamp_min: float = -0.5
    param_clamp_max: float = 0.5
    weight_decay: float = 0.99
    state_saturation_limit: float = 10.0
    
    # Compaction / Sleep Mode
    compaction_learning_rate: float = 0.1  # Aggressive learning during sleep
    compaction_decay: float = 0.5        # Aggressive forgetting during sleep
    
    # Autonomous Learning
    enable_autonomous_learning: bool = True
    learning_update_interval: int = 10  # Steps between parameter updates
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.num_levels == len(self.cms_sizes), \
            f"num_levels ({self.num_levels}) must match cms_sizes length ({len(self.cms_sizes)})"
        assert self.num_levels == len(self.cms_dims), \
            f"num_levels ({self.num_levels}) must match cms_dims length ({len(self.cms_dims)})"
        assert self.num_levels == len(self.cms_decays), \
            f"num_levels ({self.num_levels}) must match cms_decays length ({len(self.cms_decays)})"
        
        # Validate decay coefficients
        for i, d in enumerate(self.cms_decays):
            assert 0 < d < 1, f"cms_decays[{i}] = {d} must be in (0, 1)"
        
        # Validate timescale separation (faster levels decay more)
        for i in range(len(self.cms_decays) - 1):
            assert self.cms_decays[i] >= self.cms_decays[i+1], \
                f"Decay rates must be non-increasing: cms_decays[{i}] = {self.cms_decays[i]} < cms_decays[{i+1}] = {self.cms_decays[i+1]}"
        
        if self.quantization != "none":
            assert self.quantization in ["int8", "fp16"], \
                f"Quantization must be 'int8' or 'fp16', got {self.quantization}"
    
    @classmethod
    def pi5_optimized(cls) -> "HOPEConfig":
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
            
            # Enable quantization for deployment
            quantization="int8",
            device="cpu",
            dtype="float32",
            
            # Improved training parameters
            learning_rate=1e-3,
            gradient_clip=5.0,
            enable_autonomous_learning=True,
            learning_update_interval=10,
        )
    
    @classmethod
    def development(cls) -> "HOPEConfig":
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
            
            quantization="none",
            device="cpu",
        )
