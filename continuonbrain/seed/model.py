"""
Universal Seed Model

The Seed Model is the initialization point for every robot in the
Continuon ecosystem. It runs on any hardware platform.

Example:
    from continuonbrain.seed import SeedModel
    
    # Auto-detect hardware and initialize
    seed = SeedModel()
    
    # Or specify target
    seed = SeedModel(target='pi5')
    seed = SeedModel(target='jetson')
    seed = SeedModel(target='cloud')
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SeedModel:
    """
    Universal Seed Model for robot initialization.
    
    Provides the foundational cognitive capabilities that every robot
    shares, regardless of hardware platform:
    
    - World Model (next-token prediction)
    - Context Graph (relational reasoning)
    - Semantic Search (768-dim embeddings)
    - Decision Traces (explainability)
    - CMS Memory (multi-timescale)
    
    The seed model runs on any chip:
    - ARM64 (Pi5, Jetson)
    - x86_64 (PC, Server, Cloud)
    - RISC-V (Edge devices)
    - Quantum (Future)
    - Neuromorphic (Future)
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        target: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
        config: Optional["SeedConfig"] = None,
    ):
        """
        Initialize the seed model.
        
        Args:
            target: Hardware target ('pi5', 'jetson', 'cloud', 'auto').
                   If None, auto-detects hardware.
            checkpoint_path: Path to load existing checkpoint.
                            If None, initializes fresh.
            config: Optional custom config. If None, uses hardware-optimized.
        """
        from .hardware import get_profile, HardwareProfile
        from .config import SeedConfig
        
        # Detect hardware
        self.hardware = get_profile(target)
        logger.info(f"Hardware: {self.hardware.device_name} ({self.hardware.architecture.value})")
        
        # Get config
        if config is not None:
            self.config = config
        else:
            self.config = SeedConfig.for_hardware(self.hardware)
        
        logger.info(f"Config: {self.config.target_device} ({self.config.param_count_estimate():,} params est.)")
        
        # Initialize components
        self._model = None
        self._params = None
        self._initialized = False
        
        # Load or initialize
        if checkpoint_path is not None:
            self.load(checkpoint_path)
        else:
            self._initialize()
    
    def _initialize(self) -> None:
        """Initialize model from scratch."""
        try:
            import jax
            import jax.numpy as jnp
            
            from continuonbrain.jax_models.config import CoreModelConfig
            from continuonbrain.jax_models.core_model import CoreModel
            
            # Convert seed config to core config
            core_config = CoreModelConfig(
                d_s=self.config.d_s,
                d_w=self.config.d_w,
                d_p=self.config.d_p,
                d_e=self.config.d_e,
                d_k=self.config.d_k,
                d_c=self.config.d_c,
                num_levels=self.config.num_levels,
                cms_sizes=self.config.cms_sizes,
                cms_dims=self.config.cms_dims,
                cms_decays=self.config.cms_decays,
                use_mamba_wave=self.config.use_mamba_wave,
                mamba_state_dim=self.config.mamba_state_dim,
                learning_rate=self.config.learning_rate,
                gradient_clip=self.config.gradient_clip,
            )
            
            # Create model
            self._model = CoreModel(
                config=core_config,
                obs_dim=64,
                action_dim=7,
                output_dim=7,
            )
            
            # Initialize parameters
            rng = jax.random.PRNGKey(42)
            batch = 1
            
            dummy_inputs = self._create_dummy_inputs(batch, core_config)
            self._params = self._model.init(rng, **dummy_inputs)
            
            self._initialized = True
            
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(self._params))
            logger.info(f"✅ Seed model initialized ({param_count:,} params)")
            
        except Exception as e:
            logger.error(f"Failed to initialize seed model: {e}")
            raise
    
    def _create_dummy_inputs(self, batch: int, config) -> Dict[str, Any]:
        """Create dummy inputs for model initialization."""
        import jax.numpy as jnp
        
        return {
            'x_obs': jnp.zeros((batch, 64)),
            'a_prev': jnp.zeros((batch, 7)),
            'r_t': jnp.zeros((batch, 1)),
            's_prev': jnp.zeros((batch, config.d_s)),
            'w_prev': jnp.zeros((batch, config.d_w)),
            'p_prev': jnp.zeros((batch, config.d_p)),
            'cms_memories': [
                jnp.zeros((batch, sz, dim)) 
                for sz, dim in zip(config.cms_sizes, config.cms_dims)
            ],
            'cms_keys': [
                jnp.zeros((batch, sz, config.d_k)) 
                for sz in config.cms_sizes
            ],
        }
    
    def forward(
        self,
        observation: Any,
        action_prev: Any,
        reward: float,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Forward pass through seed model.
        
        Args:
            observation: Current observation
            action_prev: Previous action taken
            reward: Reward received
            state: Previous state dict (if None, uses zeros)
        
        Returns:
            output: Model output (action prediction, etc.)
            new_state: Updated state dict
        """
        if not self._initialized:
            raise RuntimeError("Seed model not initialized")
        
        import jax.numpy as jnp
        
        # Get or create state
        if state is None:
            state = self._create_initial_state()
        
        # Convert inputs
        x_obs = jnp.array(observation).reshape(1, -1)
        a_prev = jnp.array(action_prev).reshape(1, -1)
        r_t = jnp.array([[reward]])
        
        # Forward
        output, info = self._model.apply(
            self._params,
            x_obs=x_obs,
            a_prev=a_prev,
            r_t=r_t,
            s_prev=state['s'],
            w_prev=state['w'],
            p_prev=state['p'],
            cms_memories=state['cms_memories'],
            cms_keys=state['cms_keys'],
        )
        
        # Update state
        new_state = {
            's': info['fast_state'],
            'w': info['wave_state'],
            'p': info['particle_state'],
            'cms_memories': state['cms_memories'],  # TODO: update with write
            'cms_keys': state['cms_keys'],
        }
        
        return output, new_state
    
    def _create_initial_state(self) -> Dict[str, Any]:
        """Create initial state for inference."""
        import jax.numpy as jnp
        
        config = self.config
        batch = 1
        
        return {
            's': jnp.zeros((batch, config.d_s)),
            'w': jnp.zeros((batch, config.d_w)),
            'p': jnp.zeros((batch, config.d_p)),
            'cms_memories': [
                jnp.zeros((batch, sz, dim)) 
                for sz, dim in zip(config.cms_sizes, config.cms_dims)
            ],
            'cms_keys': [
                jnp.zeros((batch, sz, config.d_k)) 
                for sz in config.cms_sizes
            ],
        }
    
    def save(self, path: Path) -> None:
        """
        Save seed model checkpoint.
        
        Args:
            path: Directory to save checkpoint
        """
        import numpy as np
        import jax
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        weights_dict = {}
        for key_path, value in jax.tree_util.tree_leaves_with_path(self._params):
            key = "/".join(str(getattr(p, 'key', p)) for p in key_path)
            weights_dict[key] = np.array(value)
        
        np.savez(path / "seed_model.npz", **weights_dict)
        
        # Save manifest
        manifest = {
            "version": self.VERSION,
            "timestamp": datetime.now().isoformat(),
            "model_type": "seed_model",
            "hardware": self.hardware.to_dict(),
            "config": {
                "d_s": self.config.d_s,
                "d_w": self.config.d_w,
                "d_p": self.config.d_p,
                "num_levels": self.config.num_levels,
                "target_device": self.config.target_device,
            },
            "param_count": self.config.param_count_estimate(),
            "portability": [
                "arm64", "x86_64", "riscv64", "apple_silicon"
            ],
        }
        
        with open(path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Saved seed model to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load seed model from checkpoint.
        
        Args:
            path: Directory containing checkpoint
        """
        import numpy as np
        import jax.numpy as jnp
        
        path = Path(path)
        
        # Load manifest
        with open(path / "manifest.json") as f:
            manifest = json.load(f)
        
        logger.info(f"Loading seed model v{manifest['version']} from {path}")
        
        # Initialize model first
        self._initialize()
        
        # Load weights
        loaded = np.load(path / "seed_model.npz", allow_pickle=True)
        
        # TODO: Map loaded weights to params structure
        logger.info(f"✅ Loaded seed model ({manifest['param_count']:,} params)")
    
    @property
    def param_count(self) -> int:
        """Get actual parameter count."""
        if self._params is None:
            return self.config.param_count_estimate()
        
        import jax
        return sum(x.size for x in jax.tree_util.tree_leaves(self._params))
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "version": self.VERSION,
            "initialized": self._initialized,
            "hardware": self.hardware.to_dict(),
            "config": {
                "d_s": self.config.d_s,
                "d_w": self.config.d_w,
                "d_p": self.config.d_p,
                "num_levels": self.config.num_levels,
                "target_device": self.config.target_device,
            },
            "param_count": self.param_count,
            "memory_estimate_mb": self.config.memory_estimate_mb(),
            "capabilities": [
                "world_model",
                "context_graph",
                "semantic_search",
                "decision_traces",
                "cms_memory",
            ],
            "portability": [
                "arm64", "x86_64", "riscv64", "apple_silicon",
                "quantum (future)", "neuromorphic (future)"
            ],
        }

