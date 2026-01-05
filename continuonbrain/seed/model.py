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
        
        # Update state with new CMS memories from write operation
        new_state = {
            's': info['fast_state'],
            'w': info['wave_state'],
            'p': info['particle_state'],
            'cms_memories': info.get('cms_memories', state['cms_memories']),
            'cms_keys': info.get('cms_keys', state['cms_keys']),
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
    
    def _path_to_key(self, key_path) -> str:
        """
        Convert JAX tree path to a string key for serialization.

        Args:
            key_path: JAX tree path tuple (e.g., (DictKey('params'), DictKey('Dense_0'), ...))

        Returns:
            String key like "params/Dense_0/kernel"
        """
        parts = []
        for p in key_path:
            # Handle different path element types
            if hasattr(p, 'key'):
                parts.append(str(p.key))
            elif hasattr(p, 'idx'):
                parts.append(f"[{p.idx}]")
            else:
                parts.append(str(p))
        return "/".join(parts)

    def _key_to_path(self, key: str) -> Tuple:
        """
        Convert string key back to path components.

        Args:
            key: String key like "params/Dense_0/kernel" or "params/layers/[0]/weight"

        Returns:
            Tuple of path components
        """
        import re
        parts = key.split("/")
        result = []
        for part in parts:
            # Check if it's an index like "[0]"
            match = re.match(r'\[(\d+)\]', part)
            if match:
                result.append(('idx', int(match.group(1))))
            else:
                result.append(('key', part))
        return tuple(result)

    def _rebuild_params_tree(self, loaded: Dict[str, Any], template: Any) -> Any:
        """
        Rebuild JAX params tree from flat loaded weights using template structure.

        Args:
            loaded: Dict of loaded numpy arrays keyed by path strings
            template: Template params tree with correct structure

        Returns:
            Rebuilt params tree with loaded values
        """
        import jax
        import jax.numpy as jnp

        # Build a mapping from path keys to loaded arrays
        loaded_dict = {k: v for k, v in loaded.items()}

        # Track which keys were used for verification
        used_keys = set()
        missing_keys = []
        shape_mismatches = []

        def map_leaf(path, template_value):
            """Map a single leaf from loaded weights."""
            key = self._path_to_key(path)

            if key in loaded_dict:
                loaded_value = loaded_dict[key]
                used_keys.add(key)

                # Verify shape matches
                if loaded_value.shape != template_value.shape:
                    shape_mismatches.append({
                        'key': key,
                        'expected': template_value.shape,
                        'got': loaded_value.shape
                    })
                    logger.warning(
                        f"Shape mismatch for {key}: expected {template_value.shape}, "
                        f"got {loaded_value.shape}. Using template value."
                    )
                    return template_value

                # Convert to JAX array with same dtype
                return jnp.array(loaded_value, dtype=template_value.dtype)
            else:
                missing_keys.append(key)
                logger.warning(f"Missing key in checkpoint: {key}. Using initialized value.")
                return template_value

        # Use tree_map_with_path to rebuild the tree
        rebuilt_params = jax.tree_util.tree_map_with_path(map_leaf, template)

        # Check for unused keys in loaded checkpoint
        unused_keys = set(loaded_dict.keys()) - used_keys
        if unused_keys:
            logger.warning(f"Unused keys in checkpoint ({len(unused_keys)}): {list(unused_keys)[:5]}...")

        # Log summary
        total_keys = len(loaded_dict)
        matched_keys = len(used_keys)
        logger.info(
            f"Checkpoint loading: {matched_keys}/{total_keys} weights matched, "
            f"{len(missing_keys)} missing, {len(shape_mismatches)} shape mismatches"
        )

        return rebuilt_params

    def _verify_params_structure(self, loaded_params: Any, expected_params: Any) -> bool:
        """
        Verify that loaded params match expected structure.

        Args:
            loaded_params: The loaded parameters
            expected_params: Expected parameter structure (template)

        Returns:
            True if structures match, False otherwise
        """
        import jax

        # Get tree structures
        loaded_structure = jax.tree_util.tree_structure(loaded_params)
        expected_structure = jax.tree_util.tree_structure(expected_params)

        if loaded_structure != expected_structure:
            logger.error(
                f"Parameter structure mismatch! "
                f"Expected {expected_structure}, got {loaded_structure}"
            )
            return False

        # Verify leaf counts
        loaded_leaves = jax.tree_util.tree_leaves(loaded_params)
        expected_leaves = jax.tree_util.tree_leaves(expected_params)

        if len(loaded_leaves) != len(expected_leaves):
            logger.error(
                f"Parameter count mismatch! "
                f"Expected {len(expected_leaves)} leaves, got {len(loaded_leaves)}"
            )
            return False

        # Verify shapes
        for i, (loaded_leaf, expected_leaf) in enumerate(zip(loaded_leaves, expected_leaves)):
            if loaded_leaf.shape != expected_leaf.shape:
                logger.error(
                    f"Shape mismatch at leaf {i}: "
                    f"expected {expected_leaf.shape}, got {loaded_leaf.shape}"
                )
                return False

        # Verify total param count
        loaded_count = sum(x.size for x in loaded_leaves)
        expected_count = sum(x.size for x in expected_leaves)

        if loaded_count != expected_count:
            logger.error(
                f"Total param count mismatch! "
                f"Expected {expected_count:,}, got {loaded_count:,}"
            )
            return False

        logger.info(f"Parameter verification passed: {loaded_count:,} params, structure matches")
        return True

    def save(self, path: Path) -> None:
        """
        Save seed model checkpoint with preserved JAX tree structure.

        Args:
            path: Directory to save checkpoint
        """
        import numpy as np
        import jax

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save weights with path keys preserving tree structure
        weights_dict = {}
        param_shapes = {}

        for key_path, value in jax.tree_util.tree_leaves_with_path(self._params):
            key = self._path_to_key(key_path)
            weights_dict[key] = np.array(value)
            param_shapes[key] = list(value.shape)

        np.savez(path / "seed_model.npz", **weights_dict)

        # Get actual param count
        actual_param_count = sum(x.size for x in jax.tree_util.tree_leaves(self._params))

        # Save manifest with structure metadata
        manifest = {
            "version": self.VERSION,
            "timestamp": datetime.now().isoformat(),
            "model_type": "seed_model",
            "hardware": self.hardware.to_dict(),
            "config": {
                "d_s": self.config.d_s,
                "d_w": self.config.d_w,
                "d_p": self.config.d_p,
                "d_e": self.config.d_e,
                "d_k": self.config.d_k,
                "d_c": self.config.d_c,
                "num_levels": self.config.num_levels,
                "cms_sizes": self.config.cms_sizes,
                "cms_dims": self.config.cms_dims,
                "cms_decays": self.config.cms_decays,
                "use_mamba_wave": self.config.use_mamba_wave,
                "mamba_state_dim": self.config.mamba_state_dim,
                "learning_rate": self.config.learning_rate,
                "gradient_clip": self.config.gradient_clip,
                "target_device": self.config.target_device,
            },
            "param_count": actual_param_count,
            "param_shapes": param_shapes,
            "num_leaves": len(weights_dict),
            "portability": [
                "arm64", "x86_64", "riscv64", "apple_silicon"
            ],
        }

        with open(path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved seed model to {path} ({actual_param_count:,} params, {len(weights_dict)} leaves)")
    
    def load(self, path: Path) -> None:
        """
        Load seed model from checkpoint.

        Args:
            path: Directory containing checkpoint

        Raises:
            FileNotFoundError: If checkpoint files don't exist
            ValueError: If checkpoint is incompatible or corrupted
        """
        import numpy as np
        import jax

        path = Path(path)

        # Validate checkpoint exists
        manifest_path = path / "manifest.json"
        weights_path = path / "seed_model.npz"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        # Load manifest for config and metadata
        with open(manifest_path) as f:
            manifest = json.load(f)

        logger.info(f"Loading seed model v{manifest['version']} from {path}")

        # Check version compatibility
        saved_version = manifest.get('version', '0.0.0')
        if saved_version.split('.')[0] != self.VERSION.split('.')[0]:
            logger.warning(
                f"Major version mismatch: checkpoint v{saved_version}, "
                f"current v{self.VERSION}. Loading may fail."
            )

        # Update config from manifest if available
        saved_config = manifest.get('config', {})
        if saved_config:
            logger.debug(f"Checkpoint config: {saved_config}")

        # Initialize model with fresh params (creates template structure)
        self._initialize()

        # Store template for structure verification
        template_params = self._params

        # Load weights from NPZ
        try:
            loaded = np.load(weights_path, allow_pickle=False)
        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {e}")

        # Log checkpoint info
        loaded_keys = list(loaded.keys())
        logger.info(f"Checkpoint contains {len(loaded_keys)} weight arrays")

        # Verify expected structure from manifest
        expected_leaves = manifest.get('num_leaves', len(loaded_keys))
        if len(loaded_keys) != expected_leaves:
            logger.warning(
                f"Weight count mismatch: manifest says {expected_leaves}, "
                f"file contains {len(loaded_keys)}"
            )

        # Rebuild params tree from loaded weights using template
        self._params = self._rebuild_params_tree(loaded, template_params)

        # Verify the rebuilt structure matches expected
        if not self._verify_params_structure(self._params, template_params):
            raise ValueError(
                "Failed to verify loaded params structure. "
                "Checkpoint may be corrupted or incompatible."
            )

        # Final verification: compare param counts
        loaded_param_count = sum(x.size for x in jax.tree_util.tree_leaves(self._params))
        expected_param_count = manifest.get('param_count', loaded_param_count)

        if loaded_param_count != expected_param_count:
            logger.warning(
                f"Param count differs from manifest: expected {expected_param_count:,}, "
                f"got {loaded_param_count:,}"
            )

        logger.info(f"Loaded seed model from {path} ({loaded_param_count:,} params)")
    
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

