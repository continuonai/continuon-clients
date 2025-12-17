"""
Checkpointing Utilities

Save and load model checkpoints using Orbax for GCS storage.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json

try:
    import orbax.checkpoint as ocp
    ORBAX_AVAILABLE = True
except ImportError:
    ORBAX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class CheckpointManager:
    """
    Manages model checkpoints using Orbax.
    
    Supports both local filesystem and GCS storage.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        use_gcs: bool = False,
        gcs_bucket: Optional[str] = None,
        max_to_keep: int = 5,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Local checkpoint directory or GCS path
            use_gcs: Whether to use GCS storage
            gcs_bucket: GCS bucket name (if use_gcs=True)
            max_to_keep: Maximum number of checkpoints to keep
        """
        if not ORBAX_AVAILABLE:
            raise ImportError("Orbax is required for checkpointing. Install with: pip install orbax-checkpoint")
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_gcs = use_gcs
        self.gcs_bucket = gcs_bucket
        self.max_to_keep = max_to_keep
        
        # Create checkpoint manager
        if use_gcs:
            if not gcs_bucket:
                raise ValueError("gcs_bucket must be provided when use_gcs=True")
            # GCS path format: gs://bucket/path
            ckpt_path = f"gs://{gcs_bucket}/{checkpoint_dir}"
        else:
            ckpt_path = str(self.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
        )
        checkpointer = ocp.PyTreeCheckpointer()
        self.manager = ocp.CheckpointManager(
            ckpt_path,
            checkpointer,
            options=options,
        )
    
    def save(
        self,
        step: int,
        params: Dict[str, Any],
        opt_state: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            step: Training step number
            params: Model parameters
            opt_state: Optimizer state
            metadata: Optional metadata dictionary
        
        Returns:
            Checkpoint path
        """
        checkpoint = {
            'params': params,
            'opt_state': opt_state,
            'metadata': metadata or {},
        }
        
        # Save checkpoint
        self.manager.save(step, args=checkpoint)
        
        # Also save metadata as JSON for easy inspection
        if metadata:
            metadata_path = self.checkpoint_dir / f"metadata_step_{step}.json"
            if not self.use_gcs:
                with metadata_path.open('w') as f:
                    json.dump(metadata, f, indent=2, default=str)
        
        return f"checkpoint_step_{step}"
    
    def load(
        self,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            step: Step number to load (loads latest if None)
        
        Returns:
            Dictionary with 'params', 'opt_state', and 'metadata'
        """
        if step is None:
            step = self.manager.latest_step()
            if step is None:
                raise ValueError("No checkpoints found")
        
        checkpoint = self.manager.restore(step)
        
        return {
            'params': checkpoint['params'],
            'opt_state': checkpoint['opt_state'],
            'metadata': checkpoint.get('metadata', {}),
            'step': step,
        }
    
    def list_checkpoints(self) -> list[int]:
        """
        List all available checkpoint steps.
        
        Returns:
            List of step numbers
        """
        return self.manager.all_steps()
    
    def get_latest_step(self) -> Optional[int]:
        """Get the latest checkpoint step number."""
        return self.manager.latest_step()


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    params: Dict[str, Any],
    opt_state: Any,
    metadata: Optional[Dict[str, Any]] = None,
    use_gcs: bool = False,
    gcs_bucket: Optional[str] = None,
) -> str:
    """
    Convenience function to save a checkpoint.
    
    Args:
        checkpoint_dir: Checkpoint directory
        step: Training step number
        params: Model parameters
        opt_state: Optimizer state
        metadata: Optional metadata
        use_gcs: Whether to use GCS
        gcs_bucket: GCS bucket name
    
    Returns:
        Checkpoint path
    """
    manager = CheckpointManager(checkpoint_dir, use_gcs=use_gcs, gcs_bucket=gcs_bucket)
    return manager.save(step, params, opt_state, metadata)


def load_checkpoint(
    checkpoint_dir: str,
    step: Optional[int] = None,
    use_gcs: bool = False,
    gcs_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to load a checkpoint.
    
    Args:
        checkpoint_dir: Checkpoint directory
        step: Step number (loads latest if None)
        use_gcs: Whether to use GCS
        gcs_bucket: GCS bucket name
    
    Returns:
        Dictionary with checkpoint data
    """
    manager = CheckpointManager(checkpoint_dir, use_gcs=use_gcs, gcs_bucket=gcs_bucket)
    return manager.load(step)

