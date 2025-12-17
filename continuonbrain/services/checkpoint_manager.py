"""
Checkpoint Manager for Autonomous Learning

Handles automatic checkpoint saving, loading, and cleanup.
"""

import torch
from pathlib import Path
from typing import Optional, List
import time
import json


class CheckpointManager:
    """Manages automatic checkpointing for continuous learning."""
    
    def __init__(
        self,
        checkpoint_dir: str = './checkpoints/autonomous',
        keep_last_n: int = 10,
        save_best: bool = True,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            keep_last_n: Number of recent checkpoints to keep
            save_best: Whether to save best checkpoint separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        
        self.best_metric = float('-inf')
        self.checkpoint_count = 0
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / 'metadata.json'
        self._load_metadata()
    
    def save_checkpoint(
        self,
        brain,
        step: int,
        metric: Optional[float] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            brain: HOPE brain to save
            step: Current training step
            metric: Performance metric (for best tracking)
            is_best: Force save as best
        
        Returns:
            Path to saved checkpoint
        """
        # Generate filename
        timestamp = int(time.time())
        filename = f'hope_auto_{step:08d}_{timestamp}.pt'
        path = self.checkpoint_dir / filename
        
        # Save checkpoint
        brain.save_checkpoint(str(path))
        
        self.checkpoint_count += 1
        
        # Update metadata
        self._update_metadata(str(path), step, metric)
        
        # Save as best if applicable
        if self.save_best and metric is not None:
            if is_best or metric > self.best_metric:
                self.best_metric = metric
                best_path = self.checkpoint_dir / 'hope_best.pt'
                brain.save_checkpoint(str(best_path))
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return path
    
    def load_latest(self) -> Optional[Path]:
        """
        Load most recent checkpoint.
        
        Returns:
            Path to loaded checkpoint, or None if no checkpoints exist
        """
        checkpoints = self._get_checkpoint_list()
        
        if not checkpoints:
            return None
        
        # Sort by modification time (most recent first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return checkpoints[0]
    
    def load_best(self) -> Optional[Path]:
        """
        Load best checkpoint.
        
        Returns:
            Path to best checkpoint, or None if doesn't exist
        """
        best_path = self.checkpoint_dir / 'hope_best.pt'
        
        if best_path.exists():
            return best_path
        
        return None
    
    def _get_checkpoint_list(self) -> List[Path]:
        """Get list of checkpoint files."""
        return list(self.checkpoint_dir.glob('hope_auto_*.pt'))
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        checkpoints = self._get_checkpoint_list()
        
        if len(checkpoints) <= self.keep_last_n:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Remove old ones
        for old_checkpoint in checkpoints[self.keep_last_n:]:
            try:
                old_checkpoint.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete {old_checkpoint}: {e}")
    
    def _load_metadata(self):
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.checkpoint_count = metadata.get('checkpoint_count', 0)
                    self.best_metric = metadata.get('best_metric', float('-inf'))
            except Exception as e:
                print(f"Warning: Failed to load metadata: {e}")
    
    def _update_metadata(self, path: str, step: int, metric: Optional[float]):
        """Update metadata file."""
        metadata = {
            'checkpoint_count': self.checkpoint_count,
            'best_metric': self.best_metric,
            'last_checkpoint': path,
            'last_step': step,
            'last_metric': metric,
            'timestamp': time.time(),
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")
    
    def get_statistics(self) -> dict:
        """Get checkpoint statistics."""
        checkpoints = self._get_checkpoint_list()
        
        total_size = sum(p.stat().st_size for p in checkpoints)
        
        return {
            'total_checkpoints': len(checkpoints),
            'checkpoint_count': self.checkpoint_count,
            'best_metric': self.best_metric if self.best_metric != float('-inf') else None,
            'total_size_mb': total_size / (1024 * 1024),
            'checkpoint_dir': str(self.checkpoint_dir),
        }
