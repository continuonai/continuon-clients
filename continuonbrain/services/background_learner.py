"""
Background Learner Service

Runs continuous autonomous learning in a separate thread.
"""

import threading
import time
import torch
import numpy as np
from typing import Optional
import logging

from continuonbrain.hope_impl.curiosity_env import CuriosityEnvironment
from continuonbrain.services.checkpoint_manager import CheckpointManager


logger = logging.getLogger(__name__)


class BackgroundLearner:
    """
    Manages continuous autonomous learning in background.
    
    Runs in separate thread, allowing brain to learn while
    server handles user requests.
    """
    
    def __init__(
        self,
        brain,
        config: Optional[dict] = None,
    ):
        """
        Initialize background learner.
        
        Args:
            brain: HOPE brain instance
            config: Configuration dict
        """
        self.brain = brain
        self.config = config or self._default_config()
        
        # Create curiosity environment
        self.env = CuriosityEnvironment(
            obs_dim=brain.obs_dim,
            action_dim=brain.action_dim,
            exploration_bonus=self.config.get('exploration_bonus', 0.1),
            novelty_threshold=self.config.get('novelty_threshold', 0.5),
        )
        
        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.get('checkpoint_dir', './checkpoints/autonomous'),
            keep_last_n=self.config.get('keep_checkpoints', 10),
        )
        
        # Thread management
        self.running = False
        self.paused = False
        self.thread = None
        self.lock = threading.Lock()  # For thread-safe brain access
        
        # Statistics
        self.total_steps = 0
        self.start_time = None
        self.last_checkpoint_step = 0
        
        # Configuration
        self.steps_per_cycle = self.config.get('steps_per_cycle', 100)
        self.cycle_interval_sec = self.config.get('cycle_interval_sec', 1.0)
        self.checkpoint_interval = self.config.get('checkpoint_interval', 1000)
        
        logger.info("BackgroundLearner initialized")
    
    def start(self):
        """Start background learning thread."""
        if self.running:
            logger.warning("Background learner already running")
            return
        
        # Try to load latest checkpoint
        latest_checkpoint = self.checkpoint_manager.load_latest()
        if latest_checkpoint:
            logger.info(f"Loading checkpoint: {latest_checkpoint}")
            try:
                self.brain.load_checkpoint(str(latest_checkpoint))
                logger.info("Checkpoint loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        
        self.running = True
        self.start_time = time.time()
        
        # Start thread
        self.thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.thread.start()
        
        logger.info("Background learning started")
    
    def stop(self):
        """Stop background learning gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping background learning...")
        self.running = False
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=5.0)
        
        # Save final checkpoint
        logger.info("Saving final checkpoint...")
        self._save_checkpoint(final=True)
        
        logger.info("Background learning stopped")
    
    def pause(self):
        """Pause learning (keeps thread alive)."""
        self.paused = True
        logger.info("Background learning paused")
    
    def resume(self):
        """Resume learning."""
        self.paused = False
        logger.info("Background learning resumed")
    
    def _learning_loop(self):
        """Main learning loop (runs in separate thread)."""
        logger.info("Learning loop started")
        
        # Reset environment
        obs = self.env.reset()
        action = torch.zeros(self.brain.action_dim)
        reward = 0.0
        
        while self.running:
            # Check if paused
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # Learning cycle
                for _ in range(self.steps_per_cycle):
                    if not self.running or self.paused:
                        break
                    
                    # Convert observation to tensor
                    x_obs = torch.from_numpy(obs).float()
                    
                    # Brain step (thread-safe)
                    with self.lock:
                        state_next, y_t, info = self.brain.step(x_obs, action, reward)
                    
                    # Use output as next action
                    action = y_t
                    
                    # Get prediction for next state (for surprise calculation)
                    prediction = y_t.detach().numpy()[:self.env.obs_dim]
                    
                    # Environment step
                    obs, reward, _ = self.env.step(action.detach().numpy(), prediction)
                    
                    self.total_steps += 1
                    
                    # Check for instability
                    if not self.brain.stability_monitor.is_stable():
                        logger.warning(f"Instability detected at step {self.total_steps}")
                        # Save checkpoint and continue (don't stop)
                        self._save_checkpoint()
                    
                    # Periodic checkpointing
                    if self.total_steps - self.last_checkpoint_step >= self.checkpoint_interval:
                        self._save_checkpoint()
                
                # Log progress
                if self.total_steps % (self.steps_per_cycle * 10) == 0:
                    self._log_progress()
                
                # Sleep between cycles
                time.sleep(self.cycle_interval_sec)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}", exc_info=True)
                # Save checkpoint on error
                self._save_checkpoint()
                # Sleep and continue
                time.sleep(1.0)
        
        logger.info("Learning loop ended")
    
    def _save_checkpoint(self, final: bool = False):
        """Save checkpoint (thread-safe)."""
        try:
            with self.lock:
                # Get current performance metric
                metrics = self.brain.stability_monitor.get_metrics()
                metric = -metrics.get('lyapunov_current', 0.0)  # Negative because lower is better
                
                path = self.checkpoint_manager.save_checkpoint(
                    self.brain,
                    step=self.total_steps,
                    metric=metric,
                )
                
                self.last_checkpoint_step = self.total_steps
                
                if final:
                    logger.info(f"Final checkpoint saved: {path}")
                else:
                    logger.debug(f"Checkpoint saved: {path}")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _log_progress(self):
        """Log learning progress."""
        elapsed = time.time() - self.start_time
        steps_per_sec = self.total_steps / elapsed if elapsed > 0 else 0
        
        metrics = self.brain.stability_monitor.get_metrics()
        curiosity_stats = self.env.get_statistics()
        
        logger.info(
            f"Autonomous Learning | "
            f"Steps: {self.total_steps} | "
            f"Speed: {steps_per_sec:.1f} steps/s | "
            f"Novelty: {curiosity_stats['avg_novelty']:.3f} | "
            f"Lyapunov: {metrics.get('lyapunov_current', 0):.2f}"
        )
    
    def get_status(self) -> dict:
        """Get current learning status."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        curiosity_stats = self.env.get_statistics()
        checkpoint_stats = self.checkpoint_manager.get_statistics()
        
        return {
            'enabled': True,
            'running': self.running,
            'paused': self.paused,
            'total_steps': self.total_steps,
            'uptime_hours': elapsed / 3600,
            'steps_per_sec': self.total_steps / elapsed if elapsed > 0 else 0,
            'current_novelty': self.env.current_novelty,
            'avg_novelty': curiosity_stats['avg_novelty'],
            'novel_state_rate': curiosity_stats['novelty_rate'],
            'checkpoint_count': checkpoint_stats['checkpoint_count'],
            'last_checkpoint_step': self.last_checkpoint_step,
        }
    
    @staticmethod
    def _default_config() -> dict:
        """Default configuration."""
        return {
            'steps_per_cycle': 100,
            'cycle_interval_sec': 1.0,
            'checkpoint_interval': 1000,
            'exploration_bonus': 0.1,
            'novelty_threshold': 0.5,
            'checkpoint_dir': './checkpoints/autonomous',
            'keep_checkpoints': 10,
        }
