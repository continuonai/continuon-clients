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
import json
import uuid
from pathlib import Path

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
        resource_monitor: Optional['ResourceMonitor'] = None,
    ):
        """
        Initialize background learner.
        
        Args:
            brain: HOPE brain instance
            config: Configuration dict
            resource_monitor: Monitor for system resources
        """
        self.brain = brain
        self.config = config or self._default_config()
        self.resource_monitor = resource_monitor
        
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

        # Optional RLDS-ish logging of curiosity/stability metrics for replay/debug.
        # This intentionally uses the existing "step_metadata" string map pattern so we don't need
        # schema changes for new signals (novelty/surprise/etc).
        self.rlds_log_dir: Optional[Path] = None
        rlds_dir = self.config.get("rlds_log_dir")
        if rlds_dir:
            self.rlds_log_dir = Path(rlds_dir)
        self._rlds_episode_dir: Optional[Path] = None
        self._rlds_steps_path: Optional[Path] = None
        
        # Thread management
        self.running = False
        self.paused = False
        self.thread = None
        self.lock = threading.Lock()  # For thread-safe brain access
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.start_time = None
        self.last_checkpoint_step = 0
        
        # Learning progress tracking
        self.parameter_changes = []  # Track magnitude of parameter changes
        self.last_params = None  # For computing parameter deltas
        self.learning_updates = 0  # Count of actual parameter updates
        self.update_interval = self.config.get('learning_update_interval', 10)
        
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

        # Initialize RLDS-ish episode directory if enabled.
        if self.rlds_log_dir:
            try:
                episode_id = f"autonomous_learning_{int(self.start_time)}_{uuid.uuid4().hex[:8]}"
                episode_dir = self.rlds_log_dir / episode_id
                steps_dir = episode_dir / "steps"
                steps_dir.mkdir(parents=True, exist_ok=True)
                self._rlds_episode_dir = episode_dir
                self._rlds_steps_path = steps_dir / "000000.jsonl"

                # Minimal metadata for offline replay/debug (kept permissive).
                metadata = {
                    "xr_mode": "autonomous",
                    "control_role": "human_supervisor",
                    "environment_id": self.config.get("environment_id", "pi5-dev"),
                    "tags": [
                        "autonomous_learning",
                        "curiosity",
                        "control_role:autonomous_agent",
                    ],
                    "software": {
                        "xr_app": "n/a",
                        "continuonbrain_os": self.config.get("continuonbrain_version", "dev"),
                        "glove_firmware": "n/a",
                    },
                    "start_time_unix_ms": int(self.start_time * 1000),
                }
                (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                logger.info(f"Curiosity metrics logging enabled: {episode_dir}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to initialize curiosity RLDS logging: {exc}")
                self._rlds_episode_dir = None
                self._rlds_steps_path = None
        
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
            
            # Check Resource Constraints
            if self.resource_monitor:
                res = self.resource_monitor.check_resources()
                # Fix: Check .value for string comparison, or import ResourceLevel
                if res.level.value in ["critical", "emergency"]:
                    if self.total_steps % 100 == 0:  # Don't spam logs
                        logger.warning(f"Resource constraint ({res.level.value}): Pausing autonomous learning.")
                    time.sleep(5.0)  # Wait for resources to free up
                    continue

            
            try:
                # Learning cycle
                for step_in_cycle in range(self.steps_per_cycle):
                    if not self.running or self.paused:
                        break
                    
                    # Convert observation to tensor
                    x_obs = torch.from_numpy(obs).float()
                    
                    # Determine if we should update parameters this step
                    should_update = (self.total_steps % self.update_interval == 0)
                    
                    # Brain step (thread-safe) with parameter updates enabled
                    with self.lock:
                        state_next, y_t, info = self.brain.step(
                            x_obs, action, reward,
                            perform_param_update=should_update,
                            perform_cms_write=True,
                            log_stability=True
                        )
                        
                        # Track parameter changes if update was performed
                        if should_update:
                            param_change = self._measure_parameter_change()
                            if param_change is not None:
                                self.parameter_changes.append(param_change)
                                # Keep only recent history
                                if len(self.parameter_changes) > 1000:
                                    self.parameter_changes = self.parameter_changes[-1000:]
                            self.learning_updates += 1
                    
                    # Use output as next action
                    action = y_t
                    
                    # Get prediction for next state (for surprise calculation)
                    prediction_np = y_t.detach().cpu().numpy()
                    if prediction_np.ndim > 1:
                        prediction_np = prediction_np.squeeze()
                    prediction = prediction_np[:self.env.obs_dim]
                    
                    # Squeeze action if batched
                    action_np = action.detach().cpu().numpy()
                    if action_np.ndim > 1:
                        action_np = action_np.squeeze()
                        
                    obs, reward, done = self.env.step(action_np, prediction)
                    
                    # Track episode rewards
                    self.current_episode_reward += reward
                    
                    # Handle episode end
                    if done:
                        self.episode_rewards.append(self.current_episode_reward)
                        # Keep only recent episodes
                        if len(self.episode_rewards) > 100:
                            self.episode_rewards = self.episode_rewards[-100:]
                        self.current_episode_reward = 0.0
                        self.total_episodes += 1
                        obs = self.env.reset()
                    
                    self.total_steps += 1
                    
                    # Check for instability
                    if not self.brain.stability_monitor.is_stable():
                        if self.total_steps % 100 == 0: # Log and save less frequently
                            logger.warning(f"Instability detected at step {self.total_steps}")
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
    
    def _measure_parameter_change(self) -> Optional[float]:
        """Measure how much parameters changed since last measurement."""
        try:
            current_params = {
                name: param.clone().detach()
                for name, param in self.brain.named_parameters()
            }
            
            if self.last_params is None:
                self.last_params = current_params
                return None
            
            # Compute total change
            total_change = 0.0
            for name in current_params:
                if name in self.last_params:
                    diff = torch.norm(current_params[name] - self.last_params[name]).item()
                    total_change += diff
            
            self.last_params = current_params
            return total_change
            
        except Exception as e:
            logger.error(f"Error measuring parameter change: {e}")
            return None
    
    def _log_progress(self):
        """Log learning progress."""
        elapsed = time.time() - self.start_time
        steps_per_sec = self.total_steps / elapsed if elapsed > 0 else 0
        
        metrics = self.brain.stability_monitor.get_metrics()
        curiosity_stats = self.env.get_statistics()
        
        # Compute average parameter change
        avg_param_change = (
            sum(self.parameter_changes[-10:]) / len(self.parameter_changes[-10:])
            if self.parameter_changes else 0.0
        )
        
        # Compute average episode reward
        avg_reward = (
            sum(self.episode_rewards[-10:]) / len(self.episode_rewards[-10:])
            if self.episode_rewards else 0.0
        )
        
        logger.info(
            f"Autonomous Learning | "
            f"Steps: {self.total_steps} | "
            f"Episodes: {self.total_episodes} | "
            f"Speed: {steps_per_sec:.1f} steps/s | "
            f"Param Δ: {avg_param_change:.6f} | "
            f"Avg Reward: {avg_reward:.3f} | "
            f"Novelty: {curiosity_stats['avg_novelty']:.3f} | "
            f"Lyapunov: {metrics.get('lyapunov_current', 0):.2f}"
        )

        # Persist a lightweight RLDS-ish metrics step (one per progress log).
        self._append_rlds_metrics_step(metrics, curiosity_stats, avg_param_change, avg_reward)

    def _append_rlds_metrics_step(
        self,
        stability_metrics: dict,
        curiosity_stats: dict,
        avg_param_change: float,
        avg_reward: float,
    ) -> None:
        """Append a single JSONL step with curiosity/stability metrics if logging is enabled."""
        if not self._rlds_steps_path:
            return
        try:
            step = {
                "observation": {
                    "learning": {
                        "total_steps": self.total_steps,
                        "total_episodes": self.total_episodes,
                        "learning_updates": self.learning_updates,
                    }
                },
                "action": {
                    "kind": "autonomous_learning_tick",
                },
                "reward": float(avg_reward),
                "is_terminal": False,
                "step_metadata": {
                    "timestamp": str(time.time()),
                    "avg_parameter_change": str(avg_param_change),
                    "avg_episode_reward": str(avg_reward),
                    "avg_novelty": str(curiosity_stats.get("avg_novelty", 0.0)),
                    "novelty_rate": str(curiosity_stats.get("novelty_rate", 0.0)),
                    "lyapunov_current": str(stability_metrics.get("lyapunov_current", 0.0)),
                    "dissipation_rate": str(stability_metrics.get("dissipation_rate", 0.0)),
                    "is_stable": str(bool(self.brain.stability_monitor.is_stable())),
                },
            }
            with self._rlds_steps_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(step) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Curiosity RLDS step append failed: {exc}")
    
    def get_status(self) -> dict:
        """Get current learning status with comprehensive progress metrics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        curiosity_stats = self.env.get_statistics()
        checkpoint_stats = self.checkpoint_manager.get_statistics()
        
        # Compute learning progress metrics
        avg_param_change = (
            sum(self.parameter_changes[-100:]) / len(self.parameter_changes[-100:])
            if self.parameter_changes else 0.0
        )
        
        recent_param_change = self.parameter_changes[-1] if self.parameter_changes else 0.0
        
        avg_episode_reward = (
            sum(self.episode_rewards) / len(self.episode_rewards)
            if self.episode_rewards else 0.0
        )
        
        recent_episode_reward = (
            sum(self.episode_rewards[-10:]) / len(self.episode_rewards[-10:])
            if len(self.episode_rewards) >= 10 else avg_episode_reward
        )
        
        # Get brain metrics
        brain_metrics = self.brain.stability_monitor.get_metrics()
        brain_state = self.brain.get_state()
        
        return {
            # Basic status
            'enabled': True,
            'running': self.running,
            'paused': self.paused,
            
            # Training progress
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'learning_updates': self.learning_updates,
            'uptime_hours': elapsed / 3600,
            'steps_per_sec': self.total_steps / elapsed if elapsed > 0 else 0,
            
            # Learning metrics
            'avg_parameter_change': avg_param_change,
            'recent_parameter_change': recent_param_change,
            'avg_episode_reward': avg_episode_reward,
            'recent_episode_reward': recent_episode_reward,
            'learning_rate': brain_state.params.eta,
            
            # Curiosity metrics
            'current_novelty': self.env.current_novelty,
            'avg_novelty': curiosity_stats['avg_novelty'],
            'novel_state_rate': curiosity_stats['novelty_rate'],
            
            # Stability metrics
            'lyapunov_energy': brain_metrics.get('lyapunov_current', 0.0),
            'dissipation_rate': brain_metrics.get('dissipation_rate', 0.0),
            'is_stable': self.brain.stability_monitor.is_stable(),
            
            # Checkpoint info
            'checkpoint_count': checkpoint_stats['checkpoint_count'],
            'last_checkpoint_step': self.last_checkpoint_step,

            # RLDS-ish metrics trace (optional)
            'rlds_episode_dir': str(self._rlds_episode_dir) if self._rlds_episode_dir else None,
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
            # Optional: where to write RLDS-ish metrics traces for replay/debug.
            # Example: /opt/continuonos/brain/rlds/episodes
            'rlds_log_dir': None,
        }
