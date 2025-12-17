"""
HOPE Training Script

Comprehensive training script with web monitoring integration.

Usage:
    python examples/hope_train.py --config pi5_optimized --env random --steps 10000
"""

import sys
import argparse
import time
import torch
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from typing import Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hope_impl.config import HOPEConfig
from hope_impl.brain import HOPEBrain
from hope_impl.environments import create_environment


class HOPETrainer:
    """HOPE brain trainer with web monitoring integration."""
    
    def __init__(
        self,
        config: HOPEConfig,
        env_type: str = 'random',
        checkpoint_dir: str = './checkpoints',
        log_interval: int = 10,
        checkpoint_interval: int = 1000,
    ):
        """
        Initialize trainer.
        
        Args:
            config: HOPE configuration
            env_type: Environment type ('random', 'cartpole', 'custom')
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Steps between log messages
            checkpoint_interval: Steps between checkpoints
        """
        self.config = config
        self.env_type = env_type
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        
        # Create environment
        print(f"Creating {env_type} environment...")
        self.env = create_environment(env_type)
        print(f"  Observation dim: {self.env.obs_dim}")
        print(f"  Action dim: {self.env.action_dim}")
        
        # Create brain
        print(f"Creating HOPE brain...")
        self.brain = HOPEBrain(
            config=config,
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            output_dim=self.env.action_dim,
        )

        param_count = sum(p.numel() for p in self.brain.parameters())
        print(f"  Parameters: {param_count:,}")

        memory = self.brain.get_memory_usage()
        print(f"  Memory: {memory['total']:.2f} MB")

        # Optimizer for supervised auxiliary loss
        self.optimizer = torch.optim.Adam(
            self.brain.parameters(), lr=self.config.learning_rate
        )
        
        # Register with web monitoring
        try:
            from continuonbrain.api.routes import hope_routes
            hope_routes.set_hope_brain(self.brain)
            print(f"  ✓ Registered with web monitoring")
        except ImportError:
            print(f"  ⚠ Web monitoring not available")
        
        # Training state
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards = []
        self.start_time = time.time()

    def _compute_loss(self, y_t: torch.Tensor, x_obs: torch.Tensor, reward: float) -> torch.Tensor:
        """Compute a simple auxiliary loss for gradient-based monitoring."""
        target_action = torch.tanh(x_obs[: self.env.action_dim].to(y_t.device))
        reward_term = torch.tensor(reward, device=y_t.device, dtype=y_t.dtype)

        mse_loss = torch.mean((y_t - target_action) ** 2)
        stability_penalty = self.config.lyapunov_weight * torch.mean(y_t ** 2)

        # Reward term should encourage positive rewards
        return mse_loss + stability_penalty - 0.01 * reward_term
    
    def train(
        self,
        num_steps: int,
        early_stop_on_instability: bool = True,
    ):
        """
        Train for specified number of steps.
        
        Args:
            num_steps: Total steps to train
            early_stop_on_instability: Stop if system becomes unstable
        """
        print(f"\nStarting training for {num_steps} steps...")
        print(f"  Log interval: {self.log_interval}")
        print(f"  Checkpoint interval: {self.checkpoint_interval}")
        print(f"  Early stop on instability: {early_stop_on_instability}")
        print()
        
        # Reset environment
        obs = self.env.reset()
        action = torch.zeros(self.env.action_dim)
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(num_steps):
            # Convert observation to tensor
            x_obs = torch.from_numpy(obs).float()

            # Execute brain step with STEP reward (not cumulative) and stability logging enabled
            state_next, y_t, info = self.brain.step(
                x_obs, action, reward if step > 0 else 0.0, 
                perform_param_update=True, 
                log_stability=True
            )

            # Use output as next action
            action = y_t

            # Step environment
            obs, reward, done = self.env.step(action.detach().numpy())

            # Compute auxiliary loss to drive gradients for monitoring
            loss = self._compute_loss(y_t, x_obs, reward)
            self.optimizer.zero_grad()
            loss.backward()

            gradients = {
                name: param.grad.detach().clone()
                for name, param in self.brain.named_parameters()
                if param.grad is not None
            }

            if self.config.gradient_clip:
                clip_grad_norm_(self.brain.parameters(), self.config.gradient_clip)

            self.optimizer.step()

            # Update stability metrics with gradients
            self.brain.stability_monitor.update(state_next, gradients=gradients)
            metrics = self.brain.stability_monitor.get_metrics()
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Check stability
            if early_stop_on_instability and not self.brain.stability_monitor.is_stable():
                print(f"\n⚠ Instability detected at step {self.total_steps}!")
                print(f"  Lyapunov: {info['lyapunov']:.2f}")
                print(f"  Stopping training...")
                break
            
            # Episode end
            if done:
                self.total_episodes += 1
                self.episode_rewards.append(episode_reward)
                
                # Reset
                obs = self.env.reset()
                action = torch.zeros(self.env.action_dim)
                episode_reward = 0.0
                episode_steps = 0
            
            # Logging
            if (step + 1) % self.log_interval == 0:
                elapsed = time.time() - self.start_time
                steps_per_sec = self.total_steps / elapsed

                gradient_norm = metrics.get("gradient_norm", 0.0)

                print(f"Step {self.total_steps:6d} | "
                      f"Episodes: {self.total_episodes:4d} | "
                      f"V: {info['lyapunov']:8.2f} | "
                      f"||s||: {metrics['state_norm']:6.2f} | "
                      f"||∇||: {gradient_norm:6.2f} | "
                      f"η: {state_next.params.eta:.4f} | "
                      f"Loss: {loss.item():6.4f} | "
                      f"Speed: {steps_per_sec:5.1f} steps/s")
            
            # Checkpointing
            if (step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(step + 1)
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Total episodes: {self.total_episodes}")
        print(f"  Mean episode reward: {sum(self.episode_rewards) / max(len(self.episode_rewards), 1):.2f}")
        
        elapsed = time.time() - self.start_time
        print(f"  Training time: {elapsed:.1f}s")
        print(f"  Steps/sec: {self.total_steps / elapsed:.1f}")
        
        metrics = self.brain.stability_monitor.get_metrics()
        print(f"  Final Lyapunov: {metrics['lyapunov_current']:.2f}")
        print(f"  Stable: {self.brain.stability_monitor.is_stable()}")
        
        memory = self.brain.get_memory_usage()
        print(f"  Memory usage: {memory['total']:.2f} MB")
        print(f"{'='*70}\n")
        
        # Save final checkpoint
        self.save_checkpoint(self.total_steps, final=True)
    
    def save_checkpoint(self, step: int, final: bool = False):
        """Save training checkpoint."""
        suffix = 'final' if final else f'{step:06d}'
        path = self.checkpoint_dir / f'hope_{suffix}.pt'
        
        self.brain.save_checkpoint(str(path))
        print(f"  ✓ Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        self.brain.load_checkpoint(path)
        print(f"  ✓ Checkpoint loaded: {path}")


def main():
    parser = argparse.ArgumentParser(description='Train HOPE brain')
    
    # Configuration
    parser.add_argument('--config', type=str, default='development',
                       choices=['pi5_optimized', 'development'],
                       help='Configuration preset')
    
    # Environment
    parser.add_argument('--env', type=str, default='random',
                       choices=['random', 'cartpole'],
                       help='Environment type')
    
    # Training
    parser.add_argument('--steps', type=int, default=1000,
                       help='Total training steps')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                       help='Steps between checkpoints')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Steps between log messages')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')
    
    # Options
    parser.add_argument('--no-early-stop', action='store_true',
                       help='Disable early stopping on instability')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Load checkpoint before training')
    
    # Web monitoring
    parser.add_argument('--port', type=int, default=8080,
                       help='Web monitoring port (for info only)')
    
    args = parser.parse_args()
    
    # Create config
    if args.config == 'pi5_optimized':
        config = HOPEConfig.pi5_optimized()
    else:
        config = HOPEConfig.development()
    
    print("=" * 70)
    print("HOPE Brain Training")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print(f"Environment: {args.env}")
    print(f"Training steps: {args.steps}")
    print()
    print("Web monitoring available at:")
    print(f"  http://localhost:{args.port}/ui/hope/training")
    print(f"  http://localhost:{args.port}/ui/hope/stability")
    print()
    
    # Create trainer
    trainer = HOPETrainer(
        config=config,
        env_type=args.env,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Train
    try:
        trainer.train(
            num_steps=args.steps,
            early_stop_on_instability=not args.no_early_stop,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint(trainer.total_steps, final=True)
    
    print("Done!")


if __name__ == "__main__":
    main()
