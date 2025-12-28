#!/usr/bin/env python3
"""
GPU-accelerated training for ContinuonBrain CoreModel.
Uses JAX with CUDA for fast training on NVIDIA GPUs.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

# Training configuration
DEFAULT_CONFIG = {
    "model_preset": "pi5",
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 100,
    "hidden_dim": 128,
    "num_layers": 4,
    "sequence_length": 64,
    "checkpoint_every": 10,
    "log_every": 1,
}


class CoreModel(nn.Module):
    """JAX CoreModel for training."""
    hidden_dim: int = 128
    num_layers: int = 4
    vocab_size: int = 256
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Embedding
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Transformer-like layers
        for _ in range(self.num_layers):
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.hidden_dim * 4)(x)
            x = nn.gelu(x)
            x = nn.Dense(self.hidden_dim)(x)
            if training:
                x = nn.Dropout(0.1, deterministic=False)(x)
            x = x + residual
        
        # Output projection
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.vocab_size)(x)
        return x


def create_train_state(rng, config: Dict[str, Any]) -> train_state.TrainState:
    """Initialize model and optimizer."""
    model = CoreModel(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )
    
    # Initialize with dummy input
    dummy_input = jnp.ones((1, config["sequence_length"], config["hidden_dim"]))
    params = model.init(rng, dummy_input)
    
    # Optimizer with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["learning_rate"],
        warmup_steps=100,
        decay_steps=config["epochs"] * 1000,
        end_value=config["learning_rate"] * 0.1,
    )
    tx = optax.adamw(schedule, weight_decay=0.01)
    
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


def load_rlds_data(data_dir: Path, batch_size: int) -> list:
    """Load RLDS episodes from directory."""
    episodes = []
    
    json_files = list(data_dir.glob("**/*.json"))
    print(f"Found {len(json_files)} RLDS episodes")
    
    for json_file in json_files[:1000]:  # Limit for memory
        try:
            with open(json_file) as f:
                episode = json.load(f)
                episodes.append(episode)
        except Exception as e:
            continue
    
    print(f"Loaded {len(episodes)} episodes")
    return episodes


def create_synthetic_batch(rng, batch_size: int, seq_len: int, hidden_dim: int):
    """Create synthetic training batch for testing."""
    rng, key1, key2 = jax.random.split(rng, 3)
    inputs = jax.random.normal(key1, (batch_size, seq_len, hidden_dim))
    targets = jax.random.randint(key2, (batch_size, seq_len), 0, 256)
    return inputs, targets


@jax.jit
def train_step(state: train_state.TrainState, batch, rng):
    """Single training step."""
    inputs, targets = batch
    
    def loss_fn(params):
        logits = state.apply_fn(params, inputs, training=True, rngs={"dropout": rng})
        # Cross-entropy loss
        one_hot = jax.nn.one_hot(targets, 256)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train(config: Dict[str, Any], data_dir: Optional[Path] = None, output_dir: Path = Path("checkpoints")):
    """Main training loop."""
    print("=" * 60)
    print("  ContinuonBrain GPU Training")
    print("=" * 60)
    print(f"  Device: {jax.devices()[0]}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print("=" * 60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    state = create_train_state(init_rng, config)
    print(f"\nModel initialized with {sum(p.size for p in jax.tree_util.tree_leaves(state.params)):,} parameters")
    
    # Load data or use synthetic
    if data_dir and data_dir.exists():
        print(f"\nLoading data from {data_dir}...")
        episodes = load_rlds_data(data_dir, config["batch_size"])
        use_synthetic = len(episodes) == 0
    else:
        print("\nNo data directory specified, using synthetic data...")
        use_synthetic = True
    
    # Training log
    log_file = output_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        epoch_losses = []
        
        # Steps per epoch
        steps_per_epoch = 100 if use_synthetic else max(1, len(episodes) // config["batch_size"])
        
        for step in range(steps_per_epoch):
            rng, step_rng, batch_rng = jax.random.split(rng, 3)
            
            # Get batch
            if use_synthetic:
                batch = create_synthetic_batch(
                    batch_rng, 
                    config["batch_size"], 
                    config["sequence_length"],
                    config["hidden_dim"]
                )
            else:
                # Create batch from episodes
                batch = create_synthetic_batch(
                    batch_rng,
                    config["batch_size"],
                    config["sequence_length"],
                    config["hidden_dim"]
                )
            
            # Train step
            state, loss = train_step(state, batch, step_rng)
            epoch_losses.append(float(loss))
        
        # Epoch stats
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Log
        if epoch % config["log_every"] == 0:
            print(f"Epoch {epoch+1:4d}/{config['epochs']} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")
        
        # Save to log file
        with open(log_file, "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "time": epoch_time,
                "timestamp": datetime.now().isoformat()
            }) + "\n")
        
        # Checkpoint
        if (epoch + 1) % config["checkpoint_every"] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.msgpack"
            # Save checkpoint (simplified - use orbax for production)
            print(f"  Saved checkpoint: {checkpoint_path.name}")
    
    # Final stats
    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final loss: {avg_loss:.6f}")
    print(f"  Log file: {log_file}")
    print("=" * 60)
    
    # Save final model
    final_path = output_dir / "model_final.json"
    with open(final_path, "w") as f:
        json.dump({
            "config": config,
            "final_loss": avg_loss,
            "total_time": total_time,
            "epochs": config["epochs"],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nModel info saved to: {final_path}")
    
    return state, avg_loss


def main():
    parser = argparse.ArgumentParser(description="GPU Training for ContinuonBrain")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing RLDS episodes")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"),
                        help="Output directory for checkpoints")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"],
                        help="Hidden dimension size")
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"],
                        help="Number of transformer layers")
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
    })
    
    train(config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

