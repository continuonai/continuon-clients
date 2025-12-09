"""
TPU Training Script

Full training loop for Google Cloud TPU using the same CoreModel code as Pi.
Uses XLA compilation and TPU mesh configuration for optimal performance.
"""

import time
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from flax import struct
import optax
import numpy as np

from ...core_model import CoreModel, make_core_model, CoreModelConfig
from ...data.rlds_dataset import load_rlds_dataset
from ...export.checkpointing import CheckpointManager


@struct.dataclass
class TrainState:
    """Training state including model parameters and optimizer state."""
    step: int
    params: Dict[str, Any]
    opt_state: Any
    fast_state: jnp.ndarray
    wave_state: jnp.ndarray
    particle_state: jnp.ndarray
    cms_memories: Any
    cms_keys: Any


def create_train_state(
    rng_key: jax.random.PRNGKey,
    config: CoreModelConfig,
    obs_dim: int,
    action_dim: int,
    output_dim: int,
    learning_rate: float = 1e-4,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> TrainState:
    """
    Create initial training state.
    
    Args:
        rng_key: JAX random key
        config: Model configuration
        obs_dim: Observation dimension
        action_dim: Action dimension
        output_dim: Output dimension
        learning_rate: Learning rate
    
    Returns:
        Initial training state
    """
    model, params = make_core_model(rng_key, obs_dim, action_dim, output_dim, config)

    # Create optimizer
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)

    # Resume if checkpoint provided
    step = 0
    if checkpoint is not None:
        params = checkpoint.get("params", params)
        opt_state = checkpoint.get("opt_state", opt_state)
        step = int(checkpoint.get("step", 0))
    
    # Initialize internal states
    batch_size = 1
    fast_state = jnp.zeros((batch_size, config.d_s))
    wave_state = jnp.zeros((batch_size, config.d_w))
    particle_state = jnp.zeros((batch_size, config.d_p))
    
    # Initialize CMS memories and keys
    cms_memories = [
        jnp.zeros((size, dim)) for size, dim in zip(config.cms_sizes, config.cms_dims)
    ]
    cms_keys = [
        jnp.zeros((size, config.d_k)) for size in config.cms_sizes
    ]
    
    return TrainState(
        step=step,
        params=params,
        opt_state=opt_state,
        fast_state=fast_state,
        wave_state=wave_state,
        particle_state=particle_state,
        cms_memories=cms_memories,
        cms_keys=cms_keys,
    )


def compute_loss(
    params: Dict[str, Any],
    model: CoreModel,
    batch: Dict[str, jnp.ndarray],
    state: TrainState,
    config: CoreModelConfig,
) -> jnp.ndarray:
    """
    Compute loss for a batch.
    
    Args:
        params: Model parameters
        model: CoreModel instance
        batch: Batch dictionary with 'obs', 'action', 'reward', 'done'
        state: Current training state
        config: Model configuration
    
    Returns:
        Scalar loss value
    """
    obs = batch['obs']  # [B, obs_dim]
    action = batch['action']  # [B, action_dim]
    reward = batch['reward']  # [B, 1]
    
    batch_size = obs.shape[0]
    
    # Ensure state has correct batch size
    s_prev = state.fast_state
    w_prev = state.wave_state
    p_prev = state.particle_state
    
    if s_prev.shape[0] != batch_size:
        # Expand state to batch size
        s_prev = jnp.tile(s_prev[0:1], (batch_size, 1))
        w_prev = jnp.tile(w_prev[0:1], (batch_size, 1))
        p_prev = jnp.tile(p_prev[0:1], (batch_size, 1))
    
    # Forward pass
    y_pred, info = model.apply(
        params,
        obs,
        action,
        reward,
        s_prev,
        w_prev,
        p_prev,
        state.cms_memories,
        state.cms_keys,
    )
    
    # MSE loss: predict action from observation
    # In practice, you'd use a more sophisticated loss (policy gradient, value function)
    loss = jnp.mean((y_pred - action) ** 2)
    
    return loss


@jax.jit
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    model: CoreModel,
    config: CoreModelConfig,
    optimizer: optax.GradientTransformation,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Single training step (JIT-compiled for TPU).
    
    Args:
        state: Current training state
        batch: Training batch
        model: CoreModel instance
        config: Model configuration
        optimizer: Optimizer
    
    Returns:
        (updated_state, metrics)
    """
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(compute_loss)(
        state.params, model, batch, state, config
    )
    
    # Clip gradients
    if config.gradient_clip > 0:
        grads = optax.clip_by_global_norm(grads, config.gradient_clip)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    
    # Update internal state (simplified - in practice, update based on model output)
    updated_state = state.replace(
        step=state.step + 1,
        params=params,
        opt_state=opt_state,
    )
    
    metrics = {
        'loss': loss,
        'grad_norm': optax.global_norm(grads),
    }
    
    return updated_state, metrics


def setup_tpu_mesh(num_devices: Optional[int] = None) -> Any:
    """
    Set up TPU mesh for data parallelism.
    
    Args:
        num_devices: Number of TPU devices (auto-detected if None)
    
    Returns:
        Mesh configuration
    """
    devices = jax.devices()
    tpu_devices = [d for d in devices if d.device_kind == 'tpu']
    
    if not tpu_devices:
        raise RuntimeError("No TPU devices found")
    
    if num_devices is None:
        num_devices = len(tpu_devices)
    
    # Create device mesh
    mesh_shape = (num_devices,)
    mesh = mesh_utils.create_device_mesh(mesh_shape)
    
    print(f"TPU Mesh: {num_devices} devices")
    return mesh


def run_tpu_training(
    data_path: str,
    output_dir: str,
    config: Optional[CoreModelConfig] = None,
    obs_dim: int = 128,
    action_dim: int = 32,
    output_dim: int = 32,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    num_steps: int = 10000,
    log_every: int = 100,
    checkpoint_every: int = 1000,
    gcs_checkpoint_dir: Optional[str] = None,
    resume: bool = True,
    metrics_path: Optional[str] = None,
    eval_every: Optional[int] = None,
    eval_batches: int = 0,
) -> Dict[str, Any]:
    """
    Run full TPU training loop.
    
    Args:
        data_path: Path to TFRecord data (GCS path supported)
        output_dir: Output directory for checkpoints
        config: Model configuration (defaults to tpu_optimized)
        obs_dim: Observation dimension
        action_dim: Action dimension
        output_dim: Output dimension
        batch_size: Batch size per TPU core
        learning_rate: Learning rate
        num_steps: Number of training steps
        log_every: Log metrics every N steps
        checkpoint_every: Save checkpoint every N steps
        gcs_checkpoint_dir: GCS path for checkpoints (optional)
    
    Returns:
        Dictionary with training results
    """
    if config is None:
        config = CoreModelConfig.tpu_optimized()
    
    print("=" * 60)
    print("TPU Training Configuration")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Output dir: {output_dir}")
    print(f"Model config: d_s={config.d_s}, d_w={config.d_w}, d_p={config.d_p}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Steps: {num_steps}")
    print("=" * 60)
    
    # Setup TPU mesh
    try:
        mesh = setup_tpu_mesh()
        print("✅ TPU mesh configured")
    except RuntimeError as e:
        print(f"⚠️  TPU not available: {e}")
        print("   Falling back to CPU/GPU")
        mesh = None
    
    # Initialize checkpointing
    ckpt_dir = gcs_checkpoint_dir or output_dir
    ckpt_manager = None
    try:
        ckpt_manager = CheckpointManager(
            checkpoint_dir=ckpt_dir,
            use_gcs=ckpt_dir.startswith("gs://"),
        )
    except Exception as e:
        print(f"⚠️  Checkpoint manager unavailable: {e}")

    # Load latest checkpoint if resume enabled
    checkpoint = None
    if resume and ckpt_manager is not None:
        latest = ckpt_manager.get_latest_step()
        if latest is not None:
            try:
                checkpoint = ckpt_manager.load(latest)
                print(f"✅ Resumed from checkpoint step {latest}")
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")

    # Initialize model and training state
    rng_key = jax.random.PRNGKey(0)
    state = create_train_state(
        rng_key,
        config,
        obs_dim,
        action_dim,
        output_dim,
        learning_rate,
        checkpoint=checkpoint,
    )
    model, _ = make_core_model(rng_key, obs_dim, action_dim, output_dim, config)
    
    # Create optimizer
    optimizer = optax.adamw(learning_rate)
    
    # Create data iterator
    print(f"Loading dataset from {data_path}...")
    if not data_path:
        raise ValueError("data_path is required for TPU training")
    data_iter = load_rlds_dataset(
        tfrecord_paths=data_path,
        batch_size=batch_size,
        shuffle=True,
        repeat=True,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    
    # Training loop
    start_time = time.time()
    losses = []
    metrics_log: list[Dict[str, Any]] = []

    csv_writer = None
    metrics_file = None
    if metrics_path:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w", newline="")
        csv_writer = csv.DictWriter(metrics_file, fieldnames=["step", "loss", "grad_norm", "elapsed_s"])
        csv_writer.writeheader()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    try:
        for step in range(num_steps):
            batch = next(data_iter)
            
            # Training step
            state, metrics = train_step(state, batch, model, config, optimizer)
            
            loss_val = float(metrics['loss'])
            losses.append(loss_val)
            
            if step % log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                print(
                    f"Step {step}/{num_steps}: "
                    f"loss={loss_val:.6f}, "
                    f"grad_norm={float(metrics['grad_norm']):.4f}, "
                    f"speed={steps_per_sec:.2f} steps/s"
                )

            if csv_writer:
                elapsed = time.time() - start_time
                row = {
                    "step": int(state.step),
                    "loss": float(loss_val),
                    "grad_norm": float(metrics["grad_norm"]),
                    "elapsed_s": elapsed,
                }
                csv_writer.writerow(row)
                metrics_log.append(row)
            
            # Checkpoint
            if ckpt_manager is not None and step > 0 and step % checkpoint_every == 0:
                try:
                    ckpt_manager.save(
                        step=state.step,
                        params=state.params,
                        opt_state=state.opt_state,
                        metadata={
                            "obs_dim": obs_dim,
                            "action_dim": action_dim,
                            "output_dim": output_dim,
                        },
                    )
                    print(f"  ✅ Checkpoint saved at step {state.step}")
                except Exception as e:
                    print(f"  ⚠️  Checkpoint save failed: {e}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
    total_time = time.time() - start_time
    
    results = {
        'status': 'ok',
        'steps': len(losses),
        'final_loss': losses[-1] if losses else None,
        'avg_loss': np.mean(losses) if losses else None,
        'wall_time_s': total_time,
        'losses': losses,
    }
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Steps: {results['steps']}")
    print(f"Final loss: {results['final_loss']:.6f}")
    print(f"Avg loss: {results['avg_loss']:.6f}")
    print(f"Wall time: {total_time:.2f}s")
    print("=" * 60)
    
    if metrics_file:
        metrics_file.close()

    # Persist metrics JSON if requested
    if metrics_path:
        metrics_json = metrics_path.with_suffix(".json")
        metrics_json.write_text(json.dumps(metrics_log, indent=2))

    return results


def main():
    """CLI entry point for TPU training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TPU training for CoreModel")
    parser.add_argument("--data-path", type=str, required=True, help="Path to TFRecord data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--obs-dim", type=int, default=128, help="Observation dimension")
    parser.add_argument("--action-dim", type=int, default=32, help="Action dimension")
    parser.add_argument("--output-dim", type=int, default=32, help="Output dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-steps", type=int, default=10000, help="Number of steps")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Checkpoint every N steps")
    parser.add_argument("--gcs-checkpoint-dir", type=str, help="GCS checkpoint directory")
    parser.add_argument("--metrics-path", type=str, help="Path to write CSV metrics log (and JSON)")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume")
    
    args = parser.parse_args()
    
    results = run_tpu_training(
        data_path=args.data_path,
        output_dir=args.output_dir,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        gcs_checkpoint_dir=args.gcs_checkpoint_dir,
        metrics_path=args.metrics_path,
        resume=not args.no_resume,
    )
    
    if results['status'] == 'ok':
        print("\n✅ Training completed successfully!")
        return 0
    else:
        print("\n❌ Training failed!")
        return 1


if __name__ == "__main__":
    exit(main())

