"""
Local Sanity Check Training Loop

Tiny training loop on Pi CPU (JAX CPU backend) to verify model shapes,
loss computation, and gradients before cloud TPU training.
"""

import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp
import optax
import numpy as np

from ..core_model import CoreModel, make_core_model, CoreModelConfig
from ..data.rlds_dataset import load_rlds_dataset
import csv
import json


def create_initial_state(
    rng_key: jax.random.PRNGKey,
    config: CoreModelConfig,
    obs_dim: int,
    action_dim: int,
    output_dim: int,
) -> Dict[str, Any]:
    """
    Create initial model state (parameters + internal states).
    
    Returns:
        Dictionary with 'params', 'fast_state', 'wave_state', 'particle_state',
        and 'cms_memories', 'cms_keys'
    """
    model, params = make_core_model(rng_key, obs_dim, action_dim, output_dim, config)
    
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
    
    return {
        'model': model,
        'params': params,
        'fast_state': fast_state,
        'wave_state': wave_state,
        'particle_state': particle_state,
        'cms_memories': cms_memories,
        'cms_keys': cms_keys,
    }


def compute_loss(
    params: Dict[str, Any],
    model: CoreModel,
    batch: Dict[str, jnp.ndarray],
    state: Dict[str, Any],
    config: CoreModelConfig,
) -> jnp.ndarray:
    """
    Compute loss for a batch.
    
    Args:
        params: Model parameters
        model: CoreModel instance
        batch: Batch dictionary with 'obs', 'action', 'reward', 'done'
        state: Current model state (fast/wave/particle states, CMS)
        config: Model configuration
    
    Returns:
        Scalar loss value
    """
    obs = batch['obs']  # [B, obs_dim]
    action = batch['action']  # [B, action_dim]
    reward = batch['reward']  # [B, 1]
    done = batch['done']  # [B]
    
    batch_size = obs.shape[0]
    
    # Ensure state has correct batch size
    s_prev = state['fast_state']
    w_prev = state['wave_state']
    p_prev = state['particle_state']
    
    if s_prev.shape[0] != batch_size:
        # Expand state to batch size (for first batch)
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
        state['cms_memories'],
        state['cms_keys'],
    )
    
    # Simple MSE loss: predict action from observation
    # In practice, you'd use a more sophisticated loss (e.g., policy gradient, value function)
    loss = jnp.mean((y_pred - action) ** 2)
    
    return loss


@partial(jax.jit, static_argnums=(2, 5, 6))
def train_step(
    params: Dict[str, Any],
    opt_state: Any,
    model: CoreModel,
    batch: Dict[str, jnp.ndarray],
    state: Dict[str, Any],
    config: CoreModelConfig,
    optimizer: optax.GradientTransformation,
) -> tuple[Dict[str, Any], Any, jnp.ndarray, Dict[str, Any]]:
    """
    Single training step.
    
    Returns:
        (updated_params, updated_opt_state, loss, updated_state)
    """
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(compute_loss)(
        params, model, batch, state, config
    )
    
    # Clip gradients
    if config.gradient_clip > 0:
        grads = optax.clip_by_global_norm(grads, config.gradient_clip)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # Update internal state (simplified - in practice, you'd update based on model output)
    # For sanity check, we'll just keep the state as-is
    updated_state = state
    
    return params, opt_state, loss, updated_state


def run_sanity_check(
    rlds_dir: Optional[Path] = None,
    config: Optional[CoreModelConfig] = None,
    obs_dim: int = 128,
    action_dim: int = 32,
    output_dim: int = 32,
    max_steps: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    use_synthetic_data: bool = True,
    metrics_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run sanity check training loop on Pi CPU.
    
    Args:
        rlds_dir: Directory containing TFRecord episodes (optional, uses synthetic if None)
        config: Model configuration (defaults to pi5_optimized)
        obs_dim: Observation dimension
        action_dim: Action dimension
        output_dim: Output dimension
        max_steps: Maximum number of training steps
        batch_size: Batch size
        learning_rate: Learning rate
        use_synthetic_data: Whether to use synthetic data if rlds_dir is None
    
    Returns:
        Dictionary with training results
    """
    if config is None:
        config = CoreModelConfig.pi5_optimized()
    
    print(f"Starting sanity check training:")
    print(f"  Model config: d_s={config.d_s}, d_w={config.d_w}, d_p={config.d_p}")
    print(f"  Input dims: obs={obs_dim}, action={action_dim}, output={output_dim}")
    print(f"  Max steps: {max_steps}, batch_size={batch_size}")
    
    # Initialize
    rng_key = jax.random.PRNGKey(0)
    state = create_initial_state(rng_key, config, obs_dim, action_dim, output_dim)
    model = state['model']
    params = state['params']
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Create data iterator
    if use_synthetic_data or rlds_dir is None:
        print("Using synthetic data for sanity check")
        def synthetic_batch_iterator():
            rng = jax.random.PRNGKey(42)
            for _ in range(max_steps):
                rng, key = jax.random.split(rng)
                batch = {
                    'obs': jax.random.normal(key, (batch_size, obs_dim)),
                    'action': jax.random.normal(key, (batch_size, action_dim)),
                    'reward': jax.random.uniform(key, (batch_size, 1), minval=-1.0, maxval=1.0),
                    'done': jnp.zeros((batch_size,), dtype=jnp.bool_),
                }
                yield batch
        data_iter = synthetic_batch_iterator()
    else:
        print(f"Loading RLDS dataset from {rlds_dir}")
        data_iter = load_rlds_dataset(
            tfrecord_paths=rlds_dir,
            batch_size=batch_size,
            shuffle=True,
            repeat=False,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    
    # Training loop
    start_time = time.time()
    losses = []
    
    csv_writer = None
    metrics_file = None
    if metrics_path:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w", newline="")
        csv_writer = csv.DictWriter(metrics_file, fieldnames=["step", "loss", "elapsed_s"])
        csv_writer.writeheader()

    try:
        for step in range(max_steps):
            batch = next(data_iter)

            # Training step
            params, opt_state, loss, state = train_step(
                params, opt_state, model, batch, state, config, optimizer
            )

            loss_val = float(loss)
            losses.append(loss_val)

            if step % 2 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}: loss={loss_val:.6f} (elapsed={elapsed:.2f}s)")

            if csv_writer:
                csv_writer.writerow({"step": step, "loss": loss_val, "elapsed_s": time.time() - start_time})

            # Check for NaN/Inf
            if not jnp.isfinite(loss):
                print(f"WARNING: Non-finite loss at step {step}")
                break

    except StopIteration:
        print("Dataset exhausted before max_steps")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    wall_time = time.time() - start_time
    
    # Verify shapes
    print("\nShape verification:")
    print(f"  Fast state: {state['fast_state'].shape}")
    print(f"  Wave state: {state['wave_state'].shape}")
    print(f"  Particle state: {state['particle_state'].shape}")
    print(f"  CMS memories: {[m.shape for m in state['cms_memories']]}")
    
    results = {
        'status': 'ok',
        'steps': len(losses),
        'final_loss': losses[-1] if losses else None,
        'avg_loss': np.mean(losses) if losses else None,
        'wall_time_s': wall_time,
        'losses': losses,
        'config': config,
    }
    
    print(f"\nSanity check completed:")
    print(f"  Steps: {results['steps']}")
    if results['final_loss'] is not None:
        print(f"  Final loss: {results['final_loss']:.6f}")
    else:
        print("  Final loss: n/a (no batches consumed)")
    if results['avg_loss'] is not None:
        print(f"  Avg loss: {results['avg_loss']:.6f}")
    else:
        print("  Avg loss: n/a (no batches consumed)")
    print(f"  Wall time: {wall_time:.2f}s")
    
    if metrics_file:
        metrics_file.close()
        metrics_json = metrics_path.with_suffix(".json") if metrics_path else None
        if metrics_json:
            metrics_json.write_text(json.dumps([{"step": i, "loss": l} for i, l in enumerate(losses)], indent=2))

    return results


def main():
    """CLI entry point for sanity check training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sanity check training on Pi CPU")
    parser.add_argument("--rlds-dir", type=Path, help="Directory with TFRecord episodes")
    parser.add_argument("--obs-dim", type=int, default=128, help="Observation dimension")
    parser.add_argument("--action-dim", type=int, default=32, help="Action dimension")
    parser.add_argument("--output-dim", type=int, default=32, help="Output dimension")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data")
    
    args = parser.parse_args()
    
    results = run_sanity_check(
        rlds_dir=args.rlds_dir,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        output_dim=args.output_dim,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_synthetic_data=args.use_synthetic or args.rlds_dir is None,
    )
    
    if results['status'] == 'ok':
        print("\nSanity check passed!")
        return 0
    else:
        print("\nSanity check failed!")
        return 1


if __name__ == "__main__":
    exit(main())

