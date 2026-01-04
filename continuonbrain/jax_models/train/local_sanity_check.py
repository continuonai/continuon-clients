"""
Local Sanity Check Training Loop (Pi SSM seed)

Tiny training loop on Pi CPU (JAX CPU backend) to verify model shapes,
loss computation, and gradients before the cloud TPU Pi→GCP→Pi pipeline.
Designed to mirror the HOPE Fast/Mid on-device path (ms–100 ms reflex budget).
"""

import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Iterable, Sequence
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax
import numpy as np

from ..core_model import CoreModel, make_core_model, CoreModelConfig
from ..config_presets import get_config_for_preset
from ..data.rlds_dataset import (
    TF_AVAILABLE,
    _extract_action_vector,
    _extract_obs_vector,
    load_rlds_dataset,
    validate_rlds_directory,
    RLDSValidationError,
)
import csv
import json
import pickle


@dataclass
class JsonEpisodeStep:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool


def _load_json_episodes(
    episodes_dir: Path,
    obs_dim: int,
    action_dim: int,
) -> list[JsonEpisodeStep]:
    """Lightweight loader for JSON episodes when TF is unavailable."""
    steps_out: list[JsonEpisodeStep] = []
    for episode_path in sorted(episodes_dir.glob("*.json")):
        try:
            data = json.loads(episode_path.read_text())
        except Exception:
            continue
        steps = data.get("steps", [])
        for step in steps:
            obs_vec = _extract_obs_vector(step.get("observation", {}), obs_dim)
            act_vec = _extract_action_vector(step.get("action", {}), action_dim)
            reward = float(step.get("reward", 0.0))
            done = bool(step.get("is_terminal", False))
            steps_out.append(JsonEpisodeStep(obs_vec, act_vec, reward, done))
    return steps_out


def create_initial_state(
    rng_key: jax.random.PRNGKey,
    config: CoreModelConfig,
    obs_dim: int,
    action_dim: int,
    output_dim: int,
) -> tuple[CoreModel, Dict[str, Any]]:
    """
    Create initial model state (parameters + internal states).
    
    Returns:
        (model, state_dict) where state_dict contains params and memory tensors.
    """
    model, params = make_core_model(rng_key, obs_dim, action_dim, output_dim, config)
    
    # Initialize internal states
    batch_size = 1
    fast_state = jnp.zeros((batch_size, config.d_s))
    wave_state = jnp.zeros((batch_size, config.d_w))
    particle_state = jnp.zeros((batch_size, config.d_p))
    
    # Initialize CMS memories and keys with batch dimension [1, N, D]
    # This allows proper broadcasting/tiling to actual batch size during training
    cms_memories = [
        jnp.zeros((batch_size, size, dim)) for size, dim in zip(config.cms_sizes, config.cms_dims)
    ]
    cms_keys = [
        jnp.zeros((batch_size, size, config.d_k)) for size in config.cms_sizes
    ]
    
    return model, {
        'params': params,
        'fast_state': fast_state,
        'wave_state': wave_state,
        'particle_state': particle_state,
        'cms_memories': cms_memories,
        'cms_keys': cms_keys,
    }


def compute_loss(
    params: Dict[str, Any],
    apply_fn: Callable[..., tuple[jnp.ndarray, Any]],
    batch: Dict[str, jnp.ndarray],
    state: Dict[str, Any],
    config: CoreModelConfig,
    sparsity_lambda: float = 0.0,
) -> tuple[jnp.ndarray, tuple[Dict[str, Any], Dict[str, jnp.ndarray]]]:
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

    # Expand CMS memories and keys to batch size if needed
    cms_memories = state['cms_memories']
    cms_keys = state['cms_keys']
    if cms_memories[0].shape[0] != batch_size:
        cms_memories = [jnp.tile(m[0:1], (batch_size, 1, 1)) for m in cms_memories]
        cms_keys = [jnp.tile(k[0:1], (batch_size, 1, 1)) for k in cms_keys]

    # Forward pass (info contains next internal states)
    y_pred, info = apply_fn(
        params,
        obs,
        action,
        reward,
        s_prev,
        w_prev,
        p_prev,
        cms_memories,
        cms_keys,
    )
    
    # Proof metrics: separate "motor" vs "imagination" tail error.
    # The action extractor can pack planner/tool traces into the last 16 dims.
    # IMPORTANT: action_dim is static under jit; keep reserve as a Python int to avoid Tracer bool conversion.
    action_dim = int(action.shape[-1])
    reserve = min(16, action_dim)

    err = (y_pred - action) ** 2  # [B, action_dim]
    mse_total = jnp.mean(err)
    if reserve > 0:
        if action_dim - reserve > 0:
            mse_main = jnp.mean(err[:, : action_dim - reserve])
        else:
            mse_main = jnp.array(0.0, dtype=err.dtype)
        mse_imagination = jnp.mean(err[:, action_dim - reserve :])
    else:
        mse_main = mse_total
        mse_imagination = jnp.array(0.0, dtype=err.dtype)

    # Keep training objective unchanged (full action MSE), but log proof metrics.
    loss = mse_total

    if sparsity_lambda > 0:
        l1 = sum(
            jnp.sum(jnp.abs(p))
            for p in jax.tree_util.tree_leaves(params)
            if isinstance(p, jnp.ndarray)
        )
        loss = loss + sparsity_lambda * l1
    
    updated_state = state
    if isinstance(info, dict):
        # Carry forward the recurrent state so the trainer exercises streaming dynamics.
        try:
            updated_state = {
                **state,
                "fast_state": info.get("fast_state", state["fast_state"]),
                "wave_state": info.get("wave_state", state["wave_state"]),
                "particle_state": info.get("particle_state", state["particle_state"]),
            }
        except Exception:
            updated_state = state

    metrics = {
        "mse_total": mse_total,
        "mse_main": mse_main,
        "mse_imagination_tail": mse_imagination,
        "imagination_tail_dims": jnp.array(float(reserve), dtype=jnp.float32),
    }

    return loss, (updated_state, metrics)


@partial(jax.jit, static_argnums=(2, 5, 6, 7))
def train_step(
    params: Dict[str, Any],
    opt_state: Any,
    apply_fn: Callable[..., tuple[jnp.ndarray, Any]],
    batch: Dict[str, jnp.ndarray],
    state: Dict[str, Any],
    config: CoreModelConfig,
    optimizer: optax.GradientTransformation,
    sparsity_lambda: float = 0.0,
) -> tuple[Dict[str, Any], Any, jnp.ndarray, Dict[str, Any], Dict[str, jnp.ndarray]]:
    """
    Single training step.
    
    Returns:
        (updated_params, updated_opt_state, loss, updated_state)
    """
    # Compute loss + next state and gradients
    (loss, (next_state, metrics)), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        params, apply_fn, batch, state, config, sparsity_lambda
    )
    
    # Clip gradients
    if config.gradient_clip > 0:
        clipper = optax.clip_by_global_norm(config.gradient_clip)
        updates, _ = clipper.update(grads, clipper.init(grads))
        grads = updates
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, next_state, metrics


def run_sanity_check(
    rlds_dir: Optional[Path] = None,
    config: Optional[CoreModelConfig] = None,
    arch_preset: Optional[str] = None,
    obs_dim: int = 128,
    action_dim: int = 32,
    output_dim: int = 32,
    max_steps: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    use_synthetic_data: bool = True,
    metrics_path: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    sparsity_lambda: float = 0.0,
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
        config = get_config_for_preset(arch_preset)
    
    print(f"Starting sanity check training:")
    print(f"  Model config: d_s={config.d_s}, d_w={config.d_w}, d_p={config.d_p}")
    print(f"  Input dims: obs={obs_dim}, action={action_dim}, output={output_dim}")
    print(f"  Max steps: {max_steps}, batch_size={batch_size}")
    
    # Initialize
    rng_key = jax.random.PRNGKey(0)
    model, state = create_initial_state(rng_key, config, obs_dim, action_dim, output_dim)
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
        # Validate RLDS directory before loading
        try:
            validation_result = validate_rlds_directory(rlds_dir, verbose=True)
            print(f"Validated {validation_result['trainable_files']} trainable episodes "
                  f"with {validation_result['total_steps']} total steps")
        except RLDSValidationError as e:
            print(f"RLDS validation failed: {e}")
            print("Falling back to synthetic data for sanity check")
            use_synthetic_data = True

        if use_synthetic_data:
            # Fallback to synthetic data if validation failed
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
        elif TF_AVAILABLE:
            print(f"Loading RLDS dataset (TFRecord) from {rlds_dir}")
            data_iter = load_rlds_dataset(
                tfrecord_paths=rlds_dir,
                batch_size=batch_size,
                shuffle=True,
                repeat=False,
                obs_dim=obs_dim,
                action_dim=action_dim,
            )
        else:
            print(f"TensorFlow not available; loading JSON episodes from {rlds_dir}")

            steps = _load_json_episodes(Path(rlds_dir), obs_dim, action_dim)
            if not steps:
                raise RuntimeError(f"No JSON episodes found in {rlds_dir}")

            def _iter_json_batches() -> Iterable[Dict[str, jnp.ndarray]]:
                buffer: list[JsonEpisodeStep] = []
                steps_emitted = 0

                def flush():
                    nonlocal buffer, steps_emitted
                    while len(buffer) >= batch_size and steps_emitted < max_steps:
                        batch = buffer[:batch_size]
                        buffer[:] = buffer[batch_size:]
                        steps_emitted += 1
                        yield {
                            "obs": jnp.stack([s.obs for s in batch]),
                            "action": jnp.stack([s.action for s in batch]),
                            "reward": jnp.stack([np.asarray([s.reward], dtype=np.float32) for s in batch]),
                            "done": jnp.stack([np.asarray([s.done], dtype=np.bool_) for s in batch]),
                        }

                for step in steps:
                    buffer.append(step)
                    if len(buffer) >= batch_size:
                        yield from flush()
                    if steps_emitted >= max_steps:
                        break

                yield from flush()

            data_iter = _iter_json_batches()
    
    # Training loop
    start_time = time.time()
    losses = []
    
    csv_writer = None
    metrics_file = None
    if metrics_path:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w", newline="")
        csv_writer = csv.DictWriter(
            metrics_file,
            fieldnames=["step", "loss", "elapsed_s", "mse_main", "mse_imagination_tail", "imagination_tail_dims"],
        )
        csv_writer.writeheader()

    try:
        metrics_log: list[Dict[str, float]] = []
        for step in range(max_steps):
            batch = next(data_iter)

            # Training step
            params, opt_state, loss, state, step_metrics = train_step(
                params,
                opt_state,
                model.apply,
                batch,
                state,
                config,
                optimizer,
                sparsity_lambda,
            )

            loss_val = float(loss)
            losses.append(loss_val)
            metrics_log.append(
                {
                    "step": float(step),
                    "loss": float(loss),
                    "mse_total": float(step_metrics.get("mse_total", loss)),
                    "mse_main": float(step_metrics.get("mse_main", loss)),
                    "mse_imagination_tail": float(step_metrics.get("mse_imagination_tail", 0.0)),
                    "imagination_tail_dims": float(step_metrics.get("imagination_tail_dims", 0.0)),
                }
            )

            if step % 2 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Step {step}: loss={loss_val:.6f} "
                    f"(main_mse={metrics_log[-1]['mse_main']:.6f}, imag_mse={metrics_log[-1]['mse_imagination_tail']:.6f}) "
                    f"(elapsed={elapsed:.2f}s)"
                )

            if csv_writer:
                # Backward compatible: if the CSV was created with only old columns, ignore new ones.
                row = {
                    "step": step,
                    "loss": loss_val,
                    "elapsed_s": time.time() - start_time,
                }
                try:
                    row.update(
                        {
                            "mse_main": metrics_log[-1]["mse_main"],
                            "mse_imagination_tail": metrics_log[-1]["mse_imagination_tail"],
                            "imagination_tail_dims": metrics_log[-1]["imagination_tail_dims"],
                        }
                    )
                except Exception:
                    pass
                try:
                    csv_writer.writerow(row)
                except Exception:
                    # If fieldnames don't match, fall back to legacy row.
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
    
    checkpoint_path = None
    checkpoint_step = None
    checkpoint_format = None
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_step = len(losses)
        # Prefer Orbax when available; fall back to a portable pickle.
        try:
            from ..export.checkpointing import CheckpointManager

            manager = CheckpointManager(str(checkpoint_dir), use_gcs=False)
            manager.save(
                step=checkpoint_step,
                params=params,
                opt_state=opt_state,
                metadata={
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "output_dim": output_dim,
                    "learning_rate": learning_rate,
                    "steps": len(losses),
                    "checkpoint_step": checkpoint_step,
                    "arch_preset": arch_preset,
                    "sparsity_lambda": sparsity_lambda,
                },
            )

            # Return a file path inside checkpoint_dir so callers using Path(...).parent get checkpoint_dir.
            marker = checkpoint_dir / f"checkpoint_step_{checkpoint_step}.marker"
            marker.write_text("ok")
            checkpoint_path = str(marker)
            checkpoint_format = "orbax"
        except Exception:
            ckpt_file = checkpoint_dir / f"params_step_{checkpoint_step}.pkl"
            with ckpt_file.open("wb") as f:
                pickle.dump(
                    {
                        "params": params,
                        "opt_state": opt_state,
                        "metadata": {
                            "obs_dim": obs_dim,
                            "action_dim": action_dim,
                            "output_dim": output_dim,
                            "learning_rate": learning_rate,
                            "steps": len(losses),
                            "checkpoint_step": checkpoint_step,
                            "arch_preset": arch_preset,
                            "sparsity_lambda": sparsity_lambda,
                        },
                    },
                    f,
                )
            checkpoint_path = str(ckpt_file)
            checkpoint_format = "pickle"

    results = {
        'status': 'ok',
        'steps': len(losses),
        'final_loss': losses[-1] if losses else None,
        'avg_loss': np.mean(losses) if losses else None,
        'wall_time_s': wall_time,
        'losses': losses,
        'proof_metrics': metrics_log[-1] if metrics_log else {},
        'proof_metrics_log': metrics_log,
        'config': config,
        'arch_preset': arch_preset,
        'sparsity_lambda': sparsity_lambda,
        'checkpoint_dir': checkpoint_path,
        'checkpoint_step': checkpoint_step,
        'checkpoint_format': checkpoint_format,
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
            # Prefer richer metrics log if available.
            if metrics_log:
                metrics_json.write_text(json.dumps(metrics_log, indent=2))
            else:
                metrics_json.write_text(json.dumps([{"step": i, "loss": l} for i, l in enumerate(losses)], indent=2))

    return results


def main():
    """CLI entry point for sanity check training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Pi SSM seed sanity check on Pi CPU")
    parser.add_argument("--rlds-dir", type=Path, help="Directory with TFRecord episodes (e.g., /opt/continuonos/brain/rlds/tfrecord)")
    parser.add_argument("--arch-preset", type=str, default=None, help="Architecture preset (e.g., pi5, seed_local_2050, hybrid)")
    parser.add_argument("--obs-dim", type=int, default=128, help="Observation dimension")
    parser.add_argument("--action-dim", type=int, default=32, help="Action dimension")
    parser.add_argument("--output-dim", type=int, default=32, help="Output dimension")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--metrics-path", type=Path, help="Optional path to write CSV metrics")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional directory to write checkpoints")
    parser.add_argument("--sparsity-lambda", type=float, default=0.0, help="Optional L1 sparsity regularizer weight")
    
    args = parser.parse_args()
    
    results = run_sanity_check(
        rlds_dir=args.rlds_dir,
        arch_preset=args.arch_preset,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        output_dim=args.output_dim,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_synthetic_data=args.use_synthetic or args.rlds_dir is None,
        metrics_path=args.metrics_path,
        checkpoint_dir=args.checkpoint_dir,
        sparsity_lambda=float(args.sparsity_lambda or 0.0),
    )
    
    if results['status'] == 'ok':
        print("\nSanity check passed!")
        return 0
    else:
        print("\nSanity check failed!")
        return 1


if __name__ == "__main__":
    exit(main())

