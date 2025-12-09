"""
CPU inference CLI for JAX CoreModel exports.

Loads an exported JAX model (Orbax checkpoint) and runs a single batch,
saving outputs to disk for inspection.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from .export_jax import load_inference_model


def run_inference(model_path: Path, obs_dim: int, action_dim: int, output_path: Path) -> Path:
    model, params, manifest = load_inference_model(str(model_path))
    cfg = manifest["config"]

    batch_size = 1
    obs = jnp.zeros((batch_size, obs_dim), dtype=jnp.float32)
    act = jnp.zeros((batch_size, action_dim), dtype=jnp.float32)
    rew = jnp.zeros((batch_size, 1), dtype=jnp.float32)

    # Initialize states
    s_prev = jnp.zeros((batch_size, cfg["d_s"]))
    w_prev = jnp.zeros((batch_size, cfg["d_w"]))
    p_prev = jnp.zeros((batch_size, cfg["d_p"]))
    cms_memories = [jnp.zeros((size, dim)) for size, dim in zip(cfg["cms_sizes"], cfg["cms_dims"])]
    cms_keys = [jnp.zeros((size, cfg["d_k"])) for size in cfg["cms_sizes"]]

    y, info = model.apply(
        params,
        obs,
        act,
        rew,
        s_prev,
        w_prev,
        p_prev,
        cms_memories,
        cms_keys,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(y))

    meta = {
        "backend": "jax_cpu",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "output_shape": list(y.shape),
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"✅ Inference saved to {output_path} and {output_path.with_suffix('.json')}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU inference for JAX export")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to exported JAX model directory")
    parser.add_argument("--obs-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("inference_output.npy"))
    args = parser.parse_args()

    run_inference(args.model_path, args.obs_dim, args.action_dim, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

