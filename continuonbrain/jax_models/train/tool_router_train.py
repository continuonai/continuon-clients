from __future__ import annotations

"""
JAX tool-router trainer (text -> tool name).

This is a lightweight, JAX-native bridge to start training tool-use behavior without
full LLM fine-tuning. It learns to map a user request string to a tool name label.

Input data source:
- RLDS JSON episodes imported under /opt/continuonos/brain/rlds/episodes/toolchat_hf*
  (see `continuonbrain/tools/import_tool_calling_dataset_to_rlds.py`).

Artifacts:
- Metrics: /opt/continuonos/brain/trainer/logs/tool_router_metrics.json
- Export:  /opt/continuonos/brain/model/adapters/candidate/tool_router_seed/
          - tool_router.npz (weights)
          - tool_router_manifest.json (label map + config + stats)
"""

import json
import math
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax

    JAX_AVAILABLE = True
except Exception:  # noqa: BLE001
    JAX_AVAILABLE = False


@dataclass
class ToolRouterTrainConfig:
    episodes_root: Path = Path("/opt/continuonos/brain/rlds/episodes")
    include_dirs_prefix: str = "toolchat_hf"
    max_episodes_scan: int = 20000
    top_k_tools: int = 128
    features_dim: int = 4096

    batch_size: int = 64
    max_steps: int = 600
    learning_rate: float = 3e-3
    seed: int = 0

    metrics_path: Path = Path("/opt/continuonos/brain/trainer/logs/tool_router_metrics.json")
    export_dir: Path = Path("/opt/continuonos/brain/model/adapters/candidate/tool_router_seed")


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace("\n", " ").split() if t]


def _stable_bucket(token: str, dim: int) -> int:
    # Stable hashing across processes (avoid Python's randomized hash()).
    h = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()  # noqa: S324
    return int.from_bytes(h[:4], "little") % dim


def featurize(text: str, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    for tok in _tokenize(text):
        vec[_stable_bucket(tok, dim)] += 1.0
    # Normalize for length robustness.
    norm = float(np.linalg.norm(vec) + 1e-6)
    vec /= norm
    return vec


def _iter_toolchat_episode_files(root: Path, prefix: str, limit: int) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for p in sorted(root.glob(f"{prefix}*")):
        if not p.is_dir():
            continue
        files.extend(sorted(p.glob("*.json")))
        if len(files) >= limit:
            break
    return files[:limit]


def _extract_examples_from_episode(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return (user_text, tool_name) pairs.
    We use the first user turn as the text and each tool_call action.name as a label.
    """
    steps = payload.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return []

    user_text: Optional[str] = None
    tool_names: List[str] = []

    for s in steps:
        if not isinstance(s, dict):
            continue
        obs = s.get("observation") or {}
        if isinstance(obs, dict) and obs.get("type") == "chat" and obs.get("role") == "user" and user_text is None:
            content = obs.get("content")
            if isinstance(content, str) and content.strip():
                user_text = content.strip()

        action = s.get("action") or {}
        if isinstance(action, dict) and action.get("type") == "tool_call":
            nm = action.get("name")
            if isinstance(nm, str) and nm.strip():
                tool_names.append(nm.strip())

    if not user_text or not tool_names:
        return []
    return [(user_text, nm) for nm in tool_names]


def build_dataset(cfg: ToolRouterTrainConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    files = _iter_toolchat_episode_files(cfg.episodes_root, cfg.include_dirs_prefix, cfg.max_episodes_scan)
    raw: List[Tuple[str, str]] = []
    tool_counts: Dict[str, int] = {}

    for fp in files:
        try:
            payload = json.loads(fp.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        exs = _extract_examples_from_episode(payload)
        for text, tool in exs:
            raw.append((text, tool))
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    if not raw:
        raise RuntimeError("No tool-call training examples found (need toolchat_hf_* episodes with chat.user + tool_call steps).")

    top_tools = sorted(tool_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(2, cfg.top_k_tools)]
    labels = ["__other__"] + [t for t, _ in top_tools]
    label_to_id = {name: i for i, name in enumerate(labels)}

    xs: List[np.ndarray] = []
    ys: List[int] = []
    dropped = 0
    for text, tool in raw:
        xs.append(featurize(text, cfg.features_dim))
        ys.append(label_to_id.get(tool, 0))
        # Keep dataset bounded in memory if extremely large.
        if len(xs) >= 200000:
            break

    x = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int32)
    meta = {
        "examples": int(x.shape[0]),
        "features_dim": cfg.features_dim,
        "labels": labels,
        "top_tools": [{"name": t, "count": c} for t, c in top_tools[:20]],
        "dropped": dropped,
    }
    return x, y, meta


def train(cfg: ToolRouterTrainConfig) -> Dict[str, Any]:
    if not JAX_AVAILABLE:
        raise ImportError("JAX/optax are required for tool_router_train")

    x_np, y_np, meta = build_dataset(cfg)
    n = x_np.shape[0]
    num_classes = len(meta["labels"])

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    x_np = x_np[perm]
    y_np = y_np[perm]

    # Model params: linear classifier
    key = jax.random.PRNGKey(cfg.seed)
    w_key, _ = jax.random.split(key)
    params = {
        "W": jax.random.normal(w_key, (cfg.features_dim, num_classes), dtype=jnp.float32) * 0.01,
        "b": jnp.zeros((num_classes,), dtype=jnp.float32),
    }

    opt = optax.adam(cfg.learning_rate)
    opt_state = opt.init(params)

    def forward(p: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return x @ p["W"] + p["b"]

    def loss_fn(p: Dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logits = forward(p, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        acc = (jnp.argmax(logits, axis=-1) == y).mean()
        return loss, acc

    @jax.jit
    def step(p: Dict[str, jnp.ndarray], os: optax.OptState, x: jnp.ndarray, y: jnp.ndarray):
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, x, y)
        updates, os2 = opt.update(grads, os, p)
        p2 = optax.apply_updates(p, updates)
        return p2, os2, loss, acc

    start = time.time()
    metrics: List[Dict[str, Any]] = []
    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def batch_iter() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        i = 0
        while True:
            if i + cfg.batch_size > n:
                i = 0
            xb = x_np[i : i + cfg.batch_size]
            yb = y_np[i : i + cfg.batch_size]
            i += cfg.batch_size
            yield xb, yb

    it = batch_iter()
    for s in range(cfg.max_steps):
        xb, yb = next(it)
        params, opt_state, loss, acc = step(
            params,
            opt_state,
            jnp.asarray(xb),
            jnp.asarray(yb),
        )
        if s % 10 == 0 or s == cfg.max_steps - 1:
            metrics.append(
                {
                    "step": int(s),
                    "loss": float(loss),
                    "acc": float(acc),
                    "elapsed_s": float(time.time() - start),
                }
            )

    cfg.metrics_path.write_text(json.dumps(metrics, indent=2))

    # Export artifacts
    cfg.export_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cfg.export_dir / "tool_router.npz"
    np.savez(
        weights_path,
        W=np.asarray(params["W"]),
        b=np.asarray(params["b"]),
    )
    manifest_path = cfg.export_dir / "tool_router_manifest.json"
    manifest = {
        "kind": "tool_router_seed",
        "created_unix_s": int(time.time()),
        "weights": str(weights_path),
        "metrics": str(cfg.metrics_path),
        "labels": meta["labels"],
        "config": {
            "episodes_root": str(cfg.episodes_root),
            "include_dirs_prefix": cfg.include_dirs_prefix,
            "max_episodes_scan": cfg.max_episodes_scan,
            "top_k_tools": cfg.top_k_tools,
            "features_dim": cfg.features_dim,
            "batch_size": cfg.batch_size,
            "max_steps": cfg.max_steps,
            "learning_rate": cfg.learning_rate,
        },
        "dataset": {
            "examples": meta["examples"],
            "top_tools": meta["top_tools"],
        },
        "final": metrics[-1] if metrics else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "status": "ok",
        "examples": meta["examples"],
        "labels": len(meta["labels"]),
        "metrics_path": str(cfg.metrics_path),
        "export_dir": str(cfg.export_dir),
        "manifest": str(manifest_path),
        "weights": str(weights_path),
        "final": manifest["final"],
    }


