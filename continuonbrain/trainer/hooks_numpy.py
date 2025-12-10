"""
Minimal numpy-based ModelHooks for environments without torch.

This provides a tiny linear model with a simple SGD update so we can
exercise the local trainer without pulling heavy ML dependencies.
The design mirrors the ModelHooks contract used by local_lora_trainer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .local_lora_trainer import ModelHooks, ensure_dir


@dataclass
class _NpModel:
    """Lightweight linear model with numpy weights."""

    weights: np.ndarray | None = None  # shape: [input_dim, output_dim]
    bias: np.ndarray | None = None  # shape: [output_dim]

    def __call__(self, obs_batch: Sequence[Any]) -> np.ndarray:
        X = _obs_to_array(obs_batch)
        self._maybe_init_weights(X.shape[1])
        return X @ self.weights + self.bias

    def _maybe_init_weights(self, input_dim: int, output_dim: int = 2) -> None:
        if self.weights is None or self.bias is None:
            # Small random init to avoid all-zero gradients
            self.weights = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01
            self.bias = np.zeros((output_dim,), dtype=np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"weights": self.weights.tolist(), "bias": self.bias.tolist()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.weights = np.array(state.get("weights"), dtype=np.float32)
        self.bias = np.array(state.get("bias"), dtype=np.float32)


def _obs_to_array(obs_batch: Sequence[Any]) -> np.ndarray:
    """Flatten obs structures into a 2D float array."""
    rows = []
    for obs in obs_batch:
        if isinstance(obs, dict):
            # include nested numeric fields (e.g., pca_state, vec)
            def flatten(d):
                out = []
                for v in d.values():
                    if isinstance(v, dict):
                        out.extend(flatten(v))
                    elif isinstance(v, (int, float)):
                        out.append(float(v))
                return out
            rows.append(flatten(obs))
        elif isinstance(obs, (list, tuple)):
            rows.append([float(v) for v in obs])
        elif isinstance(obs, (int, float)):
            rows.append([float(obs)])
        else:
            rows.append([0.0])
    # Pad shorter rows to the max length
    max_len = max(len(r) for r in rows) if rows else 1
    padded = [r + [0.0] * (max_len - len(r)) for r in rows]
    return np.asarray(padded, dtype=np.float32)


def _action_to_array(actions: Sequence[Any], target_dim: int | None = None) -> np.ndarray:
    rows = []
    for act in actions:
        if isinstance(act, dict):
            # Accept common steering/throttle keys; fall back to numeric values
            if "steering" in act or "throttle" in act:
                rows.append([float(act.get("steering", 0.0)), float(act.get("throttle", 0.0))])
            else:
                rows.append([float(v) for v in act.values() if isinstance(v, (int, float))])
        elif isinstance(act, (list, tuple)):
            rows.append([float(v) for v in act])
        elif isinstance(act, (int, float)):
            rows.append([float(act)])
        else:
            rows.append([0.0])
    max_len = max(len(r) for r in rows) if rows else (target_dim or 1)
    padded = [r + [0.0] * (max_len - len(r)) for r in rows]
    arr = np.asarray(padded, dtype=np.float32)
    if target_dim:
        # Trim or pad to target_dim
        if arr.shape[1] < target_dim:
            arr = np.pad(arr, ((0, 0), (0, target_dim - arr.shape[1])), mode="constant")
        elif arr.shape[1] > target_dim:
            arr = arr[:, :target_dim]
    return arr


def build_numpy_hooks(lr: float = 5e-2, weight_decay: float = 0.0) -> ModelHooks:
    """
    Return ModelHooks backed by a numpy linear model.

    Notes:
    - This is intentionally simple and CPU-friendly.
    - Suitable for smoke tests and constrained devices without torch installed.
    """
    model_box: Dict[str, _NpModel] = {}

    def build_model() -> _NpModel:
        model = _NpModel()
        model_box["model"] = model
        return model

    def attach_lora_adapters(model: _NpModel, layers: Sequence[str]):
        # No-op for numpy model; return model as the single trainable "param".
        return [model]

    def make_optimizer(trainable_params, lr: float, weight_decay: float):
        # Optimizer is implicit; store hyperparameters.
        return {"lr": lr, "weight_decay": weight_decay}

    def train_step(model: _NpModel, optimizer: Dict[str, float], batch: Dict[str, Any]) -> float:
        lr_local = optimizer.get("lr", lr)
        wd = optimizer.get("weight_decay", weight_decay)
        X = _obs_to_array(batch["obs"])
        model._maybe_init_weights(X.shape[1])
        W = model.weights
        b = model.bias
        target = _action_to_array(batch["action"], target_dim=W.shape[1])
        preds = X @ W + b
        error = preds - target
        loss = float(np.mean(error ** 2))
        grad_W = (2.0 / X.shape[0]) * (X.T @ error) + wd * W
        grad_b = (2.0 / X.shape[0]) * np.sum(error, axis=0)
        model.weights = W - lr_local * grad_W
        model.bias = b - lr_local * grad_b
        return loss

    def save_adapters(model: _NpModel, path: Path) -> None:
        ensure_dir(path.parent)
        path.write_text(json.dumps(model.state_dict(), indent=2))

    def load_adapters(model: _NpModel, path: Path) -> _NpModel:
        payload = json.loads(path.read_text())
        model.load_state_dict(payload)
        return model

    def eval_forward(model: _NpModel, obs: Any) -> Any:
        return model([obs])

    return ModelHooks(
        build_model=build_model,
        attach_lora_adapters=attach_lora_adapters,
        make_optimizer=make_optimizer,
        train_step=train_step,
        save_adapters=save_adapters,
        load_adapters=load_adapters,
        eval_forward=eval_forward,
        action_distance=None,
        violates_safety=None,
    )

