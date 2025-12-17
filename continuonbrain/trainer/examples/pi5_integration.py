"""
Example wiring for Pi 5 Donkeycar adapter training.

This is a placeholder scaffold to be mirrored into the continuonos runtime,
showing how to plug real model loaders, LoRA injection, safety bounds, and
gate conditions into the trainer loop. Fill in the TODOs with your concrete
implementations (Torch, ONNX, or TFLite sidecar).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from continuonbrain.trainer import (
    LocalTrainerJobConfig,
    SafetyGateConfig,
    build_gating_sensors,
    build_simple_action_guards,
    build_torch_hooks,
    build_gemma_lora_hooks,
    build_pi5_gating,
    list_local_episodes,
    make_episode_loader,
    maybe_run_local_training,
)


# ----------------------- Minimal model + LoRA utilities -------------------- #


class LoRALinear(nn.Module):
    """
    Lightweight LoRA wrapper for nn.Linear to keep base weights frozen.
    """

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        in_features = base.in_features
        out_features = base.out_features
        # Low-rank factors
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        # Freeze base weights
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = x @ self.A @ self.B * (self.alpha / self.rank)
        return base_out + lora_out


class BasePolicyModel(nn.Module):
    """
    Tiny placeholder policy with named layers that match config targets.
    Replace with your actual policy when integrating into continuonos.
    """

    def __init__(self, obs_dim: int = 8, hidden: int = 32, action_dim: int = 2):
        super().__init__()
        self.name_map = {
            "policy.transformer.blocks.10": "policy_transformer_blocks_10",
            "policy.transformer.blocks.11": "policy_transformer_blocks_11",
        }
        self.layers = nn.ModuleDict(
            {
                "policy_transformer_blocks_10": nn.Linear(obs_dim, hidden),
                "policy_transformer_blocks_11": nn.Linear(hidden, action_dim),
            }
        )

    def forward(self, obs_batch: Iterable[Any]) -> torch.Tensor:
        x = _obs_to_tensor(obs_batch)
        x = F.relu(self.layers["policy_transformer_blocks_10"](x))
        return self.layers["policy_transformer_blocks_11"](x)


def _obs_to_tensor(obs_batch: Iterable[Any]) -> torch.Tensor:
    """
    Convert a batch of observations to a float tensor.
    Accepts lists of numbers or dicts containing a numeric list under 'vec'.
    """
    rows: List[List[float]] = []
    for obs in obs_batch:
        if isinstance(obs, dict):
            if "vec" in obs and isinstance(obs["vec"], (list, tuple)):
                rows.append([float(v) for v in obs["vec"]])
            else:
                rows.append([float(v) for v in obs.values() if isinstance(v, (int, float))])
        elif isinstance(obs, (list, tuple)):
            rows.append([float(v) for v in obs])
        elif isinstance(obs, (int, float)):
            rows.append([float(obs)])
        else:
            rows.append([0.0])
    return torch.tensor(rows, dtype=torch.float32)


def _action_to_tensor(action_batch: Iterable[Any]) -> Tuple[torch.Tensor, List[str]]:
    """
    Convert actions to tensor; supports dicts with steering/throttle or scalars.
    Returns tensor and list of keys used (for logging/debug).
    """
    rows: List[List[float]] = []
    keys: List[str] = []
    for action in action_batch:
        if isinstance(action, dict):
            steering = float(action.get("steering", 0.0))
            throttle = float(action.get("throttle", 0.0))
            rows.append([steering, throttle])
            keys = ["steering", "throttle"]
        elif isinstance(action, (list, tuple)) and len(action) >= 2:
            rows.append([float(action[0]), float(action[1])])
            keys = ["a0", "a1"]
        elif isinstance(action, (int, float)):
            rows.append([float(action), 0.0])
            keys = ["a0", "a1"]
        else:
            rows.append([0.0, 0.0])
            keys = ["a0", "a1"]
    return torch.tensor(rows, dtype=torch.float32), keys


def load_base_model():
    """
    Load a frozen base policy. This example creates a small MLP and loads
    weights from the configured `base_model_path` when provided.
    """
    model = BasePolicyModel()
    return model


def inject_lora(model: Any, layers: Sequence[str]):
    """
    Attach LoRA adapters to target layers. Returns trainable parameters.
    """
    trainable = []
    if not hasattr(model, "layers"):
        return trainable
    for target in layers:
        sanitized = getattr(model, "name_map", {}).get(target, target.replace(".", "_"))
        if sanitized in model.layers and isinstance(model.layers[sanitized], nn.Linear):
            base = model.layers[sanitized]
            lora = LoRALinear(base, rank=4, alpha=1.0)
            model.layers[sanitized] = lora
            trainable.extend(list(lora.parameters()))
    return trainable


def behavior_cloning_loss(pred: Any, target: Any):
    pred_tensor = pred if isinstance(pred, torch.Tensor) else torch.as_tensor(pred)
    target_tensor, _ = _action_to_tensor(target)
    # Align shapes if needed
    if pred_tensor.shape != target_tensor.shape:
        # Pad or trim to match minimal dimension
        min_dim = min(pred_tensor.shape[-1], target_tensor.shape[-1])
        pred_tensor = pred_tensor[..., :min_dim]
        target_tensor = target_tensor[..., :min_dim]
    return F.mse_loss(pred_tensor, target_tensor)


# ----------------------- Gate and safety configuration --------------------- #


def build_safety():
    # Example bounds for normalized steering/throttle actions.
    return build_simple_action_guards(
        numeric_abs_limit=None,
        per_key_bounds={"steering": (-1.0, 1.0), "throttle": (-1.0, 1.0)},
        delta_weight=1.0,
    )


def build_gates():
    return build_pi5_gating(
        robot_idle=None,        # TODO: wire to continuonos runtime
        teleop_active=None,     # TODO: wire to teleop activity flag
        battery_level=None,     # TODO: wire to device battery sensor
        cpu_temp=None,          # TODO: wire to device thermal sensor
        min_battery=0.4,
        max_temp_c=75.0,
    )


# ----------------------- Entry point --------------------------------------- #


def run_pi5_training(config_path: Path) -> Dict[str, Any]:
    cfg = LocalTrainerJobConfig.from_json(config_path)
    # Ensure RLDS dir and minimum episode count before invoking trainer
    if not cfg.rlds_dir.exists():
        raise SystemExit(f"RLDS directory not found: {cfg.rlds_dir}")
    episodes = list_local_episodes(cfg.rlds_dir)
    if len(episodes) < cfg.min_episodes:
        raise SystemExit(f"Not enough episodes in {cfg.rlds_dir} (found {len(episodes)}, need {cfg.min_episodes})")

    weights_path = cfg.base_model_path or Path("/opt/continuonos/brain/model/base_policy.pt")
    if not weights_path.exists():
        raise SystemExit(f"Base model checkpoint not found: {weights_path}")

    def _load_model():
        model = load_base_model()
        if weights_path.exists():
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        for p in model.parameters():
            p.requires_grad = False
        return model

    action_distance, violates = build_safety()

    # Choose between custom LoRA injector or Gemma-specific in-place LoRA hooks.
    hooks = build_gemma_lora_hooks(
        base_model_loader=_load_model,
        loss_fn=behavior_cloning_loss,
        target_linear_names=cfg.lora_layers,
        rank=4,
        alpha=8.0,
        dropout=0.0,
        action_distance_fn=action_distance,
        safety_fn=violates,
    )

    episode_loader = make_episode_loader()
    return maybe_run_local_training(
        cfg=cfg,
        hooks=hooks,
        safety_cfg=SafetyGateConfig(),
        gating=build_gates(),
        episode_loader=episode_loader,
    ).__dict__


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run Pi 5 local LoRA training job.")
    parser.add_argument("--config", type=Path, required=True, help="Path to job config JSON.")
    args = parser.parse_args()

    result = run_pi5_training(args.config)
    print(json.dumps(result, indent=2, default=str))
