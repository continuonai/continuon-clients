"""
Example wiring for Pi 5 Donkeycar adapter training.

This is a placeholder scaffold to be mirrored into the continuonos runtime,
showing how to plug real model loaders, LoRA injection, safety bounds, and
gate conditions into the trainer loop. Fill in the TODOs with your concrete
implementations (Torch, ONNX, or TFLite sidecar).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from continuonbrain.trainer import (
    LocalTrainerJobConfig,
    SafetyGateConfig,
    build_gating_sensors,
    build_simple_action_guards,
    build_torch_hooks,
    make_episode_loader,
    maybe_run_local_training,
)


# ----------------------- Implementation placeholders ----------------------- #


def load_base_model():
    """
    TODO: Load your frozen base policy (Torch).
    Should return a torch.nn.Module with base weights loaded.
    """
    raise NotImplementedError("Implement base model loader for Pi 5 policy.")


def inject_lora(model: Any, layers: Sequence[str]):
    """
    TODO: Add LoRA adapters to the specified layers and return trainable params.
    Example: use peft or custom modules; ensure params require_grad=True.
    """
    raise NotImplementedError("Implement LoRA injection for specified layers.")


def behavior_cloning_loss(pred: Any, target: Any):
    """
    TODO: Return a torch loss comparing predicted actions to target actions.
    """
    raise NotImplementedError("Implement behavior cloning loss.")


# ----------------------- Gate and safety configuration --------------------- #


def build_safety():
    # Example bounds for normalized steering/throttle actions.
    return build_simple_action_guards(
        numeric_abs_limit=None,
        per_key_bounds={"steering": (-1.0, 1.0), "throttle": (-1.0, 1.0)},
        delta_weight=1.0,
    )


def build_gates():
    """
    TODO: Wire to continuonos sensors. Replace lambdas with real hooks.
    """
    return build_gating_sensors(
        robot_idle=lambda: True,
        battery_level=lambda: 1.0,
        cpu_temp_c=lambda: 50.0,
        teleop_active=lambda: False,
        min_battery=0.4,
        max_temp_c=75.0,
    )


# ----------------------- Entry point --------------------------------------- #


def run_pi5_training(config_path: Path) -> Dict[str, Any]:
    cfg = LocalTrainerJobConfig.from_json(config_path)
    action_distance, violates = build_safety()

    hooks = build_torch_hooks(
        base_model_loader=load_base_model,
        lora_injector=inject_lora,
        loss_fn=behavior_cloning_loss,
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
