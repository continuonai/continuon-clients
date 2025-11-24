"""
Scheduler shim for running the local LoRA trainer when the robot is idle and healthy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .local_lora_trainer import (
    GatingSensors,
    LocalTrainerJobConfig,
    ModelHooks,
    SafetyGateConfig,
    TrainerResult,
    maybe_run_local_training,
)


def build_gating_sensors(
    robot_idle: Callable[[], bool],
    battery_level: Callable[[], float],
    cpu_temp_c: Callable[[], float],
    teleop_active: Callable[[], bool],
    min_battery: float = 0.4,
    max_temp_c: float = 75.0,
) -> GatingSensors:
    return GatingSensors(
        robot_idle=robot_idle,
        battery_level=battery_level,
        cpu_temp_c=cpu_temp_c,
        teleop_active=teleop_active,
        min_battery=min_battery,
        max_temp_c=max_temp_c,
    )


def run_if_idle(
    cfg_path: Path,
    hooks: ModelHooks,
    safety_cfg: SafetyGateConfig,
    sensors: GatingSensors,
) -> TrainerResult:
    cfg = LocalTrainerJobConfig.from_json(cfg_path)
    return maybe_run_local_training(cfg, hooks, safety_cfg, gating=sensors)
