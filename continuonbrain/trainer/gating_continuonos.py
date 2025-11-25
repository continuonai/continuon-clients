"""
Gating helpers to wire trainer execution to continuonos runtime signals.

Replace the lambdas with real hooks from your continuonos services (idle state,
battery, thermals, teleop activity). Defaults are permissive and should be
overridden on-device.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from .local_lora_trainer import GatingSensors


def read_cpu_temp_c(thermal_path: Path | None = None) -> float:
    path = thermal_path or Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        raw = float(path.read_text().strip())
        # Many SBCs report millidegrees
        return raw / 1000.0 if raw > 200 else raw
    except Exception:
        return 0.0


def read_battery_level(default: float = 1.0) -> float:
    env = os.getenv("BATTERY_LEVEL")
    if env:
        try:
            return float(env)
        except Exception:
            pass
    power_supply = Path("/sys/class/power_supply/BAT0/capacity")
    if power_supply.exists():
        try:
            return float(power_supply.read_text().strip()) / 100.0
        except Exception:
            return default
    return default


def build_pi5_gating(
    robot_idle: Callable[[], bool] | None = None,
    teleop_active: Callable[[], bool] | None = None,
    battery_level: Callable[[], float] | None = None,
    cpu_temp: Callable[[], float] | None = None,
    min_battery: float = 0.4,
    max_temp_c: float = 75.0,
) -> GatingSensors:
    return GatingSensors(
        robot_idle=robot_idle or (lambda: True),
        teleop_active=teleop_active or (lambda: False),
        battery_level=battery_level or (lambda: read_battery_level()),
        cpu_temp_c=cpu_temp or (lambda: read_cpu_temp_c()),
        min_battery=min_battery,
        max_temp_c=max_temp_c,
    )
