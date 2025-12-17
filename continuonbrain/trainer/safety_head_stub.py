"""
Safety head stub: clamps actions and logs violations.

Intended for Pi 5 offline use as a lightweight guard alongside the policy.
Replace with your real safety classifier/heuristics in continuonos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SafetyResult:
    action: Any
    clamped: bool
    violations: Dict[str, float]


def clamp_action(action: Dict[str, float], bounds: Dict[str, tuple[float, float]]) -> SafetyResult:
    clamped = False
    violations: Dict[str, float] = {}
    safe_action = dict(action)
    for key, (lo, hi) in bounds.items():
        if key not in safe_action:
            continue
        val = float(safe_action[key])
        if val < lo:
            violations[key] = val
            safe_action[key] = lo
            clamped = True
        elif val > hi:
            violations[key] = val
            safe_action[key] = hi
            clamped = True
    return SafetyResult(action=safe_action, clamped=clamped, violations=violations)


def safety_step(action: Any, bounds: Dict[str, tuple[float, float]]) -> SafetyResult:
    """
    Apply bounds and return a SafetyResult; wire this into your control loop before actuation.
    """
    if not isinstance(action, dict):
        return SafetyResult(action=action, clamped=False, violations={})
    return clamp_action(action, bounds)
