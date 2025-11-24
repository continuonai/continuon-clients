"""
Lightweight safety heuristics for offline adapter training.

These helpers keep the trainer free of robot-specific logic while allowing
callers to enforce simple action bounds during shadow evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float))


def _dict_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    keys = set(a.keys()) & set(b.keys())
    if not keys:
        return 0.0
    deltas = []
    for k in keys:
        if _is_number(a[k]) and _is_number(b[k]):
            deltas.append(abs(float(a[k]) - float(b[k])))
    return sum(deltas) / max(1, len(deltas))


def build_simple_action_guards(
    *,
    numeric_abs_limit: float | None = None,
    per_key_bounds: Dict[str, Tuple[float, float]] | None = None,
    delta_weight: float = 1.0,
):
    """
    Returns (action_distance_fn, violates_safety_fn) suitable for ModelHooks.

    numeric_abs_limit: applies to scalar actions (e.g., single throttle value).
    per_key_bounds: applies to dict actions (e.g., {"steering": x, "throttle": y}).
    delta_weight: scaling factor for distance if you want to bias rejection.
    """

    def action_distance(new_action: Any, old_action: Any) -> float:
        if _is_number(new_action) and _is_number(old_action):
            return delta_weight * abs(float(new_action) - float(old_action))
        if isinstance(new_action, Mapping) and isinstance(old_action, Mapping):
            return delta_weight * _dict_distance(new_action, old_action)
        return 0.0

    def violates_safety(action: Any) -> bool:
        if _is_number(action) and numeric_abs_limit is not None:
            return abs(float(action)) > numeric_abs_limit
        if isinstance(action, Mapping) and per_key_bounds:
            for key, (lo, hi) in per_key_bounds.items():
                if key not in action:
                    continue
                try:
                    val = float(action[key])
                except Exception:
                    continue
                if val < lo or val > hi:
                    return True
        return False

    return action_distance, violates_safety
