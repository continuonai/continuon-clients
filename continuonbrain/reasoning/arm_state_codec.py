"""Arm state/action codecs for planning (arm Searcher).

This module defines:
- how to extract/normalize arm state from RobotService status payloads
- a bounded discrete-ish primitive action set for search
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from continuonbrain.actuators.pca9685_arm import PCA9685ArmController
from continuonbrain.mamba_brain import WorldModelAction, WorldModelState


DEFAULT_JOINT_DIM = 6


@dataclass(frozen=True)
class ArmGoal:
    """Goal expressed as target normalized joint positions (len=6, range [-1,1])."""

    target_joint_pos: List[float]


def clamp_joint_vec(joint_pos: Sequence[float], limit: float = 1.0) -> List[float]:
    return [max(-limit, min(limit, float(v))) for v in joint_pos]


def extract_arm_joint_state_from_status(status: Dict[str, Any]) -> Optional[List[float]]:
    """Extract normalized joint positions list from RobotService.GetRobotStatus response."""
    if not isinstance(status, dict):
        return None
    inner = status.get("status") if "status" in status else status
    if not isinstance(inner, dict):
        return None
    joints = inner.get("joint_positions")
    if isinstance(joints, list) and joints:
        return [float(x) for x in joints]
    return None


def state_from_joints(joint_pos: Sequence[float]) -> WorldModelState:
    return WorldModelState(joint_pos=clamp_joint_vec(joint_pos))


def action_from_delta(joint_delta: Sequence[float]) -> WorldModelAction:
    delta = list(joint_delta)
    if len(delta) != DEFAULT_JOINT_DIM:
        if len(delta) < DEFAULT_JOINT_DIM:
            delta = delta + [0.0] * (DEFAULT_JOINT_DIM - len(delta))
        else:
            delta = delta[:DEFAULT_JOINT_DIM]
    return WorldModelAction(joint_delta=[float(x) for x in delta])


def goal_distance(state: WorldModelState, goal: ArmGoal) -> float:
    """L2 distance in normalized joint space."""
    a = state.joint_pos
    b = goal.target_joint_pos
    n = min(len(a), len(b))
    if n == 0:
        return 1e9
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)) ** 0.5


def build_joint_delta_primitives(
    *,
    joint_dim: int = DEFAULT_JOINT_DIM,
    step: float = 0.05,
    include_noop: bool = True,
) -> List[WorldModelAction]:
    """Generate per-joint +/- delta primitives plus optional noop."""
    actions: List[WorldModelAction] = []
    if include_noop:
        actions.append(action_from_delta([0.0] * joint_dim))
    for j in range(joint_dim):
        plus = [0.0] * joint_dim
        minus = [0.0] * joint_dim
        plus[j] = step
        minus[j] = -step
        actions.append(action_from_delta(plus))
        actions.append(action_from_delta(minus))
    return actions


def apply_action_to_arm(
    arm: PCA9685ArmController,
    current_joint_pos: Sequence[float],
    action: WorldModelAction,
) -> bool:
    """
    Convert a planned delta action into a new normalized joint target and apply to the arm controller.
    """
    if len(current_joint_pos) != DEFAULT_JOINT_DIM:
        return False
    target = [float(j) + float(d) for j, d in zip(current_joint_pos, action.joint_delta)]
    target = clamp_joint_vec(target)
    return arm.set_normalized_action(list(target))


