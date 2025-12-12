"""Budgeted tree/beam search for arm planning.

This is a first System-2 style searcher:
- Discrete-ish bounded joint-delta primitives
- Uses the world model to rollout futures
- Prunes unsafe/unhelpful branches
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from continuonbrain.mamba_brain import BaseWorldModel, WorldModelAction, WorldModelState
from .arm_state_codec import ArmGoal, build_joint_delta_primitives, goal_distance


@dataclass
class PlanStep:
    action: WorldModelAction
    predicted_state: WorldModelState
    score: float
    uncertainty: float


@dataclass
class PlanResult:
    ok: bool
    reason: str
    steps: List[PlanStep]
    diagnostics: Dict[str, Any]


def beam_search_plan(
    *,
    world_model: BaseWorldModel,
    start_state: WorldModelState,
    goal: ArmGoal,
    horizon: int = 6,
    beam_width: int = 6,
    action_step: float = 0.05,
    time_budget_ms: int = 150,
    max_uncertainty: float = 0.95,
) -> PlanResult:
    """Plan by beam search over joint-delta primitives."""
    t0 = time.time()
    deadline = t0 + (max(1, int(time_budget_ms)) / 1000.0)

    primitives = build_joint_delta_primitives(step=action_step, include_noop=True)

    # Beam element: (state, steps, score, worst_uncertainty)
    beam: List[Tuple[WorldModelState, List[PlanStep], float, float]] = [(start_state, [], 0.0, 0.0)]

    best: Optional[Tuple[WorldModelState, List[PlanStep], float, float]] = None
    expanded = 0

    def score_state(state: WorldModelState, accum_uncert: float) -> float:
        # Lower distance is better; uncertainty penalizes.
        d = goal_distance(state, goal)
        return -d - 0.25 * accum_uncert

    for depth in range(horizon):
        if time.time() >= deadline:
            break
        next_beam: List[Tuple[WorldModelState, List[PlanStep], float, float]] = []
        for state, steps, _, worst_uncert in beam:
            if time.time() >= deadline:
                break
            for action in primitives:
                if time.time() >= deadline:
                    break
                pred = world_model.predict(state, action)
                expanded += 1
                if pred.uncertainty >= max_uncertainty:
                    continue
                new_worst = max(worst_uncert, float(pred.uncertainty))
                new_steps = steps + [
                    PlanStep(
                        action=action,
                        predicted_state=pred.next_state,
                        score=0.0,
                        uncertainty=float(pred.uncertainty),
                    )
                ]
                s = score_state(pred.next_state, new_worst)
                # Store score on last step for convenience.
                new_steps[-1].score = s
                next_beam.append((pred.next_state, new_steps, s, new_worst))

        if not next_beam:
            break

        # Keep top-K
        next_beam.sort(key=lambda item: item[2], reverse=True)
        beam = next_beam[: max(1, int(beam_width))]
        if best is None or beam[0][2] > best[2]:
            best = beam[0]

    if best is None:
        return PlanResult(
            ok=False,
            reason="no_plan_found",
            steps=[],
            diagnostics={"expanded": expanded, "elapsed_ms": int((time.time() - t0) * 1000)},
        )

    _, steps, score, worst_uncert = best
    return PlanResult(
        ok=True,
        reason="ok",
        steps=steps,
        diagnostics={
            "expanded": expanded,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "horizon": horizon,
            "beam_width": beam_width,
            "action_step": action_step,
            "final_score": score,
            "worst_uncertainty": worst_uncert,
        },
    )


