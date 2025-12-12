from __future__ import annotations


def test_beam_search_finds_simple_goal() -> None:
    from continuonbrain.mamba_brain import build_world_model
    from continuonbrain.reasoning.arm_state_codec import ArmGoal, state_from_joints
    from continuonbrain.reasoning.tree_search import beam_search_plan

    wm = build_world_model(prefer_mamba=True, joint_dim=6)
    start = state_from_joints([0, 0, 0, 0, 0, 0])
    goal = ArmGoal(target_joint_pos=[0.2, 0, 0, 0, 0, 0])
    plan = beam_search_plan(world_model=wm, start_state=start, goal=goal, horizon=4, beam_width=4, action_step=0.1, time_budget_ms=50)
    assert plan.ok
    assert plan.steps
    assert abs(plan.steps[-1].predicted_state.joint_pos[0] - 0.2) < 1e-6


