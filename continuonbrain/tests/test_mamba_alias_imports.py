from __future__ import annotations


def test_mamba_brain_alias_imports() -> None:
    # Import should succeed even without optional deps installed.
    from continuonbrain.mamba_brain import build_world_model  # noqa: F401


def test_world_model_stub_predict() -> None:
    from continuonbrain.mamba_brain import build_world_model
    from continuonbrain.reasoning.arm_state_codec import state_from_joints, action_from_delta

    wm = build_world_model(prefer_mamba=True, joint_dim=6)
    s = state_from_joints([0, 0, 0, 0, 0, 0])
    a = action_from_delta([0.1, 0, 0, 0, 0, 0])
    out = wm.predict(s, a)
    assert out.next_state.joint_pos[0] == 0.1


