"""Sanity tests for the HOPE wave–particle rollout harness."""

import pytest

torch = pytest.importorskip("torch")

from continuonbrain.hope_impl.wave_particle_rollout import (
    WaveParticleRolloutConfig,
    run_wave_particle_rollout,
)


def test_rollout_produces_logs():
    cfg = WaveParticleRolloutConfig(steps=3, seed=1, device="cpu")
    result = run_wave_particle_rollout(cfg)

    assert len(result.logs) == 3
    for log in result.logs:
        assert 0.0 <= log["gate_mean"] <= 1.0
        assert log["wave_norm"] >= 0.0
        assert log["particle_norm"] >= 0.0
        assert log["fusion_norm"] >= 0.0
        assert torch.isfinite(torch.tensor(log["lyapunov"]))


def test_rollout_keeps_state_shapes():
    cfg = WaveParticleRolloutConfig(steps=1, seed=0, device="cpu")
    result = run_wave_particle_rollout(cfg)

    state = result.final_state.fast_state
    assert state.s.ndim == 1
    assert state.w.ndim == 1
    assert state.p.ndim == 1

