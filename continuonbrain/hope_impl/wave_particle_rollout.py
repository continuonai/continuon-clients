"""
Wave–Particle duality rollout harness for HOPE.

This module provides a small, Pi-friendly loop that wires together
the existing HOPE components (InputEncoder, CMS read/write, HOPECore)
to demonstrate the wave (SSM) and particle (MLP) paths updating side
by side. It is intentionally lightweight so it can be run on a
Raspberry Pi or dev laptop to verify stability and gating behavior
without a full training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from .config import HOPEConfig
from .core import HOPECore
from .encoders import InputEncoder
from .cms import CMSRead, CMSWrite
from .state import FastState, FullState, Parameters
from .stability import lyapunov_total


@dataclass
class WaveParticleRolloutConfig:
    """Configuration for the rollout harness."""

    steps: int = 8
    seed: int = 0
    device: str = "cpu"
    obs_dim: int = 8
    action_dim: int = 4
    reward_scale: float = 0.1
    config_factory: Callable[[], HOPEConfig] = HOPEConfig.development


@dataclass
class WaveParticleRolloutResult:
    """Container for rollout logs and final state."""

    logs: List[Dict[str, float]]
    final_state: FullState


def _tensor_dtype(dtype_str: str) -> torch.dtype:
    """Map HOPEConfig dtype string to torch dtype."""

    if not hasattr(torch, dtype_str):
        raise ValueError(f"Unknown dtype '{dtype_str}' for torch")
    return getattr(torch, dtype_str)


def run_wave_particle_rollout(config: WaveParticleRolloutConfig) -> WaveParticleRolloutResult:
    """Execute a short rollout that logs wave/particle contributions.

    The rollout keeps the HOPE pieces minimal: we encode a simple vector
    observation, read/write CMS, and step the HOPE core with diagnostics
    enabled so the gate, wave delta, and particle delta are exposed.
    """

    hope_config = config.config_factory()
    device = torch.device(config.device or hope_config.device)
    dtype = _tensor_dtype(hope_config.dtype)

    torch.manual_seed(config.seed)

    encoder = InputEncoder(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        d_e=hope_config.d_e,
        obs_type="vector",
    ).to(device)

    cms_read = CMSRead(
        d_s=hope_config.d_s,
        d_e=hope_config.d_e,
        d_k=hope_config.d_k,
        d_c=hope_config.d_c,
        num_levels=hope_config.num_levels,
        cms_dims=hope_config.cms_dims,
    ).to(device)

    cms_write = CMSWrite(
        d_s=hope_config.d_s,
        d_e=hope_config.d_e,
        d_c=hope_config.d_c,
        d_k=hope_config.d_k,
        num_levels=hope_config.num_levels,
        cms_dims=hope_config.cms_dims,
    ).to(device)

    hope_core = HOPECore(
        hope_config.d_s,
        hope_config.d_w,
        hope_config.d_p,
        hope_config.d_e,
        hope_config.d_c,
        use_layer_norm=hope_config.use_layer_norm,
        saturation_limit=hope_config.state_saturation_limit,
    ).to(device)

    state = FullState.zeros(
        d_s=hope_config.d_s,
        d_w=hope_config.d_w,
        d_p=hope_config.d_p,
        cms_sizes=hope_config.cms_sizes,
        cms_dims=hope_config.cms_dims,
        d_k=hope_config.d_k,
        cms_decays=hope_config.cms_decays,
        eta=hope_config.eta_init,
        device=device,
        dtype=dtype,
    )

    logs: List[Dict[str, float]] = []

    for step in range(config.steps):
        # Synthetic observation/action/reward stream
        obs = torch.randn(config.obs_dim, device=device, dtype=dtype)
        action = torch.randn(config.action_dim, device=device, dtype=dtype)
        reward = torch.tensor(config.reward_scale * step, device=device, dtype=dtype)

        e_t = encoder(obs, action, reward)
        q_t, c_t, _ = cms_read(state.cms, state.fast_state.s, e_t)

        s_t, w_t, p_t, info = hope_core(
            state.fast_state.s,
            state.fast_state.w,
            state.fast_state.p,
            e_t,
            c_t,
            return_info=True,
        )

        cms_next = cms_write(state.cms, s_t, e_t)

        state = FullState(
            fast_state=FastState(s_t, w_t, p_t),
            cms=cms_next,
            params=Parameters(theta=state.params.theta, eta=state.params.eta),
        )

        lyapunov = lyapunov_total(state).item()
        cms_energy = float(sum(level.M.norm().item() for level in state.cms.levels))

        logs.append(
            {
                "step": float(step),
                "gate_mean": float(info["gate"].mean().item()),
                "wave_norm": float(w_t.norm().item()),
                "particle_norm": float(p_t.norm().item()),
                "fusion_norm": float(info["z_t"].norm().item()),
                "cms_energy": cms_energy,
                "lyapunov": lyapunov,
            }
        )

    return WaveParticleRolloutResult(logs=logs, final_state=state)


def _format_log_line(log: Dict[str, float]) -> str:
    return (
        f"step={int(log['step'])} gate={log['gate_mean']:.3f} "
        f"wave_norm={log['wave_norm']:.3f} particle_norm={log['particle_norm']:.3f} "
        f"fusion_norm={log['fusion_norm']:.3f} cms_energy={log['cms_energy']:.3f} "
        f"lyapunov={log['lyapunov']:.3f}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a HOPE wave–particle rollout")
    parser.add_argument("--steps", type=int, default=8, help="Number of rollout steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu/cuda)",
    )
    args = parser.parse_args()

    cfg = WaveParticleRolloutConfig(steps=args.steps, seed=args.seed, device=args.device)
    result = run_wave_particle_rollout(cfg)

    print("Wave–Particle rollout complete. Per-step diagnostics:")
    for log in result.logs:
        print(_format_log_line(log))


if __name__ == "__main__":
    main()
