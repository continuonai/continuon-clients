"""
Mamba-like selective state space model (SSM) components (JAX/Flax, no custom kernels).

This module is intentionally "seed-safe":
- Pure JAX/Flax + `lax.scan` (TPU/GPU portable)
- Stable parameterization for the state transition (negative diagonal A)
- Supports both single-step recurrence (for streaming) and sequence scan (for batch training)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    # numerically stable softplus
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


@dataclass(frozen=True)
class SelectiveSSMParams:
    """Runtime-only shape hints (not Flax params)."""

    d_inner: int
    state_dim: int = 1


class SelectiveSSM(nn.Module):
    """
    A minimal selective SSM layer (Mamba-like core).

    - Stable diagonal A (learned)
    - Input-dependent step size Δ_t
    - Factored input/output projections (B_t, C_t) to keep memory small

    State shape:
      - carry: (B, d_inner, state_dim)
      - y:     (B, d_inner)
    """

    d_inner: int
    state_dim: int = 1
    dt_min: float = 1e-4

    @nn.compact
    def __call__(
        self,
        u: jnp.ndarray,
        *,
        carry: Optional[jnp.ndarray] = None,
        dt_scale: float = 1.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply SSM either as a scan (sequence input) or a single step (token input).

        Args:
          u: (B, d_inner) or (B, L, d_inner)
          carry: optional initial carry (B, d_inner, state_dim)

        Returns:
          (new_carry, y) where y is (B, d_inner) or (B, L, d_inner)
        """
        if u.ndim == 2:
            carry_in = carry
            if carry_in is None:
                carry_in = jnp.zeros((u.shape[0], self.d_inner, self.state_dim), dtype=u.dtype)
            carry_out, y = self.step(u, carry_in, dt_scale=dt_scale)
            return carry_out, y

        if u.ndim != 3:
            raise ValueError(f"SelectiveSSM expects u ndim 2 or 3, got {u.ndim}")

        B, L, _ = u.shape
        carry_in = carry
        if carry_in is None:
            carry_in = jnp.zeros((B, self.d_inner, self.state_dim), dtype=u.dtype)

        def _scan_step(c, u_t):
            c2, y_t = self.step(u_t, c, dt_scale=dt_scale)
            return c2, y_t

        carry_out, ys = jax.lax.scan(_scan_step, carry_in, jnp.swapaxes(u, 0, 1))
        y = jnp.swapaxes(ys, 0, 1)
        return carry_out, y

    def step(self, u_t: jnp.ndarray, carry: jnp.ndarray, *, dt_scale: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Single-step recurrence.

        u_t:   (B, d_inner)
        carry: (B, d_inner, state_dim)
        """
        # Stable diagonal A (shared across batch)
        a_raw = self.param("a_raw", nn.initializers.normal(0.02), (self.d_inner, self.state_dim))
        A = -_softplus(a_raw)  # negative => stable

        # Input-dependent parameters
        # Δ_t: per-channel step size
        dt = _softplus(nn.Dense(self.d_inner, name="dt_proj")(u_t)) * dt_scale + self.dt_min  # (B, d_inner)

        # Factored B_t and C_t (scalar per channel) + learned bases over state_dim
        b_t = nn.Dense(self.d_inner, name="b_proj")(u_t)  # (B, d_inner)
        c_t = nn.Dense(self.d_inner, name="c_proj")(u_t)  # (B, d_inner)
        B_base = self.param("B_base", nn.initializers.normal(0.02), (self.d_inner, self.state_dim))
        C_base = self.param("C_base", nn.initializers.normal(0.02), (self.d_inner, self.state_dim))
        D = self.param("D", nn.initializers.zeros, (self.d_inner,))

        # Discretization for diagonal A: Abar = exp(dt * A)
        Abar = jnp.exp(dt[:, :, None] * A[None, :, :])  # (B, d_inner, state_dim)

        # Seed-safe B discretization: Bbar ≈ dt * (b_t * B_base)
        Bbar = dt[:, :, None] * (b_t[:, :, None] * B_base[None, :, :])  # (B, d_inner, state_dim)

        # Update state: x = Abar * x + Bbar * u
        x = Abar * carry + Bbar * u_t[:, :, None]

        # Output: y = sum(C_t * x) + D * u
        Ct = c_t[:, :, None] * C_base[None, :, :]
        y = jnp.sum(Ct * x, axis=-1) + D[None, :] * u_t

        return x, y


class MambaLikeWave(nn.Module):
    """
    Drop-in replacement for `WaveSubsystem` that preserves the public shape (B, d_w).

    Internally, this uses SelectiveSSM and packs state_dim into d_w by reshaping.
    With state_dim=1 (default), this reduces to a stable diagonal selective recurrence.
    """

    d_w: int
    d_in: int
    state_dim: int = 1
    dt_min: float = 1e-4
    dt_scale: float = 1.0

    @nn.compact
    def __call__(self, w_prev: jnp.ndarray, u_in: jnp.ndarray) -> jnp.ndarray:
        if w_prev.ndim != 2:
            raise ValueError(f"w_prev must be (B, d_w), got {w_prev.shape}")

        B = w_prev.shape[0]
        if self.d_w % self.state_dim != 0:
            raise ValueError(f"d_w ({self.d_w}) must be divisible by state_dim ({self.state_dim})")
        d_inner = self.d_w // self.state_dim

        # Project input into the SSM channel space
        u = nn.Dense(d_inner, name="wave_in")(u_in)  # (B, d_inner)

        ssm = SelectiveSSM(d_inner=d_inner, state_dim=self.state_dim, dt_min=self.dt_min, name="ssm")
        carry_prev = jnp.reshape(w_prev, (B, d_inner, self.state_dim))
        carry_next, y = ssm(u, carry=carry_prev, dt_scale=self.dt_scale)  # y: (B, d_inner)

        # Gated residual (Mamba-ish ergonomics)
        gate = nn.sigmoid(nn.Dense(d_inner, name="gate_proj")(u_in))
        y = y * gate

        w_next = jnp.reshape(carry_next, (B, self.d_w))
        # Blend with a small projected output so w has both "state" and "emission" structure
        out = nn.Dense(self.d_w, name="wave_out")(y)
        return 0.5 * w_next + 0.5 * out


