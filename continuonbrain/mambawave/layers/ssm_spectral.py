"""SSM with Spectral Augmentation Layer.

Combines selective state space modeling with spectral (FFT-based) processing.
This is the core innovation of MambaWave - using frequency domain filtering
to augment the SSM's state transitions.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMSpectralLayer(nn.Module):
    """Selective State Space Model with Spectral Augmentation.

    Architecture:
    1. Input projection to inner dimension
    2. Parallel paths:
       a) SSM path: Conv1D -> SSM selective scan
       b) Spectral path: FFT -> learnable filter -> iFFT
    3. Gated fusion of both paths
    4. Output projection

    The spectral path provides global context via frequency domain,
    while SSM provides efficient recurrent state tracking.
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        *,
        state_dim: int = 16,
        expand: int = 2,
        conv_dim: int = 4,
        dt_rank: int = 8,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        spectral_decay_init: float = 0.01,
        use_spectral: bool = True,
        use_ssm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.expand = expand
        self.use_spectral = use_spectral
        self.use_ssm = use_ssm

        inner_dim = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, inner_dim * 2, bias=False)

        # SSM components
        if use_ssm:
            self.conv1d = nn.Conv1d(
                inner_dim,
                inner_dim,
                kernel_size=conv_dim,
                padding=conv_dim - 1,
                groups=inner_dim,
            )
            # SSM parameters: A, B, C, D, dt
            self.dt_rank = dt_rank
            self.A_log = nn.Parameter(
                torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(inner_dim, 1))
            )
            self.D = nn.Parameter(torch.ones(inner_dim))

            # Projections for selective B, C, dt
            self.x_proj = nn.Linear(inner_dim, dt_rank + state_dim * 2, bias=False)
            self.dt_proj = nn.Linear(dt_rank, inner_dim, bias=True)

            # Initialize dt bias for proper range
            dt_init_std = dt_rank**-0.5
            inv_dt = torch.exp(
                torch.rand(inner_dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            )
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)

        # Spectral components (WaveCore heritage)
        if use_spectral:
            n_fft = seq_len // 2 + 1
            self.filter_real = nn.Parameter(torch.randn(inner_dim, n_fft) * 0.02)
            self.filter_imag = nn.Parameter(torch.randn(inner_dim, n_fft) * 0.02)
            self.spectral_decay = nn.Parameter(torch.ones(inner_dim) * spectral_decay_init)

        # Fusion gate
        if use_ssm and use_spectral:
            self.fusion_gate = nn.Linear(inner_dim * 2, inner_dim)

        # Output projection
        self.out_proj = nn.Linear(inner_dim, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _ssm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """SSM selective scan forward pass."""
        batch, seq_len, inner_dim = x.shape

        # Conv1D for local context
        x_conv = x.transpose(1, 2)  # (B, D, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim padding
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, D)

        # Project to get dt, B, C
        x_dbc = self.x_proj(x_conv)
        dt_x = x_dbc[:, :, :self.dt_rank]
        B = x_dbc[:, :, self.dt_rank:self.dt_rank + self.state_dim]
        C = x_dbc[:, :, self.dt_rank + self.state_dim:]

        # Compute dt
        dt = F.softplus(self.dt_proj(dt_x))  # (B, L, D)

        # Get A
        A = -torch.exp(self.A_log)  # (D, N)

        # Discretize: dA = exp(dt * A), dB = dt * B
        # Simplified selective scan (not fully optimized like real Mamba)
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, D, N)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)

        # Scan (simplified - real Mamba uses CUDA kernels)
        h = torch.zeros(batch, inner_dim, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t:t+1, :].transpose(1, 2)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D)
        y = y + x_conv * self.D  # Skip connection with D

        return y

    def _spectral_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spectral path forward pass (WaveCore style)."""
        batch, seq_len, inner_dim = x.shape

        # FFT along sequence dimension
        x_time = x.transpose(1, 2)  # (B, D, L)
        x_freq = torch.fft.rfft(x_time, dim=-1)
        n_fft = x_freq.size(-1)

        # Apply learnable complex filter with decay
        filt_real = self.filter_real[:, :n_fft]
        filt_imag = self.filter_imag[:, :n_fft]
        filt = torch.complex(filt_real, filt_imag)

        # Frequency-dependent decay
        freqs = torch.linspace(0, 1, n_fft, device=x.device)
        decay = torch.exp(-self.spectral_decay.unsqueeze(-1) * freqs.unsqueeze(0))
        filt = filt * decay

        # Apply filter
        x_freq_filtered = x_freq * filt.unsqueeze(0)

        # iFFT back to time domain
        x_filtered = torch.fft.irfft(x_freq_filtered, n=seq_len, dim=-1)
        y = x_filtered.transpose(1, 2)  # (B, L, D)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining SSM and spectral paths."""
        residual = x

        # Input projection and split
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        z = F.silu(z)

        # Parallel paths
        if self.use_ssm and self.use_spectral:
            y_ssm = self._ssm_forward(x_inner)
            y_spectral = self._spectral_forward(x_inner)
            # Gated fusion
            y_combined = torch.cat([y_ssm, y_spectral], dim=-1)
            y = torch.sigmoid(self.fusion_gate(y_combined)) * y_ssm + \
                (1 - torch.sigmoid(self.fusion_gate(y_combined))) * y_spectral
        elif self.use_ssm:
            y = self._ssm_forward(x_inner)
        elif self.use_spectral:
            y = self._spectral_forward(x_inner)
        else:
            y = x_inner

        # Gate and project out
        y = y * z
        y = self.out_proj(y)

        # Residual + norm
        return self.norm(y + residual)
