"""Spectral Block Layer."""
from __future__ import annotations
import torch
import torch.nn as nn

class SpectralBlock(nn.Module):
    """1D spectral block with FFT + learnable complex filter.

    - FFT along the sequence dimension
    - Learnable complex filter with per-channel decay ("gravity")
    - Inverse FFT back to time domain
    - Residual + local feedforward
    """

    def __init__(self, d_model: int, seq_len: int, decay_init: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.filter_real = nn.Parameter(torch.randn(d_model, seq_len) * 0.02)
        self.filter_imag = nn.Parameter(torch.randn(d_model, seq_len) * 0.02)

        self.decay = nn.Parameter(torch.ones(d_model) * decay_init)

        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral filtering followed by residual MLP."""
        batch_size, seq_len, d_model = x.shape
        if seq_len != self.seq_len or d_model != self.d_model:
            raise ValueError(
                f"Expected input shape [B,{self.seq_len},{self.d_model}] but got {x.shape}"
            )

        x_time = x.transpose(1, 2)
        x_freq = torch.fft.rfft(x_time, dim=-1)
        n_fft = x_freq.size(-1)

        filt_real = self.filter_real[:, :n_fft]
        filt_imag = self.filter_imag[:, :n_fft]
        filt = torch.complex(filt_real, filt_imag)

        freqs = torch.linspace(0, 1, n_fft, device=x.device)
        decay = torch.exp(-self.decay.unsqueeze(-1) * freqs.unsqueeze(0))
        filt = filt * decay

        filt = filt.unsqueeze(0)
        x_freq_filtered = x_freq * filt
        x_time_filtered = torch.fft.irfft(x_freq_filtered, n=seq_len, dim=-1)
        y = x_time_filtered.transpose(1, 2)

        y = y + x
        y = y + self.ff(y)
        return y
