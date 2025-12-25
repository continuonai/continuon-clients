from __future__ import annotations
import torch
import torch.nn as nn
from continuonbrain.wavecore.config import WaveCoreConfig

class SpectralBlock(nn.Module):
    """1D spectral block with FFT + learnable complex filter."""

    def __init__(self, config: WaveCoreConfig):
        super().__init__()
        self.d_model = config.d_model
        self.seq_len = config.seq_len

        self.filter_real = nn.Parameter(torch.randn(config.d_model, config.seq_len) * 0.02)
        self.filter_imag = nn.Parameter(torch.randn(config.d_model, config.seq_len) * 0.02)

        self.decay = nn.Parameter(torch.ones(config.d_model) * config.decay_init)

        self.ff = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral filtering followed by residual MLP."""
        batch_size, seq_len, d_model = x.shape
        
        # Spectral transformation (FFT along sequence)
        x_time = x.transpose(1, 2)
        x_freq = torch.fft.rfft(x_time, dim=-1)
        n_fft = x_freq.size(-1)

        filt_real = self.filter_real[:, :n_fft]
        filt_imag = self.filter_imag[:, :n_fft]
        filt = torch.complex(filt_real, filt_imag)

        # Apply causal masking in frequency domain (simplified)
        # In a real causal spectral model, we'd use a more complex convolution kernel,
        # but for this seed model, we apply a sliding-window-like damping.
        
        # Apply gravity (decay)
        freqs = torch.linspace(0, 1, n_fft, device=x.device)
        decay = torch.exp(-self.decay.unsqueeze(-1) * freqs.unsqueeze(0))
        filt = filt * decay

        filt = filt.unsqueeze(0)
        x_freq_filtered = x_freq * filt
        x_time_filtered = torch.fft.irfft(x_freq_filtered, n=seq_len, dim=-1)
        y = x_time_filtered.transpose(1, 2)

        # Gradient normalization / clipping hook placeholder
        y = y + x
        y = y + self.ff(y)
        return y
