from __future__ import annotations
import torch
import torch.nn as nn
from continuonbrain.wavecore.config import WaveCoreConfig
from continuonbrain.wavecore.layers.spectral import SpectralBlock

class HybridBlock(nn.Module):
    """Transformer Attention + Spectral block fusion."""

    def __init__(self, config: WaveCoreConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(config.d_model)
        self.spectral = SpectralBlock(config)
        self.mix_ln = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Multi-head Attention path
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x_attn = self.attn_ln(x + attn_out)

        # Spectral path
        x_wave = self.spectral(x)
        
        # Combined fusion
        x_out = self.mix_ln(x_attn + x_wave)
        return x_out
