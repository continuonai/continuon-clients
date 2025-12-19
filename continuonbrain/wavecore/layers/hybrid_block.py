"""Hybrid Block Layer (Attention + Spectral)."""
from __future__ import annotations
import torch
import torch.nn as nn
from .spectral_block import SpectralBlock

class HybridBlock(nn.Module):
    """Transformer + spectral block fused with LayerNorm.
    
    Supports optional sliding-window attention.
    """

    def __init__(self, d_model: int, n_heads: int, seq_len: int, window_size: int = 0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_ln = nn.LayerNorm(d_model)
        self.spectral = SpectralBlock(d_model, seq_len)
        self.mix_ln = nn.LayerNorm(d_model)
        self.window_size = window_size

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Create sliding window mask if needed and not provided
        if self.window_size > 0 and attn_mask is None:
            seq_len = x.size(1)
            # Banded matrix mask
            attn_mask = torch.ones(seq_len, seq_len, device=x.device).triu(diagonal=1 + self.window_size) 
            attn_mask = attn_mask + torch.ones(seq_len, seq_len, device=x.device).tril(diagonal=-(1 + self.window_size))
            # Convert to boolean/float mask where 1/True means ignored (masked out)
            attn_mask = attn_mask.bool()

        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x_attn = self.attn_ln(x + attn_out)

        x_wave = self.spectral(x)
        x_out = self.mix_ln(x_attn + x_wave)
        return x_out
