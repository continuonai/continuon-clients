"""MambaWave Block - Complete layer combining SSM, Spectral, and optional Attention.

This is the main building block of the MambaWave architecture.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .ssm_spectral import SSMSpectralLayer


class MambaWaveBlock(nn.Module):
    """Complete MambaWave block with optional attention.

    Architecture per block:
    1. SSMSpectralLayer (Mamba + WaveCore combined)
    2. Optional multi-head attention (for slow loop / high capacity)
    3. Feedforward network
    4. Residual connections and LayerNorm

    This block can be configured for different compute budgets:
    - Fast loop: SSM + Spectral only
    - Mid loop: SSM + Spectral with larger dims
    - Slow loop: Full hybrid with attention
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        *,
        # SSM params
        state_dim: int = 16,
        expand: int = 2,
        conv_dim: int = 4,
        dt_rank: int = 8,
        # Spectral params
        spectral_decay_init: float = 0.01,
        # Feature flags
        use_ssm: bool = True,
        use_spectral: bool = True,
        use_attention: bool = False,
        # Attention params
        n_heads: int = 4,
        window_size: int = 0,
        # General
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_attention = use_attention

        # SSM + Spectral layer
        self.ssm_spectral = SSMSpectralLayer(
            d_model=d_model,
            seq_len=seq_len,
            state_dim=state_dim,
            expand=expand,
            conv_dim=conv_dim,
            dt_rank=dt_rank,
            spectral_decay_init=spectral_decay_init,
            use_spectral=use_spectral,
            use_ssm=use_ssm,
        )

        # Optional attention
        if use_attention:
            self.attn_norm = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.window_size = window_size

        # Feedforward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def _create_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sliding window attention mask."""
        if self.window_size <= 0:
            return None
        mask = torch.ones(seq_len, seq_len, device=device)
        mask = mask.triu(diagonal=1 + self.window_size)
        mask = mask + torch.ones(seq_len, seq_len, device=device).tril(
            diagonal=-(1 + self.window_size)
        )
        return mask.bool()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through MambaWave block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attn_mask: Optional attention mask

        Returns:
            Output tensor of same shape
        """
        # SSM + Spectral
        x = self.ssm_spectral(x)

        # Optional attention
        if self.use_attention:
            residual = x
            x = self.attn_norm(x)

            # Create window mask if needed
            if attn_mask is None and self.window_size > 0:
                attn_mask = self._create_window_mask(x.size(1), x.device)

            attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
            x = residual + attn_out

        # Feedforward
        residual = x
        x = self.ff_norm(x)
        x = residual + self.ff(x)

        return x
