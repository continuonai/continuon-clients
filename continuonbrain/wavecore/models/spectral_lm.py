"""Spectral Language Model."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..layers.spectral_block import SpectralBlock
from ..layers.hybrid_block import HybridBlock
from ..config import WaveCoreConfig

class SpectralLanguageModel(nn.Module):
    """Spectral language model for next-token prediction."""

    def __init__(self, config: WaveCoreConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.seq_len = config.seq_len

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        
        # Use HybridBlock for Mid/Slow loops if needed, but sticking to SpectralBlock for pure WaveCore as per toy example
        # unless we want to introduce hybrid now. The plan says "Refactor WaveCore Prototype".
        # I'll stick to SpectralBlock for now but allow easy switching.
        self.layers = nn.ModuleList([
            SpectralBlock(config.d_model, config.seq_len) 
            for _ in range(config.n_layers)
        ])
        
        self.ln_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = idx.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len} but got {seq_len}")

        device = idx.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_out(x)
        logits = self.head(x)
        return logits
