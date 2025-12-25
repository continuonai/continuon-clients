from __future__ import annotations
import torch
import torch.nn as nn
from continuonbrain.wavecore.config import WaveCoreConfig
from continuonbrain.wavecore.layers.hybrid import HybridBlock

class SpectralLanguageModel(nn.Module):
    """Spectral-hybrid language model for on-device reasoning."""

    def __init__(self, config: WaveCoreConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        
        # Use hybrid blocks for balanced wave/particle dynamics
        self.layers = nn.ModuleList([HybridBlock(config) for _ in range(config.n_layers)])
        
        self.ln_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = idx.shape
        device = idx.device
        
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.ln_out(x)
        logits = self.head(x)
        return logits