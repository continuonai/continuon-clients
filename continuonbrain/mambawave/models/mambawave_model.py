"""MambaWave Model - Full sequence model combining SSM and Spectral architectures.

This model is the evolution of WaveCore, integrating Mamba SSM for efficient
state tracking with spectral (FFT-based) processing for global context.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MambaWaveConfig
from ..layers.mamba_wave_block import MambaWaveBlock


class MambaWaveModel(nn.Module):
    """Full MambaWave sequence model.

    Architecture:
    1. Token embedding (or continuous input projection)
    2. Positional encoding (optional - SSM has implicit position)
    3. Stack of MambaWaveBlocks
    4. Output head (LM head or task-specific)

    Supports three loop variants:
    - Fast: Minimal latency for real-time inference
    - Mid: Balanced for online learning
    - Slow: Maximum capacity for batch training
    """

    def __init__(self, config: MambaWaveConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Optional positional encoding (SSM has implicit position via state)
        self.use_pos_enc = not config.use_ssm  # Only if no SSM
        if self.use_pos_enc:
            self.pos_enc = nn.Parameter(
                torch.randn(1, config.seq_len, config.d_model) * 0.02
            )

        # Input normalization
        self.input_norm = nn.LayerNorm(config.d_model)

        # Stack of MambaWave blocks
        self.blocks = nn.ModuleList([
            MambaWaveBlock(
                d_model=config.d_model,
                seq_len=config.seq_len,
                state_dim=config.ssm_state_dim,
                expand=config.ssm_expand,
                conv_dim=config.ssm_conv_dim,
                dt_rank=config.get_ssm_dt_rank(),
                spectral_decay_init=config.spectral_decay_init,
                use_ssm=config.use_ssm,
                use_spectral=config.use_spectral,
                use_attention=config.use_attention,
                n_heads=config.n_heads,
                window_size=config.window_size,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            return_hidden: If True, return hidden states from all layers

        Returns:
            Dictionary with:
                - logits: Output logits (batch, seq_len, vocab_size)
                - hidden_states: Optional list of hidden states
        """
        # Embed
        x = self.embedding(input_ids)

        # Add positional encoding if needed
        if self.use_pos_enc:
            seq_len = x.size(1)
            x = x + self.pos_enc[:, :seq_len, :]

        x = self.input_norm(x)

        # Process through blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x)
            if return_hidden:
                hidden_states.append(x)

        # Output
        x = self.output_norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}
        if return_hidden:
            result["hidden_states"] = hidden_states

        return result

    def forward_continuous(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for continuous inputs (e.g., sensor data, joint positions).

        Args:
            x: Continuous input of shape (batch, seq_len, d_model)
            return_hidden: If True, return hidden states

        Returns:
            Dictionary with hidden representations
        """
        # Add positional encoding if needed
        if self.use_pos_enc:
            seq_len = x.size(1)
            x = x + self.pos_enc[:, :seq_len, :]

        x = self.input_norm(x)

        # Process through blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x)
            if return_hidden:
                hidden_states.append(x)

        x = self.output_norm(x)

        result = {"hidden": x}
        if return_hidden:
            result["hidden_states"] = hidden_states

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Generated sequence including input
        """
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # Get context (truncate if needed)
            context = generated[:, -self.config.seq_len:]

            # Forward
            outputs = self.forward(context)
            logits = outputs["logits"][:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: MambaWaveConfig) -> "MambaWaveModel":
        """Create model from config."""
        return cls(config)

    @classmethod
    def fast_loop(cls) -> "MambaWaveModel":
        """Create fast loop variant."""
        return cls(MambaWaveConfig.fast_loop())

    @classmethod
    def mid_loop(cls) -> "MambaWaveModel":
        """Create mid loop variant."""
        return cls(MambaWaveConfig.mid_loop())

    @classmethod
    def slow_loop(cls) -> "MambaWaveModel":
        """Create slow loop variant."""
        return cls(MambaWaveConfig.slow_loop())
