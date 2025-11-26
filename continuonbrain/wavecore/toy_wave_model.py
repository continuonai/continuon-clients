"""
Toy FFT-based Spectral Language Model and HybridBlock prototype sized for
Raspberry Pi 5 CPU experimentation.

Usage (training sanity check):
    python -m continuonbrain.wavecore.toy_wave_model --steps 200

The default configuration targets Pi-class hardware by keeping the parameter
count and sequence length small. Training prints loss every 50 steps so you can
verify the spectral path is stable and gradients do not explode.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn


class SpectralBlock(nn.Module):
    """1D spectral block with FFT + learnable complex filter.

    This mirrors the design from the user-facing spec:
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


class HybridBlock(nn.Module):
    """Transformer + spectral block fused with LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_ln = nn.LayerNorm(d_model)
        self.spectral = SpectralBlock(d_model, seq_len)
        self.mix_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x_attn = self.attn_ln(x + attn_out)

        x_wave = self.spectral(x)
        x_out = self.mix_ln(x_attn + x_wave)
        return x_out


class SpectralLanguageModel(nn.Module):
    """Minimal spectral language model for next-token prediction."""

    def __init__(self, vocab_size: int, d_model: int = 128, seq_len: int = 64, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList([SpectralBlock(d_model, seq_len) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

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


@dataclass
class TrainingConfig:
    vocab_size: int = 512
    seq_len: int = 64
    batch_size: int = 16
    d_model: int = 128
    n_layers: int = 2
    lr: float = 3e-4
    steps: int = 200
    log_every: int = 50
    device: str = "cpu"


def generate_dummy_batch(config: TrainingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=config.device)
    y = (x + 1) % config.vocab_size
    return x, y


def train_toy_model(config: TrainingConfig) -> None:
    model = SpectralLanguageModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        seq_len=config.seq_len,
        n_layers=config.n_layers,
    ).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(config.steps):
        x, y = generate_dummy_batch(config)
        logits = model(x)
        loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % config.log_every == 0:
            print(f"step {step}, loss {loss.item():.4f}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train toy spectral language model")
    parser.add_argument("--steps", type=int, default=TrainingConfig.steps)
    parser.add_argument("--seq-len", type=int, default=TrainingConfig.seq_len)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--vocab-size", type=int, default=TrainingConfig.vocab_size)
    parser.add_argument("--d-model", type=int, default=TrainingConfig.d_model)
    parser.add_argument("--n-layers", type=int, default=TrainingConfig.n_layers)
    parser.add_argument("--device", type=str, default=TrainingConfig.device)
    parser.add_argument("--log-every", type=int, default=TrainingConfig.log_every)
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    return TrainingConfig(**vars(parser.parse_args()))


def main() -> None:
    config = parse_args()
    torch.manual_seed(42)
    train_toy_model(config)


if __name__ == "__main__":
    main()
