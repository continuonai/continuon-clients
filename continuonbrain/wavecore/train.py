"""WaveCore Training Runner."""
from __future__ import annotations
import argparse
import torch
import torch.nn as nn
from .config import WaveCoreConfig
from .models.spectral_lm import SpectralLanguageModel

def generate_dummy_batch(config: WaveCoreConfig, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, config.vocab_size, (batch_size, config.seq_len))
    y = (x + 1) % config.vocab_size
    return x, y

def train_sanity_check(steps: int = 100, loop_type: str = "fast"):
    print(f"Running Sanity Check ({loop_type} loop)...")
    
    if loop_type == "fast":
        config = WaveCoreConfig.fast_loop()
    elif loop_type == "mid":
        config = WaveCoreConfig.mid_loop()
    else:
        config = WaveCoreConfig.slow_loop()
        
    # Override for speed if needed
    # config.seq_len = 32 
    
    model = SpectralLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(steps):
        x, y = generate_dummy_batch(config, batch_size=8)
        logits = model(x)
        loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")
            
    print("Sanity check complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--loop", type=str, default="fast")
    args = parser.parse_args()
    
    train_sanity_check(args.steps, args.loop)
