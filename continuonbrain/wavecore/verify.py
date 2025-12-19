"""Verification and Export for WaveCore Seed Model."""
from __future__ import annotations
import torch
import argparse
from continuonbrain.wavecore.config import WaveCoreConfig
from continuonbrain.wavecore.models.spectral_lm import SpectralLanguageModel

def verify_export(model: SpectralLanguageModel, config: WaveCoreConfig):
    print("Verifying exportability...")
    model.eval()
    dummy_input = torch.randint(0, config.vocab_size, (1, config.seq_len))
    
    try:
        # TorchScript Trace
        traced_model = torch.jit.trace(model, dummy_input)
        print("✅ TorchScript Trace successful.")
        
        # Save trace
        torch.jit.save(traced_model, "wavecore_seed.pt")
        print("   Saved to wavecore_seed.pt")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

def run_intuition_test(model: SpectralLanguageModel, config: WaveCoreConfig):
    print("Running Intuition Test (Sequence Completion)...")
    model.eval()
    
    # Very simple test: Can it predict a repeating pattern?
    # Train heavily on "1 2 1 2 1 2" first? 
    # Or just check if loss drops on such a pattern over a few steps.
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Pattern: 0 1 0 1 ...
    seq = torch.tensor([i % 2 for i in range(config.seq_len)]).unsqueeze(0)
    
    print("   Fine-tuning on '0 1 0 1' pattern...")
    initial_loss = 0
    final_loss = 0
    
    for i in range(20):
        logits = model(seq)
        targets = torch.roll(seq, -1, 1) # Shift left
        loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
        
        if i == 0: initial_loss = loss.item()
        final_loss = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"   Loss Delta: {initial_loss:.4f} -> {final_loss:.4f}")
    if final_loss < initial_loss:
        print("✅ Model demonstrates learning capability (Intuition confirmed).")
    else:
        print("⚠️  Model struggled to learn simple pattern.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=64)
    args = parser.parse_args()
    
    config = WaveCoreConfig.fast_loop()
    config.d_model = args.d_model
    
    model = SpectralLanguageModel(config)
    
    run_intuition_test(model, config)
    verify_export(model, config)

if __name__ == "__main__":
    main()
