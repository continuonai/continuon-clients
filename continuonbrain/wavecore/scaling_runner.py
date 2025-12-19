"Incremental Scaling Runner for WaveCore."
from __future__ import annotations
import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict

from continuonbrain.wavecore.config import WaveCoreConfig
from continuonbrain.wavecore.models.spectral_lm import SpectralLanguageModel
from continuonbrain.resource_monitor import ResourceMonitor, ResourceLevel
from continuonbrain.wavecore.train import generate_dummy_batch

class CheckpointManager:
    """Robust checkpointing for long-running training."""
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, metrics: Dict[str, float], filename: str = "latest.pt"):
        path = self.checkpoint_dir / filename
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": model.config.__dict__ if hasattr(model, "config") else {}
        }, path)
        print(f"Checkpoint saved: {path}")

    def load(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, filename: str = "latest.pt") -> int:
        path = self.checkpoint_dir / filename
        if not path.exists():
            return 0
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Checkpoint loaded: {path} (Step {checkpoint['step']})")
        return checkpoint["step"]

def run_scaling_sweep(output_dir: str = "/tmp/wavecore_sweep"):
    """
    Sweep through model sizes to find stability limit.
    Sizes: d_model=64 -> 512
    """
    monitor = ResourceMonitor()
    sizes = [64, 128, 256, 512]
    
    ckpt_mgr = CheckpointManager(checkpoint_dir=output_dir)
    
    for d_model in sizes:
        print(f"\n=== Testing d_model={d_model} ===")
        
        # Check resources BEFORE starting
        res = monitor.check_resources()
        if res.level in (ResourceLevel.CRITICAL, ResourceLevel.EMERGENCY):
            print(f"🛑 Skipping d_model={d_model}: Insufficient resources ({res.level.value})")
            print(res.message)
            break
            
        config = WaveCoreConfig(
            vocab_size=1024,
            seq_len=64,
            d_model=d_model,
            n_layers=2, # Keep shallow for speed
            loop_type="mid"
        )
        
        try:
            model = SpectralLanguageModel(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            model.train()
            
            # Short training loop
            start_time = time.time()
            for step in range(50):
                x, y = generate_dummy_batch(config, batch_size=8)
                logits = model(x)
                loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if step % 10 == 0:
                    # Check resources DURING training
                    res = monitor.check_resources()
                    mem_used = res.total_memory_mb - res.available_memory_mb
                    print(f"   Step {step}: Loss {loss.item():.4f} | RAM Used: {mem_used}MB")
                    
                    if res.level == ResourceLevel.EMERGENCY:
                        print("🚨 EMERGENCY RESOURCE LEVEL - ABORTING SWEEP")
                        return

            # Save checkpoint if successful
            ckpt_mgr.save(model, optimizer, 50, {"final_loss": loss.item()}, filename=f"model_{d_model}.pt")
            print(f"✅ d_model={d_model} successful.")
            
            # Clean up
            del model
            del optimizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"❌ Failed at d_model={d_model}: {e}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/wavecore_sweep")
    args = parser.parse_args()
    
    run_scaling_sweep(args.output_dir)
