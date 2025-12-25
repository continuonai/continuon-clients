from __future__ import annotations
import torch
import torch.nn as nn
from continuonbrain.wavecore.config import WaveCoreConfig
from continuonbrain.wavecore.models.spectral_lm import SpectralLanguageModel
from continuonbrain.services.checkpoint_manager import CheckpointManager

class WaveCoreTrainer:
    """Helper for training and verifying WaveCore models on-device."""

    def __init__(self, config: WaveCoreConfig, checkpoint_dir: str = "./checkpoints/wavecore"):
        self.config = config
        self.model = SpectralLanguageModel(config).to(config.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        self.total_steps = 0

    def save_checkpoint(self, metric: float | None = None):
        """Save current model state."""
        # Simple bridge to CheckpointManager since it expects a 'brain' object
        # with a save_checkpoint method.
        class BrainProxy:
            def __init__(self, model, optimizer):
                self.model = model
                self.optimizer = optimizer
            def save_checkpoint(self, path):
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, path)
        
        proxy = BrainProxy(self.model, self.optimizer)
        self.checkpoint_manager.save_checkpoint(proxy, self.total_steps, metric=metric)

    def load_latest(self) -> bool:
        """Load the most recent checkpoint if it exists."""
        path = self.checkpoint_manager.load_latest()
        if path and path.exists():
            checkpoint = torch.load(path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Extract step from path or metadata if possible
            return True
        return False

    def generate_dummy_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, self.config.vocab_size, (self.config.batch_size, self.config.seq_len), device=self.config.device)
        y = (x + 1) % self.config.vocab_size
        return x, y

    def train_step(self) -> float:
        self.model.train()
        x, y = self.generate_batch()
        
        # Create causal mask for attention
        mask = torch.triu(torch.ones(self.config.seq_len, self.config.seq_len), diagonal=1).bool().to(self.config.device)
        
        logits = self.model(x, attn_mask=mask)
        loss = self.criterion(logits.view(-1, self.config.vocab_size), y.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer_step = self.optimizer.step()
        
        return loss.item()

    def distillation_step(self, teacher_model: nn.Module | None = None) -> float:
        """Step with teacher guidance."""
        from continuonbrain.wavecore.utils.distillation import DistillationLoss
        self.model.train()
        distill_loss_fn = DistillationLoss()
        
        x, y = self.generate_batch()
        mask = torch.triu(torch.ones(self.config.seq_len, self.config.seq_len), diagonal=1).bool().to(self.config.device)
        
        student_logits = self.model(x, attn_mask=mask)
        
        if teacher_model:
            with torch.no_grad():
                teacher_logits = teacher_model(x)
        else:
            # Mock teacher: slightly noisy version of labels
            teacher_logits = torch.zeros_like(student_logits)
            teacher_logits.scatter_(2, y.unsqueeze(-1), 1.0)
            teacher_logits += torch.randn_like(teacher_logits) * 0.1
            
        loss = distill_loss_fn(
            student_logits.view(-1, self.config.vocab_size),
            teacher_logits.view(-1, self.config.vocab_size),
            y.view(-1)
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def generate_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrapper for data source (could be synthetic or RLDS)."""
        if hasattr(self, "synthetic_pipeline"):
            return self.synthetic_pipeline.generate_batch(self.config.batch_size)
        return self.generate_dummy_batch()

    def self_play_step(self, horizon: int = 10) -> dict:
        """Run a self-play rollout where model outputs become next inputs."""
        self.model.eval()
        device = self.config.device
        
        # Start with random token
        current_input = torch.randint(0, self.config.vocab_size, (1, 1), device=device)
        rollout = [current_input.item()]
        
        with torch.no_grad():
            for _ in range(horizon - 1):
                # Pad/Slice to match seq_len if needed, or implement variable length
                # For this prototype, we'll just use the last token if seq_len=1
                # or a sliding window if larger.
                
                # Mocking a simple autoregressive generation
                logits = self.model(current_input)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                rollout.append(next_token.item())
                current_input = next_token # In real SSM, we'd append or update state
                
        # Simple "Stability" reward: if rollout has variety (not stuck on one token)
        unique_tokens = len(set(rollout))
        reward = unique_tokens / horizon
        
        return {"rollout": rollout, "reward": reward}

    def run_sanity_check(self, steps: int = 100) -> dict:
        """Run a short training run to verify stability."""
        losses = []
        start_time = __import__("time").time()
        
        print(f"Starting WaveCore sanity check ({steps} steps)...")
        for i in range(steps):
            loss = self.train_step()
            losses.append(loss)
            if i % 20 == 0:
                print(f"  Step {i}, loss: {loss:.4f}")
        
        duration = __import__("time").time() - start_time
        return {
            "status": "ok",
            "final_loss": losses[-1],
            "duration_s": duration,
            "avg_loss": sum(losses) / len(losses)
        }
