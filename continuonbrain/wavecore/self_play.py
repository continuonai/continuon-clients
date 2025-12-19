"""Self-Play and Training Loops."""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Dict

class SelfPlayTrainer:
    """Reinforcement-style training loop on synthetic data."""

    def __init__(self, model: nn.Module, generator, tokenizer=None):
        self.model = model
        self.generator = generator
        # Mock tokenizer for now (would use actual tokenizer in prod)
        self.tokenizer = tokenizer or self._mock_tokenizer

    def _mock_tokenizer(self, text: str) -> torch.Tensor:
        # Simple char/space encoding for toy model
        return torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)

    def train_step(self, optimizer) -> Dict[str, float]:
        """Run one step of self-play training."""
        prompt_str, target_str = self.generator.generate_sample()
        
        # In a real loop, we'd autoregressively generate.
        # Here we just do next-token training on the concatenation.
        full_text = f"{prompt_str} {target_str}"
        tokens = self.tokenizer(full_text).unsqueeze(0) # Batch=1
        
        if tokens.size(1) > 64: # Clip to model seq_len
             tokens = tokens[:, :64]
             
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        
        logits = self.model(inputs)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # "Reward" is just negative loss here, or accuracy if we checked it
        return {"loss": loss.item(), "reward": -loss.item()}

class DistillationLoss(nn.Module):
    """Compare Student logits to Teacher distribution."""
    def __init__(self, temperature=2.0):
        super().__init__()
        self.T = temperature
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        return self.kl(
            torch.log_softmax(student_logits / self.T, dim=-1),
            torch.softmax(teacher_logits / self.T, dim=-1)
        ) * (self.T ** 2)
