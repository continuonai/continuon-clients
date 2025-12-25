from __future__ import annotations
import random
import torch
from typing import List, Tuple

class SyntheticDataPipeline:
    """Generates deterministic logic patterns for model training."""

    def __init__(self, vocab_size: int, seq_len: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.level = 1 # 1: Repeat, 2: Increment, 3: Addition, 4: Logic

    def set_level(self, level: int):
        self.level = max(1, min(4, level))

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of synthetic sequences based on current level."""
        if self.level == 1:
            return self._repeat_pattern(batch_size)
        elif self.level == 2:
            return self._increment_pattern(batch_size)
        elif self.level == 3:
            return self._addition_pattern(batch_size)
        else:
            return self._logic_pattern(batch_size)

    def _repeat_pattern(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Level 1: Simple repetition (identity intuition)."""
        x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
        y = x.clone()
        return x, y

    def _increment_pattern(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Level 2: Sequence increment (counting intuition)."""
        x = torch.randint(0, self.vocab_size - 1, (batch_size, self.seq_len))
        y = (x + 1) % self.vocab_size
        return x, y

    def _addition_pattern(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Level 3: Simple addition patterns (A, B, A+B)."""
        # Half sequence is inputs, half is results (simplified)
        x = torch.zeros((batch_size, self.seq_len), dtype=torch.long)
        y = torch.zeros((batch_size, self.seq_len), dtype=torch.long)
        
        for i in range(batch_size):
            a = random.randint(0, self.vocab_size // 3)
            b = random.randint(0, self.vocab_size // 3)
            # Pattern: [a, b, a+b, a, b, a+b, ...]
            row = [a, b, (a + b) % self.vocab_size] * (self.seq_len // 3 + 1)
            x[i] = torch.tensor(row[:self.seq_len])
            y[i] = torch.tensor(row[1:self.seq_len + 1] if len(row) > self.seq_len else row[:self.seq_len])
            
        return x, y

    def _logic_pattern(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Level 4: Basic logic (IF A THEN B)."""
        # If token is even, next is +2, if odd, next is -1
        x = torch.randint(1, self.vocab_size - 2, (batch_size, self.seq_len))
        y = torch.zeros_like(x)
        for b in range(batch_size):
            for s in range(self.seq_len):
                val = x[b, s].item()
                if val % 2 == 0:
                    y[b, s] = (val + 2) % self.vocab_size
                else:
                    y[b, s] = (val - 1) % self.vocab_size
        return x, y
