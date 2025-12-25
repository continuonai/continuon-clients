from __future__ import annotations
import torch
from continuonbrain.wavecore.models.spectral_lm import SpectralLanguageModel
from continuonbrain.wavecore.utils.synthetic_data import SyntheticDataPipeline

class IntuitionTester:
    """Evaluates the trained Seed Model on logic and sequence intuition."""

    def __init__(self, model: SpectralLanguageModel):
        self.model = model
        self.pipeline = SyntheticDataPipeline(model.config.vocab_size, model.config.seq_len)

    def run_intuition_test(self) -> dict:
        """Run a standard suite of intuition tests."""
        results = {}
        # Test Levels 1-4
        for level in range(1, 5):
            self.pipeline.set_level(level)
            accuracy = self._test_level_accuracy(num_batches=10)
            results[f"level_{level}_accuracy"] = accuracy
            
        return results

    def _test_level_accuracy(self, num_batches: int = 10) -> float:
        self.model.eval()
        total_correct = 0
        total_elements = 0
        
        with torch.no_grad():
            for _ in range(num_batches):
                x, y = self.pipeline.generate_batch(batch_size=8)
                x, y = x.to(self.model.config.device), y.to(self.model.config.device)
                
                logits = self.model(x)
                preds = torch.argmax(logits, dim=-1)
                
                total_correct += (preds == y).sum().item()
                total_elements += y.numel()
                
        return total_correct / total_elements
