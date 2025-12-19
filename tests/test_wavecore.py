"""Unit tests for WaveCore layers and models."""
import unittest
import torch
from continuonbrain.wavecore.layers.spectral_block import SpectralBlock
from continuonbrain.wavecore.layers.hybrid_block import HybridBlock
from continuonbrain.wavecore.models.spectral_lm import SpectralLanguageModel
from continuonbrain.wavecore.config import WaveCoreConfig

class TestWaveCore(unittest.TestCase):
    def test_spectral_block_shape(self):
        d_model = 32
        seq_len = 16
        block = SpectralBlock(d_model, seq_len)
        x = torch.randn(4, seq_len, d_model) # Batch=4
        y = block(x)
        self.assertEqual(y.shape, x.shape)

    def test_hybrid_block_shape(self):
        d_model = 32
        seq_len = 16
        n_heads = 4
        block = HybridBlock(d_model, n_heads, seq_len)
        x = torch.randn(4, seq_len, d_model)
        y = block(x)
        self.assertEqual(y.shape, x.shape)

    def test_model_forward(self):
        config = WaveCoreConfig(
            vocab_size=100,
            seq_len=16,
            d_model=32,
            n_layers=1
        )
        model = SpectralLanguageModel(config)
        idx = torch.randint(0, 100, (2, 16))
        logits = model(idx)
        self.assertEqual(logits.shape, (2, 16, 100))

if __name__ == "__main__":
    unittest.main()
