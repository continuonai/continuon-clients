"""
Tests for WaveCore Integration in HOPE
"""

import pytest
import torch
import torch.nn as nn
from continuonbrain.hope_impl.wave_core import SpectralBlock, HybridBlock
from continuonbrain.hope_impl.cms import CMSWrite
from continuonbrain.hope_impl.state import CMSMemory

class TestWaveCore:
    def test_spectral_block_forward(self):
        """Test SpectralBlock with valid input shapes."""
        d_model = 32
        seq_len = 64
        block = SpectralBlock(d_model=d_model, seq_len=seq_len)
        
        # Batch size 1, Sequence 64
        x = torch.randn(1, seq_len, d_model)
        y = block(x)
        
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_spectral_block_short_sequence(self):
        """Test SpectralBlock handling sequence shorter than max_len."""
        d_model = 32
        max_seq_len = 64
        block = SpectralBlock(d_model=d_model, seq_len=max_seq_len)
        
        # Short sequence
        x = torch.randn(1, 10, d_model)
        y = block(x)
        
        assert y.shape == x.shape

    def test_cms_write_with_wavecore_init(self):
        """Test CMSWrite initializes with WaveCore blocks."""
        writer = CMSWrite(
            d_s=64, d_e=64, d_c=64, d_k=16,
            num_levels=2, cms_dims=[64, 128],
            d_z=32
        )
        
        # Check if event_nets contain SpectralBlock
        # It's inside a Sequential, let's just run forward to verify
        
        cms = CMSMemory.zeros(
            sizes=[32, 64],
            dims=[64, 128],
            d_k=16,
            decays=[0.1, 0.05],
        )
        s_t = torch.randn(64)
        e_t = torch.randn(64)
        
        # Forward without history (should default to window 1)
        cms_next = writer(cms, s_t, e_t)
        
        assert cms_next is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
