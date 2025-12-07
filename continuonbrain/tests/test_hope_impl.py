"""
Tests for HOPE Implementation

Unit tests for all HOPE components.
"""

import pytest
import torch
import torch.nn as nn

from hope_impl.config import HOPEConfig
from hope_impl.state import FastState, MemoryLevel, CMSMemory, Parameters, FullState
from hope_impl.encoders import InputEncoder, QueryNetwork, OutputDecoder
from hope_impl.cms import CMSRead, CMSWrite
from hope_impl.core import WaveSubsystem, ParticleSubsystem, HOPECore
from hope_impl.learning import NestedLearning
from hope_impl.stability import lyapunov_fast_state, lyapunov_memory, lyapunov_total, StabilityMonitor
from hope_impl.brain import HOPEBrain


class TestStateObjects:
    """Test state creation, serialization, device movement."""
    
    def test_fast_state_creation(self):
        """Test FastState creation and operations."""
        fast = FastState.zeros(d_s=64, d_w=64, d_p=32)
        assert fast.s.shape == (64,)
        assert fast.w.shape == (64,)
        assert fast.p.shape == (32,)
        
        # Test device movement
        if torch.cuda.is_available():
            fast_cuda = fast.to(torch.device('cuda'))
            assert fast_cuda.s.device.type == 'cuda'
    
    def test_memory_level_creation(self):
        """Test MemoryLevel creation."""
        level = MemoryLevel.zeros(N=32, d=64, d_k=16, decay=0.1)
        assert level.M.shape == (32, 64)
        assert level.K.shape == (32, 16)
        assert level.decay == 0.1
    
    def test_cms_memory_creation(self):
        """Test CMSMemory creation."""
        cms = CMSMemory.zeros(
            sizes=[32, 64],
            dims=[64, 128],
            d_k=16,
            decays=[0.1, 0.05],
        )
        assert cms.num_levels == 2
        assert cms.levels[0].M.shape == (32, 64)
        assert cms.levels[1].M.shape == (64, 128)
    
    def test_full_state_creation(self):
        """Test FullState creation."""
        state = FullState.zeros(
            d_s=64, d_w=64, d_p=32,
            cms_sizes=[32, 64],
            cms_dims=[64, 128],
            d_k=16,
            cms_decays=[0.1, 0.05],
        )
        assert state.fast_state.s.shape == (64,)
        assert state.cms.num_levels == 2


class TestEncoders:
    """Test input encoding with various input types."""
    
    def test_vector_encoder(self):
        """Test vector observation encoding."""
        encoder = InputEncoder(obs_dim=10, action_dim=4, d_e=64, obs_type="vector")
        
        x_obs = torch.randn(10)
        a_prev = torch.randn(4)
        r_t = torch.tensor(0.5)
        
        e_t = encoder(x_obs, a_prev, r_t)
        assert e_t.shape == (64,)
    
    def test_query_network(self):
        """Test query generation."""
        query_net = QueryNetwork(d_s=64, d_e=64, d_k=16)
        
        s_prev = torch.randn(64)
        e_t = torch.randn(64)
        
        q_t = query_net(s_prev, e_t)
        assert q_t.shape == (16,)
    
    def test_output_decoder(self):
        """Test output decoding."""
        decoder = OutputDecoder(d_s=64, d_c=64, output_dim=4, output_type="continuous")
        
        s_t = torch.randn(64)
        c_t = torch.randn(64)
        
        y_t = decoder(s_t, c_t)
        assert y_t.shape == (4,)


class TestCMS:
    """Test CMS read/write operations, memory decay."""
    
    def test_cms_read(self):
        """Test CMS read operation."""
        cms_read = CMSRead(
            d_s=64, d_e=64, d_k=16, d_c=64,
            num_levels=2,
            cms_dims=[64, 128],
        )
        
        cms = CMSMemory.zeros(
            sizes=[32, 64],
            dims=[64, 128],
            d_k=16,
            decays=[0.1, 0.05],
        )
        
        s_prev = torch.randn(64)
        e_t = torch.randn(64)
        
        q_t, c_t, attention_weights = cms_read(cms, s_prev, e_t)
        
        assert q_t.shape == (16,)
        assert c_t.shape == (64,)
        assert len(attention_weights) == 2
    
    def test_cms_write(self):
        """Test CMS write operation."""
        cms_write = CMSWrite(
            d_s=64, d_e=64, d_c=64, d_k=16,
            num_levels=2,
            cms_dims=[64, 128],
        )
        
        cms = CMSMemory.zeros(
            sizes=[32, 64],
            dims=[64, 128],
            d_k=16,
            decays=[0.1, 0.05],
        )
        
        s_t = torch.randn(64)
        e_t = torch.randn(64)
        
        cms_new = cms_write(cms, s_t, e_t)
        
        assert cms_new.num_levels == 2
        # Memory should have changed (decay + write)
        assert not torch.allclose(cms_new.levels[0].M, cms.levels[0].M)


class TestHOPECore:
    """Test wave/particle dynamics, gating."""
    
    def test_wave_subsystem(self):
        """Test wave subsystem."""
        wave = WaveSubsystem(d_w=64, d_c=64, d_z=32)
        
        w_prev = torch.randn(64)
        z_t = torch.randn(32)
        c_t = torch.randn(64)
        
        w_t = wave(w_prev, z_t, c_t)
        assert w_t.shape == (64,)
    
    def test_particle_subsystem(self):
        """Test particle subsystem."""
        particle = ParticleSubsystem(d_p=32, d_z=32, d_c=64)
        
        p_prev = torch.randn(32)
        z_t = torch.randn(32)
        c_t = torch.randn(64)
        
        p_t = particle(p_prev, z_t, c_t)
        assert p_t.shape == (32,)
    
    def test_hope_core(self):
        """Test full HOPE core."""
        core = HOPECore(d_s=64, d_w=64, d_p=32, d_e=64, d_c=64)
        
        fast_prev = FastState.zeros(d_s=64, d_w=64, d_p=32)
        e_t = torch.randn(64)
        c_t = torch.randn(64)
        
        fast_next = core(fast_prev, e_t, c_t)
        
        assert fast_next.s.shape == (64,)
        assert fast_next.w.shape == (64,)
        assert fast_next.p.shape == (32,)


class TestStability:
    """Test Lyapunov functions, stability monitoring."""
    
    def test_lyapunov_fast_state(self):
        """Test fast state Lyapunov function."""
        fast = FastState.randn(d_s=64, d_w=64, d_p=32)
        V = lyapunov_fast_state(fast)
        assert V.item() > 0
    
    def test_lyapunov_memory(self):
        """Test memory Lyapunov function."""
        cms = CMSMemory.randn(
            sizes=[32, 64],
            dims=[64, 128],
            d_k=16,
            decays=[0.1, 0.05],
        )
        V = lyapunov_memory(cms)
        assert V.item() > 0
    
    def test_lyapunov_total(self):
        """Test total Lyapunov function."""
        state = FullState.randn(
            d_s=64, d_w=64, d_p=32,
            cms_sizes=[32, 64],
            cms_dims=[64, 128],
            d_k=16,
            cms_decays=[0.1, 0.05],
        )
        V = lyapunov_total(state)
        assert V.item() > 0
    
    def test_stability_monitor(self):
        """Test stability monitor."""
        monitor = StabilityMonitor(window_size=10)
        
        state = FullState.zeros(
            d_s=64, d_w=64, d_p=32,
            cms_sizes=[32, 64],
            cms_dims=[64, 128],
            d_k=16,
            cms_decays=[0.1, 0.05],
        )
        
        monitor.update(state)
        metrics = monitor.get_metrics()

        assert 'lyapunov_current' in metrics
        assert 'state_norm' in metrics
        assert monitor.is_stable()

    def test_stability_monitor_missing_gradients(self):
        """Ensure gradient norms align with step count even when absent."""
        monitor = StabilityMonitor(window_size=5)

        state = FullState.zeros(
            d_s=8, d_w=8, d_p=4,
            cms_sizes=[4],
            cms_dims=[8],
            d_k=2,
            cms_decays=[0.1],
        )

        monitor.update(state, gradients={"dummy": torch.ones(1)})
        monitor.update(state)  # No gradients provided

        metrics = monitor.get_metrics()

        assert metrics.get('gradient_norm') == 0.0
        energetic_state = FullState.zeros(
            d_s=64, d_w=64, d_p=32,
            cms_sizes=[32, 64],
            cms_dims=[64, 128],
            d_k=16,
            cms_decays=[0.1, 0.05],
        )
        energetic_state.fast_state.s = torch.ones_like(energetic_state.fast_state.s) * 2.0
        monitor.update(energetic_state, gradients={"g": torch.tensor([0.1])})

        assert not monitor.is_stable(), "Energy increase with negative dissipation should flag instability"

        gradient_monitor = StabilityMonitor(
            window_size=5,
            lyapunov_threshold=100.0,
            dissipation_floor=-1.0,
            gradient_clip=0.5,
        )
        gradient_monitor.update(state, gradients={"g": torch.tensor([0.1])})
        gradient_monitor.update(state, gradients={"g": torch.tensor([1.0])})

        assert not gradient_monitor.is_stable(), "Gradient spike beyond clip should flag instability"


class TestHOPEBrain:
    """Integration tests for full brain step."""
    
    def test_brain_creation(self):
        """Test brain creation."""
        config = HOPEConfig.development()
        brain = HOPEBrain(
            config=config,
            obs_dim=10,
            action_dim=4,
            output_dim=4,
        )
        
        assert brain.obs_dim == 10
        assert brain.action_dim == 4
        assert brain.output_dim == 4
    
    def test_brain_step(self):
        """Test single brain step."""
        config = HOPEConfig.development()
        brain = HOPEBrain(
            config=config,
            obs_dim=10,
            action_dim=4,
            output_dim=4,
        )
        
        brain.reset()
        
        x_obs = torch.randn(10)
        a_prev = torch.randn(4)
        r_t = 0.5
        
        state_next, y_t, info = brain.step(x_obs, a_prev, r_t)
        
        assert y_t.shape == (4,)
        assert 'query' in info
        assert 'context' in info
        assert 'lyapunov' in info
    
    def test_brain_multi_step(self):
        """Test multiple brain steps."""
        config = HOPEConfig.development()
        brain = HOPEBrain(
            config=config,
            obs_dim=10,
            action_dim=4,
            output_dim=4,
        )
        
        brain.reset()
        
        for _ in range(10):
            x_obs = torch.randn(10)
            a_prev = torch.randn(4)
            r_t = 0.5
            
            state_next, y_t, info = brain.step(x_obs, a_prev, r_t)
            assert y_t.shape == (4,)
        
        # Check stability
        assert brain.stability_monitor.is_stable()
    
    def test_brain_checkpoint(self, tmp_path):
        """Test checkpoint save/load."""
        config = HOPEConfig.development()
        brain = HOPEBrain(
            config=config,
            obs_dim=10,
            action_dim=4,
            output_dim=4,
        )
        
        brain.reset()
        
        # Run a few steps
        for _ in range(5):
            x_obs = torch.randn(10)
            a_prev = torch.randn(4)
            r_t = 0.5
            brain.step(x_obs, a_prev, r_t)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "brain.pt"
        brain.save_checkpoint(str(checkpoint_path))
        
        # Load checkpoint
        brain_loaded = HOPEBrain.load_checkpoint(str(checkpoint_path))
        
        assert brain_loaded.obs_dim == brain.obs_dim
        assert brain_loaded.action_dim == brain.action_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
