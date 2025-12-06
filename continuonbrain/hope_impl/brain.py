"""
HOPE Brain

Main interface for the HOPE (Hierarchical Online Predictive Encoding) architecture.

Integrates all components:
    - Input encoding
    - CMS read/write
    - HOPE core dynamics
    - Nested learning
    - Output decoding
    - Stability monitoring
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Any
from pathlib import Path

from hope_impl.config import HOPEConfig
from hope_impl.state import FastState, CMSMemory, Parameters, FullState
from hope_impl.encoders import InputEncoder, OutputDecoder
from hope_impl.cms import CMSRead, CMSWrite
from hope_impl.core import HOPECore
from hope_impl.learning import NestedLearning, AdaptiveLearningRate
from hope_impl.stability import lyapunov_total, StabilityMonitor


class HOPEBrain(nn.Module):
    """
    HOPE Brain: Main interface for the HOPE architecture.
    
    One full HOPE step:
        1. Encode inputs: e_t = E_φ(x_obs, a_{t-1}, r_t)
        2. CMS read: q_t, c_t = Read(M_{t-1}, s_{t-1}, e_t)
        3. HOPE core: fast_t = HOPE_core(fast_{t-1}, e_t, c_t, Θ_{t-1})
        4. Output: y_t = H_ω(fast_t, c_t)
        5. CMS write: M_t = Write(M_{t-1}, fast_t, e_t, r_t)
        6. Nested learning: Θ_t = Update(Θ_{t-1}, fast_t, M_t, r_t)
    
    Returns:
        (next_state, output, info)
    """
    
    def __init__(
        self,
        config: HOPEConfig,
        obs_dim: int,
        action_dim: int,
        output_dim: int,
        obs_type: str = "vector",
        output_type: str = "continuous",
    ):
        """
        Args:
            config: HOPE configuration
            obs_dim: Observation dimension
            action_dim: Action dimension
            output_dim: Output dimension
            obs_type: "vector" or "image"
            output_type: "continuous" or "discrete"
        """
        super().__init__()
        
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.obs_type = obs_type
        self.output_type = output_type
        
        # Get device and dtype
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        
        # 1. Input encoder
        self.encoder = InputEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_e=config.d_e,
            obs_type=obs_type,
        )
        
        # 2. CMS read
        self.cms_read = CMSRead(
            d_s=config.d_s,
            d_e=config.d_e,
            d_k=config.d_k,
            d_c=config.d_c,
            num_levels=config.num_levels,
            cms_dims=config.cms_dims,
        )
        
        # 3. HOPE core
        self.hope_core = HOPECore(
            d_s=config.d_s,
            d_w=config.d_w,
            d_p=config.d_p,
            d_e=config.d_e,
            d_c=config.d_c,
            use_layer_norm=config.use_layer_norm,
        )
        
        # 4. Output decoder
        self.output_decoder = OutputDecoder(
            d_s=config.d_s,
            d_c=config.d_c,
            output_dim=output_dim,
            output_type=output_type,
        )
        
        # 5. CMS write
        self.cms_write = CMSWrite(
            d_s=config.d_s,
            d_e=config.d_e,
            d_c=config.d_c,
            d_k=config.d_k,
            num_levels=config.num_levels,
            cms_dims=config.cms_dims,
        )
        
        # 6. Nested learning
        d_mem = sum(config.cms_dims)  # Aggregate memory dimension
        self.nested_learning = NestedLearning(
            d_s=config.d_s,
            d_mem=d_mem,
            eta_init=config.eta_init,
        )
        
        # 7. Adaptive learning rate (optional)
        self.adaptive_lr = AdaptiveLearningRate()
        
        # 8. Stability monitor
        self.stability_monitor = StabilityMonitor()
        
        # Internal state
        self._state: Optional[FullState] = None
        
        # Move to device
        self.to(self.device)
    
    def reset(self) -> FullState:
        """
        Initialize/reset brain state.
        
        Returns:
            Initial state
        """
        self._state = FullState.zeros(
            d_s=self.config.d_s,
            d_w=self.config.d_w,
            d_p=self.config.d_p,
            cms_sizes=self.config.cms_sizes,
            cms_dims=self.config.cms_dims,
            d_k=self.config.d_k,
            cms_decays=self.config.cms_decays,
            eta=self.config.eta_init,
            device=self.device,
            dtype=self.dtype,
        )
        
        self.stability_monitor.reset()
        
        return self._state
    
    def step(
        self,
        x_obs: torch.Tensor,
        a_prev: torch.Tensor,
        r_t: float,
        perform_cms_write: bool = True,
        perform_param_update: bool = False,
    ) -> Tuple[FullState, torch.Tensor, Dict[str, Any]]:
        """
        One full HOPE step.
        
        Args:
            x_obs: Observation
            a_prev: Previous action
            r_t: Reward (scalar)
            perform_cms_write: Whether to write to CMS
            perform_param_update: Whether to update parameters
        
        Returns:
            (next_state, output, info)
        """
        # Initialize state if needed
        if self._state is None:
            self.reset()
        
        state_prev = self._state
        
        # Convert inputs to tensors
        if not isinstance(x_obs, torch.Tensor):
            x_obs = torch.tensor(x_obs, device=self.device, dtype=self.dtype)
        if not isinstance(a_prev, torch.Tensor):
            a_prev = torch.tensor(a_prev, device=self.device, dtype=self.dtype)
        if not isinstance(r_t, torch.Tensor):
            r_t = torch.tensor(r_t, device=self.device, dtype=self.dtype)
        
        # Ensure correct device
        x_obs = x_obs.to(self.device)
        a_prev = a_prev.to(self.device)
        r_t = r_t.to(self.device)
        
        # 1. Encode inputs: e_t = E_φ(x_obs, a_{t-1}, r_t)
        e_t = self.encoder(x_obs, a_prev, r_t)
        
        # 2. CMS read: q_t, c_t = Read(M_{t-1}, s_{t-1}, e_t)
        q_t, c_t, attention_weights = self.cms_read(
            state_prev.cms,
            state_prev.fast_state.s,
            e_t,
        )
        
        # 3. HOPE core: fast_t = HOPE_core(fast_{t-1}, e_t, c_t, Θ_{t-1})
        fast_next = self.hope_core(state_prev.fast_state, e_t, c_t)
        
        # 4. Output: y_t = H_ω(fast_t, c_t)
        y_t = self.output_decoder(fast_next.s, c_t)
        
        # 5. CMS write: M_t = Write(M_{t-1}, fast_t, e_t, r_t)
        cms_next = state_prev.cms
        if perform_cms_write:
            # Get level contexts for hierarchical write
            # (simplified: use None, write module will handle it)
            cms_next = self.cms_write(state_prev.cms, fast_next.s, e_t, level_contexts=None)
        
        # 6. Nested learning: Θ_t = Update(Θ_{t-1}, fast_t, M_t, r_t)
        params_next = state_prev.params
        if perform_param_update:
            params_next = self.nested_learning(state_prev.params, fast_next, cms_next, r_t)
            
            # Optionally update learning rate
            # eta_new = self.adaptive_lr(r_t, fast_next, cms_next)
            # params_next.eta = eta_new
        
        # Create next state
        state_next = FullState(
            fast_state=fast_next,
            cms=cms_next,
            params=params_next,
        )
        
        # Update internal state
        self._state = state_next
        
        # Update stability monitor
        self.stability_monitor.update(state_next)
        
        # Gather info
        info = {
            'query': q_t,
            'context': c_t,
            'attention_weights': attention_weights,
            'stability_metrics': self.stability_monitor.get_metrics(),
            'lyapunov': lyapunov_total(state_next).item(),
        }
        
        return state_next, y_t, info
    
    def forward(
        self,
        x_obs: torch.Tensor,
        a_prev: torch.Tensor,
        r_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training (simplified interface).
        
        Args:
            x_obs: Observation
            a_prev: Previous action
            r_t: Reward
        
        Returns:
            y_t: Output
        """
        _, y_t, _ = self.step(x_obs, a_prev, r_t.item() if r_t.dim() == 0 else r_t[0].item())
        return y_t
    
    def get_state(self) -> FullState:
        """Get current internal state."""
        if self._state is None:
            self.reset()
        return self._state
    
    def set_state(self, state: FullState):
        """Set internal state."""
        self._state = state.to(self.device)
    
    def save_checkpoint(self, path: str):
        """
        Save brain checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'config': self.config,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'output_dim': self.output_dim,
            'obs_type': self.obs_type,
            'output_type': self.output_type,
            'model_state_dict': self.state_dict(),
            'internal_state': self._state,
            'stability_monitor': {
                'lyapunov_history': self.stability_monitor.lyapunov_history,
                'state_norms': self.stability_monitor.state_norms,
                'step_count': self.stability_monitor.step_count,
            },
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str) -> "HOPEBrain":
        """
        Load brain from checkpoint.
        
        Args:
            path: Path to checkpoint
        
        Returns:
            Loaded HOPEBrain
        """
        checkpoint = torch.load(path, weights_only=False)
        
        # Create brain
        brain = cls(
            config=checkpoint['config'],
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            output_dim=checkpoint['output_dim'],
            obs_type=checkpoint['obs_type'],
            output_type=checkpoint['output_type'],
        )
        
        # Load model weights
        brain.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore internal state
        if checkpoint['internal_state'] is not None:
            brain.set_state(checkpoint['internal_state'])
        
        # Restore stability monitor
        if 'stability_monitor' in checkpoint:
            brain.stability_monitor.lyapunov_history = checkpoint['stability_monitor']['lyapunov_history']
            brain.stability_monitor.state_norms = checkpoint['stability_monitor']['state_norms']
            brain.stability_monitor.step_count = checkpoint['stability_monitor']['step_count']
        
        print(f"Checkpoint loaded from {path}")
        return brain
    
    def to_quantized(self, dtype: str = "int8") -> "HOPEBrain":
        """
        Convert to quantized version for deployment.
        
        Args:
            dtype: "int8" or "fp16"
        
        Returns:
            Quantized brain (new instance)
        """
        if dtype == "int8":
            # Dynamic quantization for linear layers
            quantized = torch.quantization.quantize_dynamic(
                self,
                {nn.Linear},
                dtype=torch.qint8,
            )
        elif dtype == "fp16":
            # Half precision
            quantized = self.half()
        else:
            raise ValueError(f"Unknown dtype: {dtype}")
        
        return quantized
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict with memory usage in MB
        """
        def tensor_size_mb(tensor):
            return tensor.element_size() * tensor.nelement() / (1024 ** 2)
        
        usage = {
            'model_parameters': sum(tensor_size_mb(p) for p in self.parameters()),
        }
        
        if self._state is not None:
            # Fast state
            usage['fast_state'] = (
                tensor_size_mb(self._state.fast_state.s) +
                tensor_size_mb(self._state.fast_state.w) +
                tensor_size_mb(self._state.fast_state.p)
            )
            
            # CMS memory
            cms_size = 0
            for level in self._state.cms.levels:
                cms_size += tensor_size_mb(level.M) + tensor_size_mb(level.K)
            usage['cms_memory'] = cms_size
            
            # Parameters
            params_size = sum(tensor_size_mb(v) for v in self._state.params.theta.values())
            usage['adaptive_params'] = params_size
            
            usage['total_state'] = usage['fast_state'] + usage['cms_memory'] + usage['adaptive_params']
        
        usage['total'] = usage['model_parameters'] + usage.get('total_state', 0)
        
        return usage
