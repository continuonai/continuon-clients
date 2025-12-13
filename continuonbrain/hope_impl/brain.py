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
import math
import numpy as np
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Union
from pathlib import Path

from .config import HOPEConfig
from .state import FastState, CMSMemory, Parameters, FullState, MemoryLevel
from .encoders import InputEncoder, OutputDecoder
from .cms import CMSRead, CMSWrite
from .core import HOPECore
from .learning import NestedLearning, AdaptiveLearningRate
from .stability import lyapunov_total, StabilityMonitor


class HOPEColumn(nn.Module):
    """
    Single Cortical Column for HOPE architecture.
    """
    def __init__(self, config: HOPEConfig, obs_dim: int, action_dim: int, output_dim: int, obs_type: str, output_type: str, device, dtype):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        
        # 1. Input encoder
        self.encoder = InputEncoder(obs_dim, action_dim, config.d_e, obs_type)
        
        # 2. CMS read
        self.cms_read = CMSRead(config.d_s, config.d_e, config.d_k, config.d_c, config.num_levels, config.cms_dims)
        
        # 3. HOPE core
        self.hope_core = HOPECore(
            config.d_s, config.d_w, config.d_p, config.d_e, config.d_c,
            use_layer_norm=config.use_layer_norm,
            saturation_limit=config.state_saturation_limit
        )
        
        # 4. Output decoder
        self.output_decoder = OutputDecoder(config.d_s, config.d_c, output_dim, output_type)
        
        # 5. CMS write
        self.cms_write = CMSWrite(config.d_s, config.d_e, config.d_c, config.d_k, config.num_levels, config.cms_dims)
        
        # 6. Nested learning
        d_mem = sum(config.cms_dims)
        self.nested_learning = NestedLearning(
            d_s=config.d_s, 
            d_mem=d_mem, 
            eta_init=config.eta_init,
            param_clamp_range=(config.param_clamp_min, config.param_clamp_max),
            weight_decay=config.weight_decay
        )
        
        # 8. Stability monitor (Each column monitors its own stability)
        self.stability_monitor = StabilityMonitor(config.lyapunov_threshold, config.dissipation_floor, config.gradient_clip)
        
        # State
        self._state: Optional[FullState] = None
        
        # History
        self.history_buffer = []
        self.history_window_size = 64
        self.to(device)

    def reset(self) -> FullState:
        self._state = FullState.zeros(
            self.config.d_s, self.config.d_w, self.config.d_p,
            self.config.cms_sizes, self.config.cms_dims, self.config.d_k,
            self.config.cms_decays, self.config.eta_init, self.device, self.dtype
        )
        self.stability_monitor.reset()
        self.history_buffer = []
        return self._state

    def step(self, x_obs, a_prev, r_t, perform_cms_write=True, perform_param_update=False, gradients=None, log_stability=True):
        if self._state is None: self.reset()
        state_prev = self._state
        
        # Ensure tensors
        if not isinstance(x_obs, torch.Tensor): x_obs = torch.tensor(x_obs, device=self.device, dtype=self.dtype)
        if not isinstance(a_prev, torch.Tensor): a_prev = torch.tensor(a_prev, device=self.device, dtype=self.dtype)
        if not isinstance(r_t, torch.Tensor): r_t = torch.tensor(r_t, device=self.device, dtype=self.dtype)
        
        x_obs, a_prev, r_t = x_obs.to(self.device), a_prev.to(self.device), r_t.to(self.device)

        # Helper to ensure state has batch dimension if needed
        s_vals = state_prev.fast_state.s
        w_vals = state_prev.fast_state.w
        p_vals = state_prev.fast_state.p
        
        if x_obs.dim() == 1: # Unbatched vector
             x_obs = x_obs.unsqueeze(0)
        elif x_obs.dim() == 3: # Unbatched image [C,H,W]
             x_obs = x_obs.unsqueeze(0)
            
        if s_vals.dim() == 1:
            s_vals = s_vals.unsqueeze(0)
            w_vals = w_vals.unsqueeze(0)
            p_vals = p_vals.unsqueeze(0)
            
        if a_prev.dim() == 1:
            a_prev = a_prev.unsqueeze(0)
            
        if r_t.dim() < 2:
            r_t = r_t.reshape(-1, 1)

        # 1. Encode
        e_t = self.encoder(x_obs, a_prev, r_t)
        
        # 2. CMS Read
        q_t, c_t, attn = self.cms_read(state_prev.cms, s_vals, e_t)
        
        # 3. Core Dynamics
        s_next_vals, w_next_vals, p_next_vals = self.hope_core(s_vals, w_vals, p_vals, e_t, c_t)
        fast_next = FastState(s_next_vals, w_next_vals, p_next_vals)
        
        # 4. Decode
        y_t = self.output_decoder(s_next_vals, c_t)
        
        # 5. Write
        cms_next = state_prev.cms
        if perform_cms_write:
            self.history_buffer.append((state_prev.fast_state.s.detach(), e_t.detach()))
            if len(self.history_buffer) > self.history_window_size: self.history_buffer.pop(0)
            cms_next = self.cms_write(state_prev.cms, fast_next.s, e_t, level_contexts=None, history_buffer=self.history_buffer)
            
        # 6. Parameter Update (Nested Learning)
        if perform_param_update:
            # DEBUG SHAPES
            # print(f"DEBUG_LEARN: fast.s={fast_next.s.shape}, r_t={r_t.shape}", file=sys.stderr)
            params_next = self.nested_learning(state_prev.params, fast_next, cms_next, r_t, brain_module=self)
        else:
            params_next = state_prev.params
            
        state_next = FullState(fast_next, cms_next, params_next)
        self._state = state_next
        
        if log_stability:
            self.stability_monitor.update(state_next, gradients=gradients)
            
        metrics = self.stability_monitor.get_metrics()
        lyapunov = lyapunov_total(state_next).item()
        
        info = {
            'query': q_t, 'context': c_t, 'attention_weights': attn,
            'stability_metrics': metrics, 'lyapunov': lyapunov
        }
        return state_next, y_t, info
        return state_next, y_t, info

    def compact_memory(self):
        """
        Execute Memory Compaction (Consolidation Cycle).
        
        Triggers:
        1. Force Nested Learning update (Transfer M -> Theta)
        2. Flush CMS Memory (High Decay)
        """
        # 1. Use Compaction Hyperparameters
        original_lr = self._state.params.eta
        original_decay = self._state.cms.levels[0].decay # Assuming homogeneous decay for now
        
        # Set aggressive values
        self._state.params.eta = self.config.compaction_learning_rate
        flush_decay = self.config.compaction_decay
        
        # 2. Force Learning Update (Consolidation)
        # We use the current state as the "summary" to be learned
        # In a more advanced version, we would replay a sequence.
        # Here, we do a "One-Shot Consolidation" of the current memory stats.
        
        # Note: effectively we are saying "The current aggregate statistical state of memory
        # should be baked into the weights"
        
        # Pass force_update=True to bypass threshold
        # Ensure r_t has correct shape [1, 1] to match fast.s [1, d_s]
        r_consolidation = torch.tensor([[1.0]], device=self.device)
        
        self._state.params = self.nested_learning(
            self._state.params, 
            self._state.fast_state, 
            self._state.cms, 
            r_consolidation, # Positive reinforcement for consolidation
            brain_module=self.hope_core,
            force_update=True
        )
        
        # 3. Flush Memory (High Decay)
        # We manually decay the CMS levels
        new_levels = []
        for level in self._state.cms.levels:
            # Apply strong decay factor
            # M_new = (1 - flush) * M_old
            M_flushed = (1 - flush_decay) * level.M
            K_flushed = (1 - flush_decay) * level.K
            new_levels.append(MemoryLevel(M_flushed, K_flushed, level.decay))
            
        self._state.cms = CMSMemory(new_levels)
        
        # 4. Restore Hyperparameters
        self._state.params.eta = original_lr
        
        # 5. Calculate compaction metrics
        cms_energy_before = sum(level.M.norm().item() for level in self._state.cms.levels)
        # Note: cms_energy_after would be after flush, but we already flushed above
        # Calculate approximate energy reduction
        energy_reduction = flush_decay * cms_energy_before
        
        return {
            "status": "compacted", 
            "energy_transfer": "cms->theta",
            "cms_energy_reduction": float(energy_reduction),
            "flush_decay": float(flush_decay),
            "learning_rate_used": float(self.config.compaction_learning_rate)
        }

class HOPEBrain(nn.Module):
    """
    HOPE Brain: Main interface. Supports Hybrid (Thousand Brains) mode.
    """
    def __init__(self, config: HOPEConfig, obs_dim: int, action_dim: int, output_dim: int, obs_type: str = "vector", output_type: str = "continuous"):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.obs_type = obs_type
        self.output_type = output_type

        # Hybrid Mode: Create N columns
        self.columns = nn.ModuleList()
        num_cols = config.num_columns if config.use_hybrid_mode else 1
        
        for _ in range(num_cols):
            col = HOPEColumn(config, obs_dim, action_dim, output_dim, obs_type, output_type, self.device, self.dtype)
            self.columns.append(col)
            
        self.to(self.device)
        self.active_column_idx = 0 # Trace which column won the vote

    def initialize(self, num_columns: int = None):
        """
        Re-initialize the brain, optionally changing topology (Hybrid vs Standard).
        
        Args:
            num_columns: Number of columns to use. 1 = Standard, >1 = Hybrid.
        """
        if num_columns is not None:
            self.config.num_columns = num_columns
            self.config.use_hybrid_mode = (num_columns > 1)
            
        # Re-create columns
        self.columns = nn.ModuleList()
        # Ensure we use the potentially updated config value
        num_cols = self.config.num_columns if self.config.use_hybrid_mode else 1
        
        for _ in range(num_cols):
            col = HOPEColumn(
                self.config, 
                self.obs_dim, 
                self.action_dim, 
                self.output_dim, 
                self.obs_type, 
                self.output_type, 
                self.device, 
                self.dtype
            )
            self.columns.append(col)
            
        self.to(self.device)
        self.active_column_idx = 0
        print(f"HOPEBrain initialized with {num_cols} columns (Hybrid={self.config.use_hybrid_mode})")

    def reset(self) -> FullState:
        # Reset all columns
        states = [col.reset() for col in self.columns]
        # Return state of "active" column (usually 0 at start)
        self.active_column_idx = 0
        return states[0]

    def step(self, x_obs, a_prev, r_t, perform_cms_write=True, perform_param_update=False, gradients=None, log_stability=True):
        """
        Execute step. If Hybrid, run all columns and Vote.
        """
        results = []
        
        # Parallel Execution (Sequential loop for now, conceptually parallel)
        for i, col in enumerate(self.columns):
            # Future improvement: Slice input x_obs for different columns here
            s_next, y, info = col.step(x_obs, a_prev, r_t, perform_cms_write, perform_param_update, gradients, log_stability)
            results.append((s_next, y, info))
            
        # Voting Mechanism
        if len(self.columns) > 1:
            # Strategies:
            # 1. Lowest Lyapunov Energy (Most Stable) - PREFERRED based on theory (stability = prediction match)
            # 2. Highest Reward Prediction (if we had a critic)
            
            best_idx = 0
            min_energy = float('inf')
            
            votes = []
            for i, (s, y, info) in enumerate(results):
                energy = info['lyapunov']
                votes.append(f"Col{i}:{energy:.2f}")
                
                # Check for NaN exploders
                if math.isnan(energy) or math.isinf(energy):
                    energy = float('inf')
                    
                if energy < min_energy:
                    min_energy = energy
                    best_idx = i
            
            # Update winner
            self.active_column_idx = best_idx
            state_next, y_t, info = results[best_idx]
            
            # Inject voting info
            info['hybrid_votes'] = votes
            info['winner_column'] = best_idx
            
        else:
            state_next, y_t, info = results[0]
            self.active_column_idx = 0
            
        return state_next, y_t, info

    def forward(self, x_obs, a_prev, r_t):
        """
        Forward pass (inference only, no state update).
        """
        # Only use active column for simple forward
        return self.columns[self.active_column_idx].forward(x_obs, a_prev, r_t)

    def compact_memory(self):
        """
        Trigger memory compaction on all columns.
        """
        results = []
        for i, col in enumerate(self.columns):
            res = col.compact_memory()
            results.append(f"Col{i}:{res['status']}")
        return results


        
    def get_state(self) -> FullState:
        """Get current internal state of the active column."""
        return self.columns[self.active_column_idx]._state

    def set_state(self, state: FullState):
        """Set internal state for the active column."""
        self.columns[self.active_column_idx]._state = state.to(self.device)
        
    # Proxy methods for backward compatibility
    @property
    def stability_monitor(self):
        return self.columns[self.active_column_idx].stability_monitor
        
    def save_checkpoint(self, path: str):
        """
        Save brain checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        # Save all columns
        checkpoint = {
            'config': self.config,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'output_dim': self.output_dim,
            'obs_type': self.obs_type,
            'output_type': self.output_type,
            'columns_state_dicts': [c.state_dict() for c in self.columns],
            'columns_internal_states': [c._state for c in self.columns],
            'is_hybrid': self.config.use_hybrid_mode
        }
        torch.save(checkpoint, path)
        print(f"Hybrid Checkpoint saved to {path} ({len(self.columns)} cols)")

    @classmethod
    def load_checkpoint(cls, path: str) -> "HOPEBrain":
        """
        Load brain from checkpoint.
        
        Args:
            path: Path to checkpoint
        
        Returns:
            Loaded HOPEBrain
        """
        # Load logic needs to handle single vs hybrid
        checkpoint = torch.load(path, weights_only=False)
        brain = cls(
            checkpoint['config'], checkpoint['obs_dim'], checkpoint['action_dim'],
            checkpoint['output_dim'], checkpoint['obs_type'], checkpoint['output_type']
        )
        
        if 'columns_state_dicts' in checkpoint:
            # New format
            for i, col in enumerate(brain.columns):
                if i < len(checkpoint['columns_state_dicts']):
                    col.load_state_dict(checkpoint['columns_state_dicts'][i])
                    if checkpoint['columns_internal_states'][i] is not None:
                        col._state = checkpoint['columns_internal_states'][i]
        else:
            # Old format - load into first column
            # Ensure there's at least one column for the old format
            if not brain.columns:
                brain.columns.append(HOPEColumn(brain.config, brain.obs_dim, brain.action_dim, brain.output_dim, brain.obs_type, brain.output_type, brain.device, brain.dtype))
            
            brain.columns[0].load_state_dict(checkpoint['model_state_dict'])
            if checkpoint['internal_state'] is not None:
                brain.columns[0]._state = checkpoint['internal_state']
                
            # Restore stability monitor for the first column if it exists in old checkpoint
            if 'stability_monitor' in checkpoint:
                brain.columns[0].stability_monitor.lyapunov_history = checkpoint['stability_monitor']['lyapunov_history']
                brain.columns[0].stability_monitor.state_norms = checkpoint['stability_monitor']['state_norms']
                brain.columns[0].stability_monitor.gradient_norms = checkpoint['stability_monitor'].get('gradient_norms', [])
                brain.columns[0].stability_monitor.step_count = checkpoint['stability_monitor']['step_count']
                
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
        # Create a new brain instance to hold quantized columns
        quantized_brain = HOPEBrain(
            config=self.config,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            output_dim=self.output_dim,
            obs_type=self.obs_type,
            output_type=self.output_type,
        )
        
        # Quantize each column
        quantized_columns = nn.ModuleList()
        for col in self.columns:
            if dtype == "int8":
                quantized_col = torch.quantization.quantize_dynamic(
                    col,
                    {nn.Linear},
                    dtype=torch.qint8,
                )
            elif dtype == "fp16":
                quantized_col = col.half()
            else:
                raise ValueError(f"Unknown dtype: {dtype}")
            quantized_columns.append(quantized_col)
            
        quantized_brain.columns = quantized_columns
        quantized_brain.active_column_idx = self.active_column_idx
        
        return quantized_brain
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict with memory usage in MB
        """
        total_usage = {}
        for i, col in enumerate(self.columns):
            # Calculate per column
             def tensor_size_mb(tensor): return tensor.element_size() * tensor.nelement() / (1024 ** 2)
             
             col_usage = {
                'model_parameters': sum(tensor_size_mb(p) for p in col.parameters()),
             }
             
             if col._state is not None:
                col_usage['fast_state'] = (
                    tensor_size_mb(col._state.fast_state.s) +
                    tensor_size_mb(col._state.fast_state.w) +
                    tensor_size_mb(col._state.fast_state.p)
                )
                
                cms_size = 0
                for level in col._state.cms.levels:
                    cms_size += tensor_size_mb(level.M) + tensor_size_mb(level.K)
                col_usage['cms_memory'] = cms_size
                
                params_size = sum(tensor_size_mb(v) for v in col._state.params.theta.values())
                col_usage['adaptive_params'] = params_size
                
                col_usage['total_state'] = col_usage['fast_state'] + col_usage['cms_memory'] + col_usage['adaptive_params']
             
             col_usage['total'] = col_usage['model_parameters'] + col_usage.get('total_state', 0)
             total_usage[f'column_{i}'] = col_usage
        
        # Aggregate totals
        if self.columns:
            total_usage['overall_model_parameters'] = sum(total_usage[f'column_{i}']['model_parameters'] for i in range(len(self.columns)))
            total_usage['overall_total_state'] = sum(total_usage[f'column_{i}'].get('total_state', 0) for i in range(len(self.columns)))
            total_usage['overall_total'] = sum(total_usage[f'column_{i}']['total'] for i in range(len(self.columns)))
        
        return total_usage
