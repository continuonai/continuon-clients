"""
Nested Learning

Slow parameter adaptation for HOPE architecture.

Math:
    Θ_t = Θ_{t-1} + η_t * U_ξ(s_t, M_t, r_t)

where:
    - η_t is a per-step learning-rate / budget gate (small, maybe sparse)
    - U_ξ is some update functional (gradient-like, Hebbian-like, etc.)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .state import FastState, CMSMemory, Parameters


class NestedLearning(nn.Module):
    """
    Nested Learning: Slow parameter adaptation.
    
    Implements LoRA-style low-rank adaptation with Hebbian-like updates.
    
    The update is computed from:
        - Current fast state s_t
        - CMS memory statistics
        - Reward signal r_t
    
    Updates are sparse and budget-gated by η_t.
    """
    
    def __init__(
        self,
        d_s: int,
        d_mem: int,  # Aggregate memory dimension
        rank: int = 8,  # Low-rank dimension for LoRA
        eta_init: float = 0.01,
        update_threshold: float = 0.1,  # Only update if signal > threshold
        param_clamp_range: Tuple[float, float] = (-0.5, 0.5), # (min, max)
        weight_decay: float = 0.99,
    ):
        """
        Args:
            d_s: Fast state dimension
            d_mem: Aggregate memory dimension
            rank: Low-rank dimension for parameter updates
            eta_init: Initial learning rate
            update_threshold: Threshold for sparse updates
            param_clamp_range: Hard limits for parameter values
            weight_decay: Multiplicative decay factor per step (e.g. 0.99)
        """
        super().__init__()
        
        self.d_s = d_s
        self.d_mem = d_mem
        self.rank = rank
        self.eta_init = eta_init
        self.update_threshold = update_threshold
        self.param_clamp_min = param_clamp_range[0]
        self.param_clamp_max = param_clamp_range[1]
        self.weight_decay = weight_decay
        
        # Update signal network: computes whether to update
        self.update_signal_net = nn.Sequential(
            nn.Linear(d_s + 1, 64),  # s_t + r_t
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Update direction network: computes low-rank update
        self.update_direction_net = nn.Sequential(
            nn.Linear(d_s + d_mem + 1, 128),
            nn.ReLU(),
            nn.Linear(128, rank),
            nn.Tanh(),  # Bounded updates
        )
    
    def compute_memory_stats(self, cms: CMSMemory) -> torch.Tensor:
        """
        Compute aggregate statistics from CMS memory.
        
        Args:
            cms: CMS memory
        
        Returns:
            mem_stats: Aggregate memory statistics [d_mem]
        """
        # Simple approach: concatenate mean of each level
        level_means = []
        for level in cms.levels:
            level_mean = level.M.mean(dim=0)  # [d_ℓ]
            level_means.append(level_mean)
        
        # Concatenate and project to fixed dimension
        mem_stats = torch.cat(level_means, dim=-1)  # [Σ d_ℓ]
        
        # If dimension doesn't match, project
        if mem_stats.shape[0] != self.d_mem:
            # Simple projection (could be learned)
            if mem_stats.shape[0] > self.d_mem:
                mem_stats = mem_stats[:self.d_mem]
            else:
                # Pad with zeros
                padding = torch.zeros(
                    self.d_mem - mem_stats.shape[0],
                    device=mem_stats.device,
                    dtype=mem_stats.dtype,
                )
                mem_stats = torch.cat([mem_stats, padding], dim=-1)
        
        return mem_stats
    
    def forward(
        self,
        params: Parameters,
        fast: FastState,
        cms: CMSMemory,
        r_t: torch.Tensor,
        brain_module: 'nn.Module' = None,
        force_update: bool = False,
    ) -> Parameters:
        """
        Compute parameter update and apply to brain model.
        
        Args:
            params: Current parameters
            fast: Current fast state
            cms: Current CMS memory
            r_t: Reward signal (scalar)
            brain_module: Reference to brain module for parameter updates
            force_update: If True, bypass signal threshold check
        
        Returns:
            Updated parameters
        """
        # Normalize shapes:
        # - fast.s should be [B, d_s]
        # - r_t should be [B, 1]
        fast_s = fast.s
        if fast_s.dim() == 1:
            fast_s = fast_s.unsqueeze(0)

        if r_t.dim() == 0:
            r_t = r_t.view(1, 1)
        elif r_t.dim() == 1:
            # If provided as [B] or [1], reshape to [B, 1]
            if r_t.numel() == fast_s.size(0):
                r_t = r_t.view(fast_s.size(0), 1)
            else:
                r_t = r_t.view(1, 1)
        elif r_t.dim() == 2:
            # Accept [B, 1] or coerce to [B, 1] by reducing extras
            if r_t.size(-1) != 1:
                r_t = r_t.mean(dim=-1, keepdim=True)
        else:
            # Collapse weird shapes
            r_t = r_t.reshape(-1).mean().view(1, 1)
        
        # 1. Compute update signal (should we update?)
        signal_input = torch.cat([fast_s, r_t], dim=-1)
        update_signal = self.update_signal_net(signal_input).squeeze(-1)  # scalar
        
        # 2. Sparse update: only update if signal > threshold (reduced from 0.1 to 0.01)
        # UNLESS force_update is True (Compaction Mode)
        if not force_update and update_signal.item() < 0.01:
            # No update
            return params
        
        # 3. Compute memory statistics
        mem_stats = self.compute_memory_stats(cms)  # [d_mem]
        
        # Ensure mem_stats has batch dimension matching fast.s
        if mem_stats.dim() == 1:
            batch_size = fast_s.size(0)
            mem_stats = mem_stats.unsqueeze(0).expand(batch_size, -1) # [B, d_mem]
        
        # 4. Compute update direction
        direction_input = torch.cat([fast_s, mem_stats, r_t], dim=-1)
        update_direction = self.update_direction_net(direction_input)  # [B, rank]
        
        # Average over batch if needed
        if update_direction.dim() > 1:
            update_direction = update_direction.mean(dim=0) # [rank]
        
        # 5. Apply update to brain model parameters (if provided)
        # This is a simplified gradient-free update mechanism
        # Scale update by learning rate and signal strength
        if brain_module is not None:
            update_scale = params.eta * update_signal.item()
            
            # Apply small updates to a subset of parameters
            # Target the output decoder and core modules for faster adaptation
            param_count = 0
            for name, param in brain_module.named_parameters():
                if param.requires_grad and ('output_decoder' in name or 'hope_core' in name):
                    if param_count < self.rank:
                        # Apply scaled update (gradient-free Hebbian-like)
                        with torch.no_grad():
                            # Stabilized Update:
                            # 1. Clamp update scale and magnitude
                            update_scale = max(min(update_scale, 0.01), -0.01)
                            update_idx = param_count % self.rank
                            update_magnitude = max(min(update_direction[update_idx].item(), 1.0), -1.0)
                            
                            # 2. Apply update with Decay (Strong Dissipation)
                            noise = torch.randn_like(param) * 0.01
                            delta = update_scale * update_magnitude * noise
                            
                            # Strong Decay to force energy dissipation (Lyapunov stability)
                            param.data.mul_(self.weight_decay) 
                            param.data.add_(delta)
                            
                            # 3. Strict Clamp to very small range to guarantee stability
                            # e.g. [-0.5, 0.5] or configuration based
                            param.data.clamp_(self.param_clamp_min, self.param_clamp_max) 
                        param_count += 1
        
        # 6. Store update metadata in params.theta for tracking
        theta_new = params.theta.copy()
        update_key = f"update_{len(theta_new)}"
        theta_new[update_key] = update_signal * update_direction
        
        return Parameters(theta=theta_new, eta=params.eta)


class AdaptiveLearningRate(nn.Module):
    """
    Adaptive learning rate η_t based on state and performance.
    
    Adjusts η_t based on:
        - Recent reward trends
        - State stability
        - Memory utilization
    """
    
    def __init__(self, eta_min: float = 1e-5, eta_max: float = 0.1):
        """
        Args:
            eta_min: Minimum learning rate
            eta_max: Maximum learning rate
        """
        super().__init__()
        
        self.eta_min = eta_min
        self.eta_max = eta_max
        
        # Simple MLP for eta prediction
        self.eta_net = nn.Sequential(
            nn.Linear(3, 16),  # [reward, state_norm, memory_norm]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(
        self,
        r_t: torch.Tensor,
        fast: FastState,
        cms: CMSMemory,
    ) -> float:
        """
        Compute adaptive learning rate.
        
        Args:
            r_t: Current reward
            fast: Current fast state
            cms: Current CMS memory
        
        Returns:
            eta_t: Adaptive learning rate
        """
        # Compute features
        # Collapse reward to scalar feature
        if r_t.dim() == 0:
            reward_feat = r_t.view(1)
        else:
            reward_feat = r_t.reshape(-1).mean().view(1)
        state_norm = torch.norm(fast.s).unsqueeze(0)
        
        # Memory norm (average across levels)
        memory_norms = [torch.norm(level.M) for level in cms.levels]
        memory_norm = torch.stack(memory_norms).mean().unsqueeze(0)
        
        # Concatenate features
        features = torch.cat([reward_feat, state_norm, memory_norm], dim=-1)
        
        # Predict eta in [eta_min, eta_max]
        eta_normalized = self.eta_net(features).squeeze(-1)
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * eta_normalized
        
        return eta_t.item()
