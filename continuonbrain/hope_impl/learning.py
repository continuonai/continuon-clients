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
from typing import Dict, Optional

from hope_impl.state import FastState, CMSMemory, Parameters


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
    ):
        """
        Args:
            d_s: Fast state dimension
            d_mem: Aggregate memory dimension
            rank: Low-rank dimension for parameter updates
            eta_init: Initial learning rate
            update_threshold: Threshold for sparse updates
        """
        super().__init__()
        
        self.d_s = d_s
        self.d_mem = d_mem
        self.rank = rank
        self.eta_init = eta_init
        self.update_threshold = update_threshold
        
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
    ) -> Parameters:
        """
        Compute parameter update.
        
        Args:
            params: Current parameters
            fast: Current fast state
            cms: Current CMS memory
            r_t: Reward signal (scalar)
        
        Returns:
            Updated parameters
        """
        # Ensure reward is 1D
        if r_t.dim() == 0:
            r_t = r_t.unsqueeze(0)
        
        # 1. Compute update signal (should we update?)
        signal_input = torch.cat([fast.s, r_t], dim=-1)
        update_signal = self.update_signal_net(signal_input).squeeze(-1)  # scalar
        
        # 2. Sparse update: only update if signal > threshold
        if update_signal.item() < self.update_threshold:
            # No update
            return params
        
        # 3. Compute memory statistics
        mem_stats = self.compute_memory_stats(cms)  # [d_mem]
        
        # 4. Compute update direction
        direction_input = torch.cat([fast.s, mem_stats, r_t], dim=-1)
        update_direction = self.update_direction_net(direction_input)  # [rank]
        
        # 5. Apply update to parameters
        # For now, we store the update direction as a named parameter
        # In a full implementation, this would update LoRA matrices
        theta_new = params.theta.copy()
        
        # Store update (simplified: just store the direction)
        update_key = f"update_{len(theta_new)}"
        theta_new[update_key] = params.eta * update_signal * update_direction
        
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
        reward_feat = r_t.unsqueeze(0) if r_t.dim() == 0 else r_t
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
