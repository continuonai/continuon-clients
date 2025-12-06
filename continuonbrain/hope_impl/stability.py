"""
Stability Analysis and Monitoring

Lyapunov-based stability functions for HOPE architecture.

Math:
    V(X) = V_fast(s,w,p) + V_mem(M) + V_params(Θ)

Used for:
    - Training regularization
    - Stability monitoring during inference
    - Diagnostic tool for debugging
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from hope_impl.state import FastState, CMSMemory, Parameters, FullState


def lyapunov_fast_state(fast: FastState, P_s: float = 1.0, Q: float = 1.0, P_p: float = 1.0) -> torch.Tensor:
    """
    Candidate Lyapunov function for fast state.
    
    Math:
        V_fast(s,w,p) = P_s ||s||² + Q ||w||² + P_p ||p||²
    
    Args:
        fast: Fast state (s, w, p)
        P_s: Weight for s
        Q: Weight for w
        P_p: Weight for p
    
    Returns:
        V_fast: Energy scalar
    """
    V_s = P_s * torch.sum(fast.s ** 2)
    V_w = Q * torch.sum(fast.w ** 2)
    V_p = P_p * torch.sum(fast.p ** 2)
    
    return V_s + V_w + V_p


def lyapunov_memory(cms: CMSMemory, lambda_weights: list = None) -> torch.Tensor:
    """
    Memory Lyapunov term.
    
    Math:
        V_mem(M) = Σ_ℓ λ_ℓ ||M^(ℓ)||_F²
    
    Args:
        cms: CMS memory
        lambda_weights: Per-level weights (default: all 1.0)
    
    Returns:
        V_mem: Memory energy scalar
    """
    if lambda_weights is None:
        lambda_weights = [1.0] * cms.num_levels
    
    V_mem = torch.tensor(0.0, device=cms.levels[0].M.device)
    
    for ell, level in enumerate(cms.levels):
        # Frobenius norm squared
        M_norm_sq = torch.sum(level.M ** 2)
        V_mem = V_mem + lambda_weights[ell] * M_norm_sq
    
    return V_mem


def lyapunov_params(params: Parameters, theta_star: Dict[str, torch.Tensor] = None, mu: float = 1.0) -> torch.Tensor:
    """
    Parameter Lyapunov term.
    
    Math:
        V_params(Θ) = μ ||Θ - Θ*||²
    
    where Θ* is some nominal parameter (e.g., initialization).
    
    Args:
        params: Current parameters
        theta_star: Nominal parameters (default: zero)
        mu: Weight
    
    Returns:
        V_params: Parameter distance scalar
    """
    if len(params.theta) == 0:
        return torch.tensor(0.0)
    
    V_params = torch.tensor(0.0)
    
    for key, value in params.theta.items():
        if theta_star is not None and key in theta_star:
            diff = value - theta_star[key]
        else:
            diff = value
        
        V_params = V_params + torch.sum(diff ** 2)
    
    return mu * V_params


def lyapunov_total(
    state: FullState,
    weights: Dict[str, float] = None,
    theta_star: Dict[str, torch.Tensor] = None,
) -> torch.Tensor:
    """
    Total candidate Lyapunov function.
    
    Math:
        V(X) = V_fast + V_mem + V_params
    
    The stability story:
        - During continuous flow, we want dV/dt ≤ -α V + β ||u||²
        - At jumps, we want ΔV to be bounded and preferably small
    
    Args:
        state: Full HOPE state
        weights: Dict with keys 'fast', 'memory', 'params' for component weights
        theta_star: Nominal parameters for params term
    
    Returns:
        V_total: Total energy scalar
    """
    if weights is None:
        weights = {'fast': 1.0, 'memory': 1.0, 'params': 0.1}
    
    V_fast = lyapunov_fast_state(state.fast_state)
    V_mem = lyapunov_memory(state.cms)
    V_par = lyapunov_params(state.params, theta_star=theta_star)
    
    V_total = (
        weights.get('fast', 1.0) * V_fast +
        weights.get('memory', 1.0) * V_mem +
        weights.get('params', 0.1) * V_par
    )
    
    return V_total


class StabilityMonitor:
    """
    Monitor for tracking stability metrics during training/inference.
    
    Tracks:
        - Lyapunov energy over time
        - Energy dissipation rate
        - Gradient norms
        - State norms
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of steps to track
        """
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.lyapunov_history = []
        self.gradient_norms = []
        self.state_norms = []
        self.step_count = 0
    
    def update(
        self,
        state: FullState,
        gradients: Dict[str, torch.Tensor] = None,
    ):
        """
        Update monitoring metrics.
        
        Args:
            state: Current full state
            gradients: Optional gradient dict
        """
        # Compute Lyapunov energy
        V = lyapunov_total(state).item()
        self.lyapunov_history.append(V)
        
        # Compute state norm
        state_norm = torch.norm(state.fast_state.s).item()
        self.state_norms.append(state_norm)
        
        # Compute gradient norm if provided
        if gradients is not None:
            grad_norm = sum(torch.norm(g).item() ** 2 for g in gradients.values()) ** 0.5
            self.gradient_norms.append(grad_norm)
        
        # Keep only recent history
        if len(self.lyapunov_history) > self.window_size:
            self.lyapunov_history = self.lyapunov_history[-self.window_size:]
            self.state_norms = self.state_norms[-self.window_size:]
            if gradients is not None:
                self.gradient_norms = self.gradient_norms[-self.window_size:]
        
        self.step_count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current stability metrics.
        
        Returns:
            Dict with metrics
        """
        metrics = {
            'lyapunov_current': self.lyapunov_history[-1] if self.lyapunov_history else 0.0,
            'lyapunov_mean': sum(self.lyapunov_history) / len(self.lyapunov_history) if self.lyapunov_history else 0.0,
            'state_norm': self.state_norms[-1] if self.state_norms else 0.0,
            'steps': self.step_count,
        }
        
        if self.gradient_norms:
            metrics['gradient_norm'] = self.gradient_norms[-1]
        
        # Compute dissipation rate (negative = energy decreasing = stable)
        if len(self.lyapunov_history) >= 2:
            recent_delta = self.lyapunov_history[-1] - self.lyapunov_history[-2]
            metrics['dissipation_rate'] = -recent_delta  # Positive = dissipating
        
        return metrics
    
    def is_stable(self, threshold: float = 1e6) -> bool:
        """
        Check if system appears stable.
        
        Args:
            threshold: Maximum allowed Lyapunov energy
        
        Returns:
            True if stable
        """
        if not self.lyapunov_history:
            return True
        
        current_V = self.lyapunov_history[-1]
        
        # Check for explosion
        if current_V > threshold:
            return False
        
        # Check for NaN/Inf
        if not torch.isfinite(torch.tensor(current_V)):
            return False
        
        return True
