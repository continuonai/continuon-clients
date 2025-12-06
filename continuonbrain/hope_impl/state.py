"""
HOPE State Objects

PyTorch-based state representations for the HOPE architecture.
All state objects support device movement, serialization, and checkpointing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn


@dataclass
class FastState:
    """
    Fast latent state of the HOPE core.
    
    Components:
        s: Unified fast state (mixed wave + particle) [d_s]
        w: Wave / SSM-like global state [d_w]
        p: Particle / local nonlinear state [d_p]
    
    Math:
        s(t) ∈ ℝ^{d_s}
        w(t) ∈ ℝ^{d_w}
        p(t) ∈ ℝ^{d_p}
    """
    s: torch.Tensor  # [d_s]
    w: torch.Tensor  # [d_w]
    p: torch.Tensor  # [d_p]
    
    def to(self, device: torch.device) -> FastState:
        """Move state to device."""
        return FastState(
            s=self.s.to(device),
            w=self.w.to(device),
            p=self.p.to(device),
        )
    
    def detach(self) -> FastState:
        """Detach from computation graph."""
        return FastState(
            s=self.s.detach(),
            w=self.w.detach(),
            p=self.p.detach(),
        )
    
    def clone(self) -> FastState:
        """Create a copy."""
        return FastState(
            s=self.s.clone(),
            w=self.w.clone(),
            p=self.p.clone(),
        )
    
    @staticmethod
    def zeros(d_s: int, d_w: int, d_p: int, device: torch.device = None, dtype: torch.dtype = None) -> FastState:
        """Create zero-initialized fast state."""
        return FastState(
            s=torch.zeros(d_s, device=device, dtype=dtype),
            w=torch.zeros(d_w, device=device, dtype=dtype),
            p=torch.zeros(d_p, device=device, dtype=dtype),
        )
    
    @staticmethod
    def randn(d_s: int, d_w: int, d_p: int, device: torch.device = None, dtype: torch.dtype = None) -> FastState:
        """Create random-initialized fast state."""
        return FastState(
            s=torch.randn(d_s, device=device, dtype=dtype),
            w=torch.randn(d_w, device=device, dtype=dtype),
            p=torch.randn(d_p, device=device, dtype=dtype),
        )


@dataclass
class MemoryLevel:
    """
    One level of the CMS (Continuous Memory System).
    
    Components:
        M: Memory matrix [N_ℓ, d_ℓ]
        K: Key matrix for content addressing [N_ℓ, d_k]
        decay: Decay coefficient d_ℓ ∈ (0,1)
    
    Math:
        M_t^{(ℓ)} ∈ ℝ^{N_ℓ × d_ℓ}
        K_t^{(ℓ)} ∈ ℝ^{N_ℓ × d_k}
        d_ℓ : decay coefficient per level
    """
    M: torch.Tensor  # [N_ℓ, d_ℓ]
    K: torch.Tensor  # [N_ℓ, d_k]
    decay: float
    
    def to(self, device: torch.device) -> MemoryLevel:
        """Move memory level to device."""
        return MemoryLevel(
            M=self.M.to(device),
            K=self.K.to(device),
            decay=self.decay,
        )
    
    def detach(self) -> MemoryLevel:
        """Detach from computation graph."""
        return MemoryLevel(
            M=self.M.detach(),
            K=self.K.detach(),
            decay=self.decay,
        )
    
    def clone(self) -> MemoryLevel:
        """Create a copy."""
        return MemoryLevel(
            M=self.M.clone(),
            K=self.K.clone(),
            decay=self.decay,
        )
    
    @staticmethod
    def zeros(N: int, d: int, d_k: int, decay: float, device: torch.device = None, dtype: torch.dtype = None) -> MemoryLevel:
        """Create zero-initialized memory level."""
        return MemoryLevel(
            M=torch.zeros(N, d, device=device, dtype=dtype),
            K=torch.zeros(N, d_k, device=device, dtype=dtype),
            decay=decay,
        )
    
    @staticmethod
    def randn(N: int, d: int, d_k: int, decay: float, device: torch.device = None, dtype: torch.dtype = None) -> MemoryLevel:
        """Create random-initialized memory level."""
        return MemoryLevel(
            M=torch.randn(N, d, device=device, dtype=dtype) * 0.01,  # Small init
            K=torch.randn(N, d_k, device=device, dtype=dtype) * 0.01,
            decay=decay,
        )


@dataclass
class CMSMemory:
    """
    Hierarchical CMS (Continuous Memory System).
    
    Components:
        levels: Ordered list of MemoryLevel, from fastest (episodic) to slowest (semantic)
    
    Hierarchy:
        Level 0: Episodic (fast decay, small capacity)
        Level 1: Working memory (medium decay, medium capacity)
        Level L: Semantic (slow decay, large capacity)
    """
    levels: List[MemoryLevel]
    
    def to(self, device: torch.device) -> CMSMemory:
        """Move all levels to device."""
        return CMSMemory(levels=[lvl.to(device) for lvl in self.levels])
    
    def detach(self) -> CMSMemory:
        """Detach all levels from computation graph."""
        return CMSMemory(levels=[lvl.detach() for lvl in self.levels])
    
    def clone(self) -> CMSMemory:
        """Create a copy."""
        return CMSMemory(levels=[lvl.clone() for lvl in self.levels])
    
    @property
    def num_levels(self) -> int:
        """Number of hierarchy levels."""
        return len(self.levels)
    
    @staticmethod
    def zeros(sizes: List[int], dims: List[int], d_k: int, decays: List[float], 
              device: torch.device = None, dtype: torch.dtype = None) -> CMSMemory:
        """
        Create zero-initialized CMS memory.
        
        Args:
            sizes: [N_0, N_1, ..., N_L] - slots per level
            dims: [d_0, d_1, ..., d_L] - dimensions per level
            d_k: Key dimension (shared across levels)
            decays: [d_0, d_1, ..., d_L] - decay coefficients
        """
        assert len(sizes) == len(dims) == len(decays), "Mismatched level specifications"
        
        levels = [
            MemoryLevel.zeros(N, d, d_k, decay, device=device, dtype=dtype)
            for N, d, decay in zip(sizes, dims, decays)
        ]
        return CMSMemory(levels=levels)
    
    @staticmethod
    def randn(sizes: List[int], dims: List[int], d_k: int, decays: List[float],
              device: torch.device = None, dtype: torch.dtype = None) -> CMSMemory:
        """Create random-initialized CMS memory."""
        assert len(sizes) == len(dims) == len(decays), "Mismatched level specifications"
        
        levels = [
            MemoryLevel.randn(N, d, d_k, decay, device=device, dtype=dtype)
            for N, d, decay in zip(sizes, dims, decays)
        ]
        return CMSMemory(levels=levels)


@dataclass
class Parameters:
    """
    Adaptable parameter block for nested learning.
    
    Components:
        theta: Local adapters (LoRA-like, low-rank modules, etc.)
               Stored as dict of named tensors
        eta: Learning-rate / update-budget gate (scalar)
    
    Math:
        Θ_t = Θ_{t-1} + η_t * U_ξ(s_t, M_t, r_t)
    """
    theta: Dict[str, torch.Tensor]
    eta: float
    
    def to(self, device: torch.device) -> Parameters:
        """Move parameters to device."""
        return Parameters(
            theta={k: v.to(device) for k, v in self.theta.items()},
            eta=self.eta,
        )
    
    def detach(self) -> Parameters:
        """Detach from computation graph."""
        return Parameters(
            theta={k: v.detach() for k, v in self.theta.items()},
            eta=self.eta,
        )
    
    def clone(self) -> Parameters:
        """Create a copy."""
        return Parameters(
            theta={k: v.clone() for k, v in self.theta.items()},
            eta=self.eta,
        )
    
    @staticmethod
    def empty(eta: float = 0.01) -> Parameters:
        """Create empty parameter set."""
        return Parameters(theta={}, eta=eta)


@dataclass
class FullState:
    """
    Full internal state of the HOPE brain.
    
    Components:
        fast_state: (s, w, p) - Fast latent dynamics
        cms: Hierarchical memory M_t^{(ℓ)}
        params: Adaptable local parameters Θ_t
    """
    fast_state: FastState
    cms: CMSMemory
    params: Parameters
    
    def to(self, device: torch.device) -> FullState:
        """Move entire state to device."""
        return FullState(
            fast_state=self.fast_state.to(device),
            cms=self.cms.to(device),
            params=self.params.to(device),
        )
    
    def detach(self) -> FullState:
        """Detach entire state from computation graph."""
        return FullState(
            fast_state=self.fast_state.detach(),
            cms=self.cms.detach(),
            params=self.params.detach(),
        )
    
    def clone(self) -> FullState:
        """Create a copy of entire state."""
        return FullState(
            fast_state=self.fast_state.clone(),
            cms=self.cms.clone(),
            params=self.params.clone(),
        )
    
    @staticmethod
    def zeros(d_s: int, d_w: int, d_p: int, cms_sizes: List[int], cms_dims: List[int],
              d_k: int, cms_decays: List[float], eta: float = 0.01,
              device: torch.device = None, dtype: torch.dtype = None) -> FullState:
        """Create zero-initialized full state."""
        return FullState(
            fast_state=FastState.zeros(d_s, d_w, d_p, device=device, dtype=dtype),
            cms=CMSMemory.zeros(cms_sizes, cms_dims, d_k, cms_decays, device=device, dtype=dtype),
            params=Parameters.empty(eta=eta),
        )
    
    @staticmethod
    def randn(d_s: int, d_w: int, d_p: int, cms_sizes: List[int], cms_dims: List[int],
              d_k: int, cms_decays: List[float], eta: float = 0.01,
              device: torch.device = None, dtype: torch.dtype = None) -> FullState:
        """Create random-initialized full state."""
        return FullState(
            fast_state=FastState.randn(d_s, d_w, d_p, device=device, dtype=dtype),
            cms=CMSMemory.randn(cms_sizes, cms_dims, d_k, cms_decays, device=device, dtype=dtype),
            params=Parameters.empty(eta=eta),
        )
