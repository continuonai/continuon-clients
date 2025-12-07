"""
HOPE Core Dynamics

Wave-particle hybrid recurrence implementing the core HOPE dynamics.

Math:
    Wave:     w_t = A(c_t, Θ_t) w_{t-1} + B(c_t, Θ_t) z_t
    Particle: p_t = p_{t-1} + φ_Θ(p_{t-1}, z_t, c_t)
    Gate:     g_t = σ(W_g [s_{t-1} || e_t || c_t])
    Mixed:    s_t = s_{t-1} + g_t ⊙ U_p p_t + (1-g_t) ⊙ U_w w_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .state import FastState


class WaveSubsystem(nn.Module):
    """
    Wave subsystem: SSM-like global linear dynamics.
    
    Math:
        w_t = A(c_t, Θ_t) w_{t-1} + B(c_t, Θ_t) z_t
    
    where A is constrained to have spectral radius < 1 for stability.
    
    Implementation:
        A = tanh(W_A c_t) ⊙ A_base
        B = W_B [c_t || z_t]
    
    where A_base is initialized with small eigenvalues.
    """
    
    def __init__(self, d_w: int, d_c: int, d_z: int):
        """
        Args:
            d_w: Wave state dimension
            d_c: Context dimension
            d_z: Driving signal dimension
        """
        super().__init__()
        
        self.d_w = d_w
        self.d_c = d_c
        self.d_z = d_z
        
        # Base A matrix: diagonal + low-rank for stability
        # A_base = diag(a) + U V^T where a ∈ (-1, 1)
        self.A_diag = nn.Parameter(torch.zeros(d_w))
        self.A_U = nn.Parameter(torch.randn(d_w, d_w // 4) * 0.01)
        self.A_V = nn.Parameter(torch.randn(d_w, d_w // 4) * 0.01)
        
        # Context-dependent modulation of A
        self.A_modulation = nn.Sequential(
            nn.Linear(d_c, d_w),
            nn.Tanh(),  # Bounded modulation
        )
        
        # B matrix network
        self.B_net = nn.Linear(d_c + d_z, d_w)
    
    def get_A_matrix(self, c_t: torch.Tensor) -> torch.Tensor:
        """
        Compute context-dependent A matrix.
        
        Args:
            c_t: Context [d_c]
        
        Returns:
            A: State transition matrix [d_w, d_w]
        """
        # Base A: diagonal + low-rank
        A_diag_stable = torch.tanh(self.A_diag) * 0.9  # Keep eigenvalues < 1
        A_base = torch.diag(A_diag_stable) + torch.matmul(self.A_U, self.A_V.t())
        
        # Context modulation (element-wise)
        modulation = self.A_modulation(c_t)  # [d_w]
        A = A_base * modulation.unsqueeze(0)  # Broadcasting
        
        return A
    
    def forward(
        self,
        w_prev: torch.Tensor,
        z_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Wave subsystem update.
        
        Args:
            w_prev: Previous wave state [d_w]
            z_t: Driving signal [d_z]
            c_t: Context [d_c]
        
        Returns:
            w_t: Updated wave state [d_w]
        """
        # Compute A matrix
        A = self.get_A_matrix(c_t)  # [d_w, d_w]
        
        # Linear transition: A w_{t-1}
        w_linear = torch.matmul(A, w_prev)  # [d_w]
        
        # Input term: B z_t
        B_input = torch.cat([c_t, z_t], dim=-1)
        w_input = self.B_net(B_input)  # [d_w]
        
        # Combined update
        w_t = w_linear + w_input
        
        return w_t


class ParticleSubsystem(nn.Module):
    """
    Particle subsystem: Local nonlinear dynamics.
    
    Math:
        p_t = p_{t-1} + φ_Θ(p_{t-1}, z_t, c_t)
    
    where φ_Θ is a nonlinear function (MLP) capturing high-frequency structure.
    """
    
    def __init__(self, d_p: int, d_z: int, d_c: int, hidden_dim: Optional[int] = None):
        """
        Args:
            d_p: Particle state dimension
            d_z: Driving signal dimension
            d_c: Context dimension
            hidden_dim: Hidden dimension (default: 2 * d_p)
        """
        super().__init__()
        
        self.d_p = d_p
        self.d_z = d_z
        self.d_c = d_c
        self.hidden_dim = hidden_dim or (2 * d_p)
        
        # Nonlinear update function φ_Θ
        self.phi = nn.Sequential(
            nn.Linear(d_p + d_z + d_c, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, d_p),
            nn.Tanh(),  # Bounded updates for stability
        )
    
    def forward(
        self,
        p_prev: torch.Tensor,
        z_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Particle subsystem update.
        
        Args:
            p_prev: Previous particle state [d_p]
            z_t: Driving signal [d_z]
            c_t: Context [d_c]
        
        Returns:
            p_t: Updated particle state [d_p]
        """
        # Concatenate inputs
        phi_input = torch.cat([p_prev, z_t, c_t], dim=-1)
        
        # Nonlinear update
        delta_p = self.phi(phi_input)  # [d_p]
        
        # Residual update
        p_t = p_prev + delta_p
        
        return p_t


class HOPECore(nn.Module):
    """
    HOPE Core: Wave-particle hybrid recurrence.
    
    Combines:
        1. Fusion network: z_t = P_Θ([s_{t-1} || e_t || c_t])
        2. Wave subsystem: w_t = A(c_t) w_{t-1} + B(c_t) z_t
        3. Particle subsystem: p_t = p_{t-1} + φ_Θ(p_{t-1}, z_t, c_t)
        4. Gating: g_t = σ(W_g [s_{t-1} || e_t || c_t])
        5. Mixed state: s_t = s_{t-1} + g_t ⊙ U_p p_t + (1-g_t) ⊙ U_w w_t
    
    Returns:
        Updated fast state (s_t, w_t, p_t)
    """
    
    def __init__(
        self,
        d_s: int,
        d_w: int,
        d_p: int,
        d_e: int,
        d_c: int,
        d_z: int = 128,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            d_s: Fast state dimension
            d_w: Wave state dimension
            d_p: Particle state dimension
            d_e: Encoded input dimension
            d_c: Context dimension
            d_z: Fusion/driving signal dimension
            use_layer_norm: Apply layer normalization for stability
        """
        super().__init__()
        
        self.d_s = d_s
        self.d_w = d_w
        self.d_p = d_p
        self.d_e = d_e
        self.d_c = d_c
        self.d_z = d_z
        self.use_layer_norm = use_layer_norm
        
        # 1. Fusion network: z_t = P_Θ([s_{t-1} || e_t || c_t])
        self.fusion_net = nn.Sequential(
            nn.Linear(d_s + d_e + d_c, d_z * 2),
            nn.LayerNorm(d_z * 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(d_z * 2, d_z),
        )
        
        # 2. Wave subsystem
        self.wave = WaveSubsystem(d_w, d_c, d_z)
        
        # 3. Particle subsystem
        self.particle = ParticleSubsystem(d_p, d_z, d_c)
        
        # 4. Gating network: g_t = σ(W_g [s_{t-1} || e_t || c_t])
        self.gate_net = nn.Sequential(
            nn.Linear(d_s + d_e + d_c, d_s),
            nn.Sigmoid(),
        )
        
        # 5. Projection matrices: U_w, U_p
        self.U_w = nn.Linear(d_w, d_s, bias=False)
        self.U_p = nn.Linear(d_p, d_s, bias=False)
        
        # Optional layer norm for output
        self.output_norm = nn.LayerNorm(d_s) if use_layer_norm else nn.Identity()
    
    def forward(
        self,
        fast_prev: FastState,
        e_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> FastState:
        """
        HOPE core recurrence step.
        
        Args:
            fast_prev: Previous fast state (s_{t-1}, w_{t-1}, p_{t-1})
            e_t: Encoded input [d_e]
            c_t: Context from CMS [d_c]
        
        Returns:
            fast_next: Updated fast state (s_t, w_t, p_t)
        """
        s_prev, w_prev, p_prev = fast_prev.s, fast_prev.w, fast_prev.p
        
        # 1. Fusion: z_t = P_Θ([s_{t-1} || e_t || c_t])
        fusion_input = torch.cat([s_prev, e_t, c_t], dim=-1)
        z_t = self.fusion_net(fusion_input)  # [d_z]
        
        # 2. Wave update: w_t = A(c_t) w_{t-1} + B(c_t) z_t
        w_t = self.wave(w_prev, z_t, c_t)  # [d_w]
        
        # 3. Particle update: p_t = p_{t-1} + φ_Θ(p_{t-1}, z_t, c_t)
        p_t = self.particle(p_prev, z_t, c_t)  # [d_p]
        
        # 4. Gating: g_t = σ(W_g [s_{t-1} || e_t || c_t])
        g_t = self.gate_net(fusion_input)  # [d_s]
        
        # 5. Mixed state update
        # Project wave and particle to state space
        w_proj = self.U_w(w_t)  # [d_s]
        p_proj = self.U_p(p_t)  # [d_s]
        
        # Gated mixing: s_t = s_{t-1} + g_t ⊙ p_proj + (1-g_t) ⊙ w_proj
        s_t = s_prev + g_t * p_proj + (1 - g_t) * w_proj
        
        # Optional normalization
        s_t = self.output_norm(s_t)
        
        return FastState(s=s_t, w=w_t, p=p_t)
