"""
Continuous Memory System (CMS)

Hierarchical content-addressable memory with decay and write operations.

Math:
    Read:  c_t = CMS_Read(M_{t-1}, s_{t-1}, e_t)
    Write: M_t = CMS_Write(M_{t-1}, s_t, e_t, r_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math

from hope_impl.state import CMSMemory, MemoryLevel


class CMSRead(nn.Module):
    """
    CMS Read: Content-addressable hierarchical retrieval.
    
    Steps:
        1. Generate query: q_t = Q_ψ(s_{t-1}, e_t)
        2. Per-level attention: α_t^(ℓ) = softmax(K^(ℓ) q_t / √d_k)
        3. Context retrieval: c_t^(ℓ) = Σ_i α_{t,i}^(ℓ) M_i^(ℓ)
        4. Hierarchical mixing: c_t = Σ_ℓ β_t^(ℓ) U^(ℓ) c_t^(ℓ)
    
    Returns:
        q_t: Query vector
        c_t: Mixed context from all levels
        attention_weights: Per-level attention weights (for analysis)
    """
    
    def __init__(
        self,
        d_s: int,
        d_e: int,
        d_k: int,
        d_c: int,
        num_levels: int,
        cms_dims: List[int],
        temperature: float = 1.0,
    ):
        """
        Args:
            d_s: Fast state dimension
            d_e: Encoded input dimension
            d_k: Key/query dimension
            d_c: Output context dimension
            num_levels: Number of CMS levels
            cms_dims: Dimensions per level [d_0, d_1, ..., d_L]
            temperature: Temperature for attention softmax (default: 1.0 = √d_k scaling)
        """
        super().__init__()
        
        self.d_s = d_s
        self.d_e = d_e
        self.d_k = d_k
        self.d_c = d_c
        self.num_levels = num_levels
        self.cms_dims = cms_dims
        self.temperature = temperature
        
        # Query network: q_t = Q_ψ(s_{t-1}, e_t)
        from hope_impl.encoders import QueryNetwork
        self.query_net = QueryNetwork(d_s, d_e, d_k)
        
        # Per-level projection matrices: U^(ℓ) ∈ ℝ^{d_c × d_ℓ}
        self.level_projections = nn.ModuleList([
            nn.Linear(cms_dims[ell], d_c, bias=False)
            for ell in range(num_levels)
        ])
        
        # Mixing network: β_t = softmax(W_β [c_t^(0) || ... || c_t^(L)])
        self.mixing_net = nn.Sequential(
            nn.Linear(num_levels * d_c, num_levels),
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        cms: CMSMemory,
        s_prev: torch.Tensor,
        e_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Read from CMS memory.
        
        Args:
            cms: CMS memory state
            s_prev: Previous fast state [d_s]
            e_t: Encoded input [d_e]
        
        Returns:
            q_t: Query vector [d_k]
            c_t: Mixed context [d_c]
            attention_weights: List of attention weights per level
        """
        # 1. Generate query
        q_t = self.query_net(s_prev, e_t)  # [d_k]
        
        # 2. Per-level attention and retrieval
        level_contexts = []
        attention_weights = []
        
        for ell, level in enumerate(cms.levels):
            # Attention: α_t^(ℓ) = softmax(K^(ℓ) q_t / √d_k)
            # K^(ℓ): [N_ℓ, d_k], q_t: [d_k] -> scores: [N_ℓ]
            scores = torch.matmul(level.K, q_t)  # [N_ℓ]
            
            # Temperature scaling
            if self.temperature == 1.0:
                scores = scores / math.sqrt(self.d_k)  # Standard scaling
            else:
                scores = scores / (math.sqrt(self.d_k) * self.temperature)
            
            alpha = F.softmax(scores, dim=0)  # [N_ℓ]
            attention_weights.append(alpha)
            
            # Context retrieval: c_t^(ℓ) = Σ_i α_{t,i}^(ℓ) M_i^(ℓ)
            # alpha: [N_ℓ], M: [N_ℓ, d_ℓ] -> c: [d_ℓ]
            c_level = torch.matmul(alpha, level.M)  # [d_ℓ]
            
            # Project to common dimension: U^(ℓ) c_t^(ℓ)
            c_level_proj = self.level_projections[ell](c_level)  # [d_c]
            level_contexts.append(c_level_proj)
        
        # 3. Hierarchical mixing
        # Concatenate all level contexts
        all_contexts = torch.stack(level_contexts, dim=0)  # [num_levels, d_c]
        all_contexts_flat = all_contexts.flatten()  # [num_levels * d_c]
        
        # Compute mixing weights: β_t ∈ ℝ^{num_levels}
        beta = self.mixing_net(all_contexts_flat)  # [num_levels]
        
        # Mixed context: c_t = Σ_ℓ β_t^(ℓ) U^(ℓ) c_t^(ℓ)
        c_t = torch.matmul(beta, all_contexts)  # [d_c]
        
        return q_t, c_t, attention_weights


class CMSWrite(nn.Module):
    """
    CMS Write: Discrete jump map for memory updates.
    
    For each level ℓ:
        1. Compute event signal: z_t^(ℓ) = Z_ℓ(s_t, c_t^(ℓ-1), e_t)
        2. Compute write gate: g_t^(ℓ) = σ(W_g^(ℓ) z_t^(ℓ))
        3. Compute write value: v_t^(ℓ) = V_ℓ(z_t^(ℓ))
        4. Compute write key: k_t^(ℓ) = K_ℓ(z_t^(ℓ))
        5. Compute write weights: α̃_t^(ℓ) = softmax(K_{t-1}^(ℓ) k_t^(ℓ))
        6. Update: M_t^(ℓ) = (1-d_ℓ)M_{t-1}^(ℓ) + g_t^(ℓ)(α̃_t^(ℓ) ⊗ v_t^(ℓ))
    """
    
    def __init__(
        self,
        d_s: int,
        d_e: int,
        d_c: int,
        d_k: int,
        num_levels: int,
        cms_dims: List[int],
        d_z: int = 128,  # Event signal dimension
        sparse_write_threshold: float = 0.1,  # Only write if g_t > threshold
    ):
        """
        Args:
            d_s: Fast state dimension
            d_e: Encoded input dimension
            d_c: Context dimension
            d_k: Key dimension
            num_levels: Number of CMS levels
            cms_dims: Dimensions per level [d_0, d_1, ..., d_L]
            d_z: Event signal dimension
            sparse_write_threshold: Threshold for sparse writes
        """
        super().__init__()
        
        self.d_s = d_s
        self.d_e = d_e
        self.d_c = d_c
        self.d_k = d_k
        self.num_levels = num_levels
        self.cms_dims = cms_dims
        self.d_z = d_z
        self.sparse_write_threshold = sparse_write_threshold
        
        # Per-level event signal networks
        self.event_nets = nn.ModuleList()
        for ell in range(num_levels):
            if ell == 0:
                # Level 0: no lower level, use s_t + e_t
                input_dim = d_s + d_e
            else:
                # Level ℓ > 0: use s_t + c_t^(ℓ-1) + e_t
                input_dim = d_s + d_c + d_e
            
            event_net = nn.Sequential(
                nn.Linear(input_dim, d_z),
                nn.LayerNorm(d_z),
                nn.ReLU(),
            )
            self.event_nets.append(event_net)
        
        # Per-level write gate networks
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_z, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_levels)
        ])
        
        # Per-level write value networks
        self.value_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_z, cms_dims[ell]),
                nn.Tanh(),  # Bounded values for stability
            )
            for ell in range(num_levels)
        ])
        
        # Per-level write key networks
        self.key_nets = nn.ModuleList([
            nn.Linear(d_z, d_k)
            for _ in range(num_levels)
        ])
    
    def forward(
        self,
        cms: CMSMemory,
        s_t: torch.Tensor,
        e_t: torch.Tensor,
        level_contexts: Optional[List[torch.Tensor]] = None,
    ) -> CMSMemory:
        """
        Write to CMS memory.
        
        Args:
            cms: Current CMS memory state
            s_t: Current fast state [d_s]
            e_t: Encoded input [d_e]
            level_contexts: Optional per-level contexts from read [num_levels, d_c]
        
        Returns:
            Updated CMS memory
        """
        updated_levels = []
        
        for ell, level in enumerate(cms.levels):
            # 1. Compute event signal z_t^(ℓ)
            if ell == 0:
                # Level 0: no lower level
                event_input = torch.cat([s_t, e_t], dim=-1)
            else:
                # Level ℓ > 0: use context from level ℓ-1
                if level_contexts is not None and ell > 0:
                    c_prev = level_contexts[ell - 1]
                else:
                    # Fallback: use zeros if contexts not provided
                    c_prev = torch.zeros(self.d_c, device=s_t.device, dtype=s_t.dtype)
                event_input = torch.cat([s_t, c_prev, e_t], dim=-1)
            
            z_t = self.event_nets[ell](event_input)  # [d_z]
            
            # 2. Compute write gate g_t^(ℓ)
            g_t = self.gate_nets[ell](z_t).squeeze(-1)  # scalar
            
            # 3. Sparse write optimization
            if g_t.item() < self.sparse_write_threshold:
                # Only apply decay, skip write
                M_new = (1 - level.decay) * level.M
                K_new = (1 - level.decay) * level.K
            else:
                # 4. Compute write value v_t^(ℓ)
                v_t = self.value_nets[ell](z_t)  # [d_ℓ]
                
                # 5. Compute write key k_t^(ℓ)
                k_t = self.key_nets[ell](z_t)  # [d_k]
                
                # 6. Compute write weights α̃_t^(ℓ) (content-based addressing)
                scores = torch.matmul(level.K, k_t)  # [N_ℓ]
                scores = scores / math.sqrt(self.d_k)
                alpha_tilde = F.softmax(scores, dim=0)  # [N_ℓ]
                
                # 7. Memory update: M_t^(ℓ) = (1-d_ℓ)M_{t-1}^(ℓ) + g_t^(ℓ)(α̃_t^(ℓ) ⊗ v_t^(ℓ))
                # Outer product: α̃_t^(ℓ) ⊗ v_t^(ℓ) ∈ ℝ^{N_ℓ × d_ℓ}
                write_matrix = torch.outer(alpha_tilde, v_t)  # [N_ℓ, d_ℓ]
                M_new = (1 - level.decay) * level.M + g_t * write_matrix
                
                # 8. Key update (optional, same rule)
                key_write_matrix = torch.outer(alpha_tilde, k_t)  # [N_ℓ, d_k]
                K_new = (1 - level.decay) * level.K + g_t * key_write_matrix
            
            # Create updated level
            updated_level = MemoryLevel(M=M_new, K=K_new, decay=level.decay)
            updated_levels.append(updated_level)
        
        return CMSMemory(levels=updated_levels)
