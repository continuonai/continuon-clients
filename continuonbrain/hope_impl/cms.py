"""
Continuous Memory System (CMS)

Hierarchical content-addressable memory with decay and write operations.

Math:
    Read:  c_t = CMS_Read(M_{t-1}, s_{t-1}, e_t)
    Write: M_t = CMS_Write(M_{t-1}, s_t, e_t, r_t)
"""

import torch
print("DEBUG: CMS MODULE LOADED FROM", __file__)
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math

from .state import CMSMemory, MemoryLevel


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

        import sys
        print(f"DEBUG_CMS_INIT: d_s={d_s}, d_e={d_e}, d_k={d_k}, d_c={d_c}, num_levels={num_levels}, cms_dims={cms_dims}", file=sys.stderr)
        
        # Query network: q_t = Q_ψ(s_{t-1}, e_t)
        from .encoders import QueryNetwork
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
            # K^(ℓ): [N_ℓ, d_k], q_t: [B, d_k] -> scores: [B, N_ℓ]
            
            # Use F.linear: q_t @ K^T
            scores = F.linear(q_t, level.K)  # [B, N_ℓ]
            
            # Temperature scaling
            if self.temperature == 1.0:
                scores = scores / math.sqrt(self.d_k)  # Standard scaling
            else:
                scores = scores / (math.sqrt(self.d_k) * self.temperature)
            
            alpha = F.softmax(scores, dim=-1)  # [B, N_ℓ]
            attention_weights.append(alpha)
            
            # Context retrieval: c_t^(ℓ) = Σ_i α_{t,i}^(ℓ) M_i^(ℓ)
            # alpha: [N_ℓ], M: [N_ℓ, d_ℓ] -> c: [d_ℓ]
            c_level = torch.matmul(alpha, level.M)  # [d_ℓ]
            
            # Project to common dimension: U^(ℓ) c_t^(ℓ)
            c_level_proj = self.level_projections[ell](c_level)  # [d_c]
            level_contexts.append(c_level_proj)
        
        
        # 3. Hierarchical mixing
        # Concatenate all level contexts
        # level_contexts: List of [B, d_c] tensors
        all_contexts = torch.stack(level_contexts, dim=1)  # [B, num_levels, d_c]
        
        # Flatten for mixing network: [B, num_levels * d_c]
        all_contexts_flat = all_contexts.flatten(start_dim=1)
        
        # Compute mixing weights: β_t ∈ ℝ^{B, num_levels}
        beta = self.mixing_net(all_contexts_flat)  # [B, num_levels]
        
        # Mixed context: c_t = Σ_ℓ β_t^(ℓ) U^(ℓ) c_t^(ℓ)
        # BMM: [B, 1, num_levels] @ [B, num_levels, d_c] -> [B, 1, d_c]
        c_t = torch.matmul(beta.unsqueeze(1), all_contexts).squeeze(1)  # [B, d_c]
        
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
        
        # Per-level event signal networks (WaveCore Integration)
        self.event_nets = nn.ModuleList()
        
        # We use a fixed sequence length for the spectral block window
        self.window_size = 64 
        
        from .wave_core import SpectralBlock
        
        for ell in range(num_levels):
            if ell == 0:
                # Level 0: no lower level, use s_t + e_t
                input_dim = d_s + d_e
            else:
                # Level ℓ > 0: use s_t + c_t^(ℓ-1) + e_t
                input_dim = d_s + d_c + d_e
            
            # Project input to d_z for spectral processing
            self.event_nets.append(nn.Sequential(
                nn.Linear(input_dim, d_z),
                SpectralBlock(d_z, self.window_size),
                nn.LayerNorm(d_z),
                # We will take the last token of the sequence
            ))
        
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
        history_buffer: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
    ) -> CMSMemory:
        """
        Write to CMS memory.
        
        Args:
            cms: Current CMS memory state
            s_t: Current fast state [d_s]
            e_t: Encoded input [d_e]
            level_contexts: Optional per-level contexts from read [num_levels, d_c]
            history_buffer: List of (s, c_prev, e) tuples for spectral window [window_size]
                            If None, we construct a 1-step window (no efficient spectral usage)
        
        Returns:
            Updated CMS memory
        """
        updated_levels = []
        
        # Construct current input tuple needed for history
        # Note: We need c_prev for level > 0, which depends on the level index.
        # This implies we need to reconstruct the input sequence PER LEVEL.
        
        # Optimization: We assume history_buffer contains raw inputs (s, e) ?
        # Actually, `history_buffer` passed from Brain should likely contain (s, e).
        # We need to reconstruct c_prev locally or store it?
        # For simplicity in this iteration: We'll construct the input sequence just for the current write
        # by looking at the history of (s, e) and assuming c_prev ~ 0 or just using current c_prev for the whole window (approximation).
        # Better: Brain stores (s, e, c) history.
        
        for ell, level in enumerate(cms.levels):
            # 1. Compute event signal z_t^(ℓ)
            # We need a sequence [batches, window, input_dim]
            
            # Prepare single step input first
            if ell == 0:
                # Level 0: no lower level
                current_input = torch.cat([s_t, e_t], dim=-1) # [d_s + d_e]
            else:
                # Level ℓ > 0: use context from level ℓ-1
                if level_contexts is not None and ell > 0:
                    c_prev = level_contexts[ell - 1]
                else:
                    # c_prev needs to match s_t batch size
                    if s_t.dim() == 2:
                        batch_size = s_t.size(0)
                        c_prev = torch.zeros(batch_size, self.d_c, device=s_t.device, dtype=s_t.dtype)
                    else:
                        c_prev = torch.zeros(self.d_c, device=s_t.device, dtype=s_t.dtype)
                
                current_input = torch.cat([s_t, c_prev, e_t], dim=-1) # [d_s + d_c + d_e]
            
            # Construct Sequence
            # If no history, just use current (window=1). Spectral block handles it (but n_fft small).
            # Ideally we want a [1, window, input_dim] tensor.
            
            if history_buffer is not None and len(history_buffer) > 0:
                # We need to adhere to the per-level input schema for the whole buffer.
                # This is tricky because `c_prev` for level ℓ depends on previous reads.
                # If the buffer stores (s, e, all_c), we can reconstruct.
                # Let's assume history_buffer is List of (s, e, list_of_c).
                
                # Check format of history_buffer[0]
                # It is likely complex.
                # Fallback implementation:
                # If we can't easily reconstruct history for level ℓ, we just repeat current input
                # or verify if history_buffer is just 'z_inputs' from previous steps?
                # A robust way: Brain passes fully formed 'z_input' history for each level? Too much memory.
                
                # NEW STRATEGY: Brain passes list of (s, e, c_mixed).
                # But we need c_prev^(ell-1).
                # Approximation: Use current c_prev for past events? No, that breaks causality.
                # 
                # Simplification for v1: Use a simple rolling buffer OF THE INPUTS to the network.
                # We will process simply the current step, but wrapped in a sequence of 1?
                # No, that defeats the purpose.
                
                # Let's assume for now we just process the current step as a window of 1
                # UNTIL we update Brain.py to pass the right buffer.
                # But we defined the SpectralBlock with window=64.
                # So we pad?
                pass

            # V1 Integration: Just use current step reshaped to [1, 1, Dim] 
            # and pad to window size? Or just pass 1?
            # SpectralBlock expects [B, Seq, D].
            
            # Create a localized buffer for this step
            # Note: This is inefficient for a "sliding window" if we recompute every step.
            # But correct for a "Brain" that runs step-by-step.
            # To do this efficiently, we'd need a stateful WaveCore (like an RNN).
            # But WaveCore is FFT-based (non-causal within window, or causal masked?).
            # The Toy model is whole-sequence.
            # If we run this step-by-step, we must re-run the FFT on the last N steps.
            
            # For this step, let's assemble the history from the buffer.
            # We assume history_buffer is inputs pertinent to this level?
            # Or we reconstruct.
            
            # Let's assume history_buffer is a pre-prepared tensor [Window, InputDim]?
            # No, that assumes Brain knows per-level details.
            
            # Let's assume input is [1, 1, input_dim] and we rely on Brain to provide
            # a `z_history` if needed, OR we just ignore history for this momentary integration
            # and verify shapes work.
            
            # Prepare sequence for SpectralBlock: [B, Seq, D]
            # current_input: [B, D] (assuming batched)
            
            # If current_input is unbatched [D]:
            if current_input.dim() == 1:
                seq_input = current_input.unsqueeze(0).unsqueeze(0) # [1, 1, D]
            else:
                # Batched [B, D] -> [B, 1, D]
                seq_input = current_input.unsqueeze(1)
            
            # Pad to window size 64 to mimic spectral window? 
            # Or just let FFT handle length 1 (DC component only).
            # For meaningful wave dynamics, we need length > 1.
            
            # If we just do length 1, it works but is trivial.
            # This is acceptable for "Integration" task (getting plumbing working).
            # The "Improvement" task can wire up the buffer.
            
            z_seq = self.event_nets[ell](seq_input) # [B, Seq, d_z]
            z_t = z_seq[:, -1, :] # Take last token: [B, d_z]
            z_t = z_t.squeeze(0)

            
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
                # Compute attention scores
                # level.K: [N_ℓ, d_k]
                # k_t: [d_k] (or [B, d_k] if batched)
                
                # Use F.linear for flexible batch handling: scores = k_t @ K^T
                # F.linear(input, weight) computes input @ weight.T
                # Here, input is k_t, weight is level.K.
                # So, scores = k_t @ level.K.T
                scores = F.linear(k_t, level.K)  # [N_ℓ] (or [B, N_ℓ] if k_t is [B, d_k])
                scores = scores / math.sqrt(self.d_k)
                
                # Softmax
                alpha_tilde = F.softmax(scores, dim=-1)  # [N_ℓ] (or [B, N_ℓ])
                
                # 7. Memory update: M_t^(ℓ) = (1-d_ℓ)M_{t-1}^(ℓ) + g_t^(ℓ)(α̃_t^(ℓ) ⊗ v_t^(ℓ))
                # Outer product: α̃_t^(ℓ) ⊗ v_t^(ℓ) ∈ ℝ^{N_ℓ × d_ℓ}
                # If alpha_tilde is [N_ℓ] and v_t is [d_ℓ], torch.outer is correct.
                # If alpha_tilde is [B, N_ℓ] and v_t is [B, d_ℓ], we need batched outer product.
                # For now, assuming single item (no batch dim on k_t, v_t, g_t)
                write_matrix = torch.outer(alpha_tilde, v_t)  # [N_ℓ, d_ℓ]
                M_new = (1 - level.decay) * level.M + g_t * write_matrix
                
                # 8. Key update (optional, same rule)
                key_write_matrix = torch.outer(alpha_tilde, k_t)  # [N_ℓ, d_k]
                K_new = (1 - level.decay) * level.K + g_t * key_write_matrix
            
            # Create updated level
            updated_level = MemoryLevel(M=M_new, K=K_new, decay=level.decay)
            updated_levels.append(updated_level)
        
        return CMSMemory(levels=updated_levels)
