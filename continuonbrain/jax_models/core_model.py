"""
CoreModel: JAX/Flax Implementation

Canonical v0 seed model inspired by HOPE architecture.
This model runs identically on Pi (sanity checks) and cloud TPU (full training).

Architecture:
- Fast state (s_t): low-level reactive state
- Wave state (w_t): SSM-like global coordination
- Particle state (p_t): local nonlinear dynamics
- CMS memory integration (hierarchical memory levels)
- AINA-inspired policy head for manipulation tasks
"""

from typing import Any, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

from .config import CoreModelConfig
from .mamba_ssm import MambaLikeWave
from .batch_utils import (
    normalize_inputs,
    denormalize_outputs,
    validate_batch_consistency,
)


class InputEncoder(nn.Module):
    """
    Input encoder combining observation, action, reward, and optional modalities.

    Encodes:
        x_obs: Observation (can be vector, image, etc.)
        a_prev: Previous action
        r_t: Scalar reward
        object_features: Optional detected object features
        audio_features: Optional audio mel spectrogram features

    Output:
        e_t ∈ ℝ^{d_e}: Unified encoded feature
    """
    config: CoreModelConfig
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(
        self,
        x_obs: jnp.ndarray,
        a_prev: jnp.ndarray,
        r_t: jnp.ndarray,
        object_features: Optional[jnp.ndarray] = None,
        audio_features: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Encode inputs into unified feature.

        Args:
            x_obs: Observation (pixels or tokens)
            a_prev: Previous action
            r_t: Scalar reward
            object_features: Optional [B, max_objects, object_dim]
            audio_features: Optional [B, num_frames, audio_dim] mel spectrogram
        """
        d_e = self.config.d_e
        hidden_dim = 2 * d_e

        # Ensure batch dimension
        if x_obs.ndim == 1 and self.config.obs_type != "vqvae":
            x_obs = x_obs[None, :]
        if self.config.obs_type == "vqvae" and x_obs.ndim == 1:
            # tokens: [L] -> [1, L]
            x_obs = x_obs[None, :]

        if a_prev.ndim == 1:
            a_prev = a_prev[None, :]
        if r_t.ndim == 0:
            r_t = r_t[None, None]
        elif r_t.ndim == 1:
            r_t = r_t[None, :]

        batch_size = x_obs.shape[0]

        # Observation encoder
        if self.config.obs_type == "vector":
            obs_feat = nn.Dense(hidden_dim)(x_obs)
            obs_feat = nn.LayerNorm()(obs_feat)
            obs_feat = nn.relu(obs_feat)
            obs_feat = nn.Dense(d_e)(obs_feat)
        elif self.config.obs_type == "image":
            # Simple CNN for image observations
            x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x_obs)
            x = nn.relu(x)
            x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
            x = nn.relu(x)
            x = jnp.mean(x, axis=(2, 3))  # [B, 64]
            obs_feat = nn.Dense(d_e)(x)
        elif self.config.obs_type == "vqvae":
            # VQ-VAE Token Encoder
            # x_obs is [B, L] integers
            vocab_size = self.config.num_vq_vocab
            embed_dim = 64

            x = nn.Embed(vocab_size, embed_dim)(x_obs) # [B, L, 64]
            # Simple Flatten + Dense (fastest for Pi)
            x = x.reshape((x.shape[0], -1)) # [B, L*64]
            obs_feat = nn.Dense(d_e)(x)
        else:
            raise ValueError(f"Unknown obs_type: {self.config.obs_type}")

        # Object Feature Encoder (Permutation Invariant)
        obj_feat = jnp.zeros((batch_size, d_e))
        if self.config.use_object_features and object_features is not None:
             # object_features: [B, N, D]
             # Encode each object: MLP(obj) -> [B, N, d_e]
             x_obj = nn.Dense(d_e)(object_features)
             x_obj = nn.relu(x_obj)
             # Max pool over objects (permutation invariant)
             obj_feat = jnp.max(x_obj, axis=1) # [B, d_e]

        # Audio Feature Encoder
        audio_feat = jnp.zeros((batch_size, d_e))
        if self.config.use_audio_features and audio_features is not None:
            # audio_features: [B, T, F] where T=frames, F=mel_bands
            if self.config.audio_encoder_type == "cnn":
                # 1D CNN over time dimension
                # Reshape to [B, T, F, 1] for Conv
                x_audio = audio_features[:, :, :, None]
                x_audio = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 1), padding='SAME')(x_audio)
                x_audio = nn.relu(x_audio)
                x_audio = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 1), padding='SAME')(x_audio)
                x_audio = nn.relu(x_audio)
                x_audio = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 1), padding='SAME')(x_audio)
                x_audio = nn.relu(x_audio)
                # Global average pool over time and frequency
                x_audio = jnp.mean(x_audio, axis=(1, 2))  # [B, 64]
                audio_feat = nn.Dense(d_e)(x_audio)
            else:
                # Simple transformer-style encoding
                # Flatten and project: [B, T, F] -> [B, T*F] -> [B, d_e]
                x_audio = audio_features.reshape((batch_size, -1))
                audio_feat = nn.Dense(hidden_dim)(x_audio)
                audio_feat = nn.LayerNorm()(audio_feat)
                audio_feat = nn.relu(audio_feat)
                audio_feat = nn.Dense(d_e)(audio_feat)

        # Action encoder
        action_feat = nn.Dense(d_e // 2)(a_prev)
        action_feat = nn.relu(action_feat)

        # Reward encoder
        reward_feat = nn.Dense(d_e // 4)(r_t)
        reward_feat = nn.relu(reward_feat)

        # Fusion layer
        # Concat: [obs, action, reward] + [objects] + [audio] if enabled
        features_to_combine = [obs_feat, action_feat, reward_feat]
        if self.config.use_object_features:
            features_to_combine.append(obj_feat)
        if self.config.use_audio_features:
            features_to_combine.append(audio_feat)

        combined = jnp.concatenate(features_to_combine, axis=-1)

        e_t = nn.Dense(hidden_dim)(combined)
        e_t = nn.LayerNorm()(e_t)
        e_t = nn.relu(e_t)
        e_t = nn.Dense(d_e)(e_t)
        e_t = nn.LayerNorm()(e_t)

        return e_t


class QueryNetwork(nn.Module):
    """
    Query network for CMS read operation.
    
    Computes query vector from fast state and encoded input.
    
    Math:
        q_t = Q_ψ(s_{t-1}, e_t) ∈ ℝ^{d_k}
    """
    config: CoreModelConfig
    
    @nn.compact
    def __call__(self, s_prev: jnp.ndarray, e_t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute query vector.
        
        Args:
            s_prev: Previous fast state [d_s] or [B, d_s]
            e_t: Encoded input [d_e] or [B, d_e]
        
        Returns:
            q_t: Query vector [d_k] or [B, d_k]
        """
        hidden_dim = self.config.d_s + self.config.d_e
        combined = jnp.concatenate([s_prev, e_t], axis=-1)
        
        q_t = nn.Dense(hidden_dim)(combined)
        q_t = nn.LayerNorm()(q_t)
        q_t = nn.relu(q_t)
        q_t = nn.Dense(self.config.d_k)(q_t)
        
        return q_t


class WaveSubsystem(nn.Module):
    """
    Wave subsystem: SSM-like global linear dynamics.
    
    Math:
        w_t = A(c_t, Θ_t) w_{t-1} + B(c_t, Θ_t) z_t
    
    where A is constrained to have spectral radius < 1 for stability.
    """
    config: CoreModelConfig
    
    def setup(self):
        d_w = self.config.d_w
        d_c = self.config.d_c
        
        # Base A matrix: diagonal + low-rank for stability
        # A_base = diag(a) + U V^T where a ∈ (-1, 1)
        self.A_diag = self.param('A_diag', nn.initializers.zeros, (d_w,))
        self.A_U = self.param('A_U', nn.initializers.normal(stddev=0.01), (d_w, d_w // 4))
        self.A_V = self.param('A_V', nn.initializers.normal(stddev=0.01), (d_w, d_w // 4))
        
        # Context-dependent modulation of A
        self.A_modulation = nn.Dense(d_w, name='A_modulation')
        
        # B matrix network
        self.B_net = nn.Dense(d_w, name='B_net')
    
    def __call__(self, w_prev: jnp.ndarray, z_t: jnp.ndarray, c_t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute next wave state.
        
        Args:
            w_prev: Previous wave state [d_w] or [B, d_w]
            z_t: Driving signal [d_p] or [B, d_p]
            c_t: Context [d_c] or [B, d_c]
        
        Returns:
            w_t: Next wave state [d_w] or [B, d_w]
        """
        # Base A: diagonal + low-rank
        A_diag_stable = jnp.tanh(self.A_diag) * 0.9  # Keep eigenvalues < 1
        A_base = jnp.diag(A_diag_stable) + jnp.dot(self.A_U, self.A_V.T)  # [d_w, d_w]
        
        # Context modulation (element-wise)
        modulation = jnp.tanh(self.A_modulation(c_t))  # [B, d_w] or [d_w]
        
        # Broadcast A_base and apply modulation
        if w_prev.ndim == 1:
            A = A_base * modulation[:, None]
        else:
            A = A_base[None, :, :] * modulation[:, :, None]  # [B, d_w, d_w]
        
        # B matrix
        B_input = jnp.concatenate([c_t, z_t], axis=-1)
        B = self.B_net(B_input)  # [B, d_w] or [d_w]
        
        # Compute w_t
        if w_prev.ndim == 1:
            w_t = jnp.dot(A, w_prev) + B
        else:
            # Batch matrix multiplication: [B, d_w, d_w] @ [B, d_w] -> [B, d_w]
            w_t = jnp.einsum('bij,bj->bi', A, w_prev) + B
        
        return w_t


class ParticleSubsystem(nn.Module):
    """
    Particle subsystem: local nonlinear dynamics.
    
    Math:
        p_t = p_{t-1} + φ_Θ(p_{t-1}, z_t, c_t)
    """
    config: CoreModelConfig
    
    @nn.compact
    def __call__(self, p_prev: jnp.ndarray, z_t: jnp.ndarray, c_t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute next particle state.
        
        Args:
            p_prev: Previous particle state [d_p] or [B, d_p]
            z_t: Driving signal [d_p] or [B, d_p]
            c_t: Context [d_c] or [B, d_c]
        
        Returns:
            p_t: Next particle state [d_p] or [B, d_p]
        """
        hidden_dim = self.config.d_p + self.config.d_c
        combined = jnp.concatenate([p_prev, c_t], axis=-1)
        
        delta_p = nn.Dense(hidden_dim)(combined)
        delta_p = nn.LayerNorm()(delta_p)
        delta_p = nn.relu(delta_p)
        delta_p = nn.Dense(self.config.d_p)(delta_p)
        
        # Add residual connection
        p_t = p_prev + delta_p
        
        return p_t


class HOPECore(nn.Module):
    """
    HOPE Core Dynamics: Wave-particle hybrid recurrence.
    
    Math:
        Wave:     w_t = A(c_t, Θ_t) w_{t-1} + B(c_t, Θ_t) z_t
        Particle: p_t = p_{t-1} + φ_Θ(p_{t-1}, z_t, c_t)
        Gate:     g_t = σ(W_g [s_{t-1} || e_t || c_t])
        Mixed:    s_t = s_{t-1} + g_t ⊙ U_p p_t + (1-g_t) ⊙ U_w w_t
    """
    config: CoreModelConfig
    
    def setup(self):
        # Canonical seed wave dynamics: Mamba-like selective SSM (portable, stable).
        # Keep legacy WaveSubsystem available behind config flag.
        if getattr(self.config, "use_mamba_wave", False):
            self.wave = MambaLikeWave(
                d_w=self.config.d_w,
                d_in=self.config.d_p + self.config.d_c,
                state_dim=getattr(self.config, "mamba_state_dim", 1),
                dt_min=getattr(self.config, "mamba_dt_min", 1e-4),
                dt_scale=getattr(self.config, "mamba_dt_scale", 1.0),
                name="mamba_wave",
            )
        else:
            self.wave = WaveSubsystem(self.config)
        self.particle = ParticleSubsystem(self.config)
        
        # Gate network
        gate_input_dim = self.config.d_s + self.config.d_e + self.config.d_c
        self.gate_net = nn.Dense(self.config.d_s, name='gate_net')
        
        # Driving signal projection
        self.z_projection = nn.Dense(self.config.d_p, name='z_projection')
        
        # Projection matrices
        self.U_p = self.param('U_p', nn.initializers.xavier_uniform(), 
                              (self.config.d_s, self.config.d_p))
        self.U_w = self.param('U_w', nn.initializers.xavier_uniform(),
                              (self.config.d_s, self.config.d_w))
    
    def __call__(
        self,
        s_prev: jnp.ndarray,
        w_prev: jnp.ndarray,
        p_prev: jnp.ndarray,
        e_t: jnp.ndarray,
        c_t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute next fast, wave, and particle states.
        
        Args:
            s_prev: Previous fast state [d_s] or [B, d_s]
            w_prev: Previous wave state [d_w] or [B, d_w]
            p_prev: Previous particle state [d_p] or [B, d_p]
            e_t: Encoded input [d_e] or [B, d_e]
            c_t: Context [d_c] or [B, d_c]
        
        Returns:
            s_t: Next fast state [d_s] or [B, d_s]
            w_t: Next wave state [d_w] or [B, d_w]
            p_t: Next particle state [d_p] or [B, d_p]
        """
        # Driving signal: z_t = projection of e_t to d_p
        z_t = self.z_projection(e_t)
        
        # Wave update
        if getattr(self.config, "use_mamba_wave", False):
            w_in = jnp.concatenate([z_t, c_t], axis=-1)
            w_t = self.wave(w_prev, w_in)
        else:
            w_t = self.wave(w_prev, z_t, c_t)
        
        # Particle update
        p_t = self.particle(p_prev, z_t, c_t)
        
        # Gate computation
        gate_input = jnp.concatenate([s_prev, e_t, c_t], axis=-1)
        g_t = nn.sigmoid(self.gate_net(gate_input))  # [d_s] or [B, d_s]
        
        # Project wave and particle contributions
        if s_prev.ndim == 1:
            wave_contrib = jnp.dot(self.U_w, w_t)
            particle_contrib = jnp.dot(self.U_p, p_t)
        else:
            wave_contrib = jnp.dot(w_t, self.U_w.T)  # [B, d_s]
            particle_contrib = jnp.dot(p_t, self.U_p.T)  # [B, d_s]
        
        # Mix wave and particle
        s_t = s_prev + g_t * particle_contrib + (1 - g_t) * wave_contrib

        # Apply saturation limit to ALL states for stability
        if self.config.state_saturation_limit > 0:
            limit = self.config.state_saturation_limit
            s_t = jnp.clip(s_t, -limit, limit)
            w_t = jnp.clip(w_t, -limit, limit)
            p_t = jnp.clip(p_t, -limit, limit)

        return s_t, w_t, p_t


class CMSRead(nn.Module):
    """
    CMS Read: Content-addressable hierarchical retrieval.
    
    Steps:
        1. Generate query: q_t = Q_ψ(s_{t-1}, e_t)
        2. Per-level attention: α_t^(ℓ) = softmax(K^(ℓ) q_t / √d_k)
        3. Context retrieval: c_t^(ℓ) = Σ_i α_{t,i}^(ℓ) M_i^(ℓ)
        4. Hierarchical mixing: c_t = Σ_ℓ β_t^(ℓ) U^(ℓ) c_t^(ℓ)
    """
    config: CoreModelConfig
    
    def setup(self):
        self.query_net = QueryNetwork(self.config)
        
        # Per-level projection matrices: U^(ℓ) ∈ ℝ^{d_c × d_ℓ}
        # Note: Flax requires using a list comprehension in setup() or using a ModuleList-like pattern
        # We'll create them as a list of modules
        self.level_projections = [
            nn.Dense(self.config.d_c, use_bias=False, name=f'level_proj_{i}')
            for i in range(self.config.num_levels)
        ]
        
        # Mixing network: β_t = softmax(W_β [c_t^(0) || ... || c_t^(L)])
        self.mixing_net = nn.Dense(self.config.num_levels, name='mixing_net')
    
    def __call__(
        self,
        cms_memories: List[jnp.ndarray],  # List of [B, N_ℓ, d_ℓ] memory matrices
        cms_keys: List[jnp.ndarray],  # List of [B, N_ℓ, d_k] key matrices
        s_prev: jnp.ndarray,
        e_t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
        """
        Read from CMS memory.

        Note: Inputs are always batched [B, ...] since CoreModel.__call__ normalizes them.

        Args:
            cms_memories: List of memory matrices [B, N_ℓ, d_ℓ], one per level
            cms_keys: List of key matrices [B, N_ℓ, d_k], one per level
            s_prev: Previous fast state [B, d_s]
            e_t: Encoded input [B, d_e]

        Returns:
            q_t: Query vector [B, d_k]
            c_t: Mixed context [B, d_c]
            attention_weights: List of attention weights per level [B, N_ℓ]
        """
        # Generate query
        q_t = self.query_net(s_prev, e_t)  # [B, d_k]

        # Per-level attention and retrieval
        level_contexts = []
        attention_weights = []

        for level_idx, (M_level, K_level) in enumerate(zip(cms_memories, cms_keys)):
            # Compute attention: α = softmax(K q / √d_k)
            # K_level [B, N, d_k], q_t [B, d_k] -> scores [B, N]
            scores = jnp.einsum('bnd,bd->bn', K_level, q_t) / jnp.sqrt(float(self.config.d_k))
            attn = nn.softmax(scores, axis=-1)  # [B, N]
            # M_level [B, N, D], attn [B, N] -> c_level [B, D]
            c_level = jnp.einsum('bn,bnd->bd', attn, M_level)

            # Project to d_c
            level_proj = self.level_projections[level_idx]
            c_level_proj = level_proj(c_level)  # [B, d_c]
            level_contexts.append(c_level_proj)
            attention_weights.append(attn)

        # Hierarchical mixing
        combined_contexts = jnp.concatenate(level_contexts, axis=-1)  # [B, L * d_c]
        mixing_weights = nn.softmax(self.mixing_net(combined_contexts), axis=-1)  # [B, L]

        # Weighted sum: mixing_weights [B, L], level_contexts list of [B, d_c]
        c_t = sum(w[:, None] * c for w, c in zip(mixing_weights.T, level_contexts))

        return q_t, c_t, attention_weights


class CMSWrite(nn.Module):
    """
    CMS Write: Gated memory update with decay.
    
    Math:
        For each level ℓ:
            # Write gate (novelty-based)
            g_write = σ(W_g [s_t || e_t || r_t])
            
            # Value to write
            v_write = W_v [s_t || e_t]
            
            # Key to write
            k_write = W_k [s_t || e_t]
            
            # Decay existing memories
            M_t^(ℓ) = (1 - λ_ℓ) * M_{t-1}^(ℓ)
            K_t^(ℓ) = (1 - λ_ℓ) * K_{t-1}^(ℓ)
            
            # Gated write to slot 0 (FIFO-like)
            M_t^(ℓ)[0] = g_write * v_write + (1 - g_write) * M_t^(ℓ)[0]
            K_t^(ℓ)[0] = g_write * k_write + (1 - g_write) * K_t^(ℓ)[0]
    """
    config: CoreModelConfig
    
    @nn.compact
    def __call__(
        self,
        cms_memories: List[jnp.ndarray],
        cms_keys: List[jnp.ndarray],
        s_t: jnp.ndarray,
        e_t: jnp.ndarray,
        r_t: jnp.ndarray,
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Write to CMS memory.
        
        Args:
            cms_memories: List of [B, N_ℓ, d_ℓ] memory matrices
            cms_keys: List of [B, N_ℓ, d_k] key matrices
            s_t: Current fast state [B, d_s]
            e_t: Encoded input [B, d_e]
            r_t: Reward signal [B, 1]
        
        Returns:
            new_cms_memories: Updated memory matrices
            new_cms_keys: Updated key matrices
        """
        # Compute write features
        write_input = jnp.concatenate([s_t, e_t], axis=-1)  # [B, d_s + d_e]
        gate_input = jnp.concatenate([s_t, e_t, r_t], axis=-1)  # [B, d_s + d_e + 1]
        
        new_memories = []
        new_keys = []
        
        for level_idx, (M, K) in enumerate(zip(cms_memories, cms_keys)):
            decay = self.config.cms_decays[level_idx]
            mem_dim = self.config.cms_dims[level_idx]
            
            # Write gate: higher for novel/salient experiences
            g_write = nn.Dense(1, name=f"write_gate_{level_idx}")(gate_input)
            g_write = nn.sigmoid(g_write)  # [B, 1]
            
            # Value and key to write
            v_write = nn.Dense(mem_dim, name=f"write_value_{level_idx}")(write_input)  # [B, d_ℓ]
            k_write = nn.Dense(self.config.d_k, name=f"write_key_{level_idx}")(write_input)  # [B, d_k]
            
            # Apply decay to existing memories (exponential forgetting)
            M_decayed = M * (1.0 - decay)  # [B, N, d_ℓ]
            K_decayed = K * (1.0 - decay)  # [B, N, d_k]
            
            # Gated write to first slot (slot 0)
            # M_new[0] = g * v + (1-g) * M_old[0]
            M_slot0_old = M_decayed[:, 0, :]  # [B, d_ℓ]
            K_slot0_old = K_decayed[:, 0, :]  # [B, d_k]
            
            M_slot0_new = g_write * v_write + (1 - g_write) * M_slot0_old  # [B, d_ℓ]
            K_slot0_new = g_write * k_write + (1 - g_write) * K_slot0_old  # [B, d_k]
            
            # Reconstruct memory with updated slot 0
            M_new = M_decayed.at[:, 0, :].set(M_slot0_new)
            K_new = K_decayed.at[:, 0, :].set(K_slot0_new)
            
            new_memories.append(M_new)
            new_keys.append(K_new)
        
        return new_memories, new_keys


class OutputDecoder(nn.Module):
    """
    Output decoder for producing predictions/actions.
    
    Decodes fast state and context into output.
    
    Math:
        y_t = H_ω(s_t, c_t)
    """
    config: CoreModelConfig
    output_dim: int
    
    @nn.compact
    def __call__(self, s_t: jnp.ndarray, c_t: jnp.ndarray) -> jnp.ndarray:
        """
        Decode state and context into output.
        
        Args:
            s_t: Fast state [d_s] or [B, d_s]
            c_t: Context [d_c] or [B, d_c]
        
        Returns:
            y_t: Output [output_dim] or [B, output_dim]
        """
        hidden_dim = self.config.d_s + self.config.d_c
        combined = jnp.concatenate([s_t, c_t], axis=-1)
        
        y_t = nn.Dense(hidden_dim)(combined)
        y_t = nn.LayerNorm()(y_t)
        y_t = nn.relu(y_t)
        y_t = nn.Dense(hidden_dim // 2)(y_t)
        y_t = nn.relu(y_t)
        y_t = nn.Dense(self.output_dim)(y_t)
        
        # Add activation based on output type
        if self.config.output_type == "continuous":
            y_t = jnp.tanh(y_t)  # Bounded continuous outputs
        # For discrete, return logits (no activation)
        
        return y_t


class CoreModel(nn.Module):
    """
    CoreModel: Canonical v0 seed model.
    
    This is the main model that runs identically on Pi and cloud TPU.
    Architecture inspired by HOPE looping system with CMS memory integration.
    """
    config: CoreModelConfig
    obs_dim: int
    action_dim: int
    output_dim: int
    
    def setup(self):
        # Input encoder
        self.encoder = InputEncoder(self.config, self.obs_dim, self.action_dim)
        
        # CMS read
        self.cms_read = CMSRead(self.config)
        
        # CMS write (memory update)
        self.cms_write = CMSWrite(self.config)
        
        # HOPE core
        self.hope_core = HOPECore(self.config)
        
        # Output decoder
        self.decoder = OutputDecoder(self.config, self.output_dim)
    
    def forward_batch(
        self,
        x_obs: jnp.ndarray,
        a_prev: jnp.ndarray,
        r_t: jnp.ndarray,
        s_prev: jnp.ndarray,
        w_prev: jnp.ndarray,
        p_prev: jnp.ndarray,
        cms_memories: List[jnp.ndarray],
        cms_keys: List[jnp.ndarray],
        object_features: Optional[jnp.ndarray] = None,
        audio_features: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Forward pass for batched inputs.

        All inputs MUST have a leading batch dimension. This is the core
        computation method - use this when you know inputs are batched.

        Args:
            x_obs: Observation [B, obs_dim]
            a_prev: Previous action [B, action_dim]
            r_t: Reward [B, 1]
            s_prev: Previous fast state [B, d_s]
            w_prev: Previous wave state [B, d_w]
            p_prev: Previous particle state [B, d_p]
            cms_memories: List of [B, N_l, d_l] memory matrices
            cms_keys: List of [B, N_l, d_k] key matrices
            object_features: Optional [B, max_objects, object_dim]
            audio_features: Optional [B, num_frames, audio_dim] mel spectrogram

        Returns:
            y_t: Output [B, output_dim]
            info: Dictionary with intermediate states (all batched)
        """
        # 1. Encode
        e_t = self.encoder(x_obs, a_prev, r_t, object_features=object_features, audio_features=audio_features)

        # 2. CMS Read (retrieve context from memory)
        q_t, c_t, attn_weights = self.cms_read(cms_memories, cms_keys, s_prev, e_t)

        # 3. Core Dynamics
        s_t, w_t, p_t = self.hope_core(s_prev, w_prev, p_prev, e_t, c_t)

        # 4. CMS Write (update memory with new experience)
        new_cms_memories, new_cms_keys = self.cms_write(
            cms_memories, cms_keys, s_t, e_t, r_t
        )

        # 5. Decode
        y_t = self.decoder(s_t, c_t)

        info = {
            'query': q_t,
            'context': c_t,
            'attention_weights': attn_weights,
            'fast_state': s_t,
            'wave_state': w_t,
            'particle_state': p_t,
            'cms_memories': new_cms_memories,
            'cms_keys': new_cms_keys,
        }

        return y_t, info

    def forward_single(
        self,
        x_obs: jnp.ndarray,
        a_prev: jnp.ndarray,
        r_t: jnp.ndarray,
        s_prev: jnp.ndarray,
        w_prev: jnp.ndarray,
        p_prev: jnp.ndarray,
        cms_memories: List[jnp.ndarray],
        cms_keys: List[jnp.ndarray],
        object_features: Optional[jnp.ndarray] = None,
        audio_features: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Forward pass for a single (unbatched) input.

        Convenience method that handles batch dimension management.
        All inputs should be unbatched tensors (no leading batch dimension).

        Args:
            x_obs: Observation [obs_dim]
            a_prev: Previous action [action_dim]
            r_t: Reward (scalar or [1])
            s_prev: Previous fast state [d_s]
            w_prev: Previous wave state [d_w]
            p_prev: Previous particle state [d_p]
            cms_memories: List of [N_l, d_l] memory matrices
            cms_keys: List of [N_l, d_k] key matrices
            object_features: Optional [max_objects, object_dim]
            audio_features: Optional [num_frames, audio_dim] mel spectrogram

        Returns:
            y_t: Output [output_dim]
            info: Dictionary with intermediate states (all unbatched)
        """
        # Add batch dimensions
        x_obs_b = x_obs[None, :]
        a_prev_b = a_prev[None, :]
        r_t_b = r_t[None, None] if r_t.ndim == 0 else r_t[None, :]
        s_prev_b = s_prev[None, :]
        w_prev_b = w_prev[None, :]
        p_prev_b = p_prev[None, :]
        cms_memories_b = [m[None, :, :] for m in cms_memories]
        cms_keys_b = [k[None, :, :] for k in cms_keys]
        object_features_b = object_features[None, :, :] if object_features is not None else None
        audio_features_b = audio_features[None, :, :] if audio_features is not None else None

        # Run batched forward
        y_t, info = self.forward_batch(
            x_obs_b, a_prev_b, r_t_b,
            s_prev_b, w_prev_b, p_prev_b,
            cms_memories_b, cms_keys_b,
            object_features_b,
            audio_features_b
        )

        # Remove batch dimensions
        y_t = y_t[0]
        info = {
            k: v[0] if isinstance(v, jnp.ndarray) else
               [x[0] for x in v] if isinstance(v, list) else v
            for k, v in info.items()
        }

        return y_t, info

    def __call__(
        self,
        x_obs: jnp.ndarray,
        a_prev: jnp.ndarray,
        r_t: jnp.ndarray,
        s_prev: jnp.ndarray,
        w_prev: jnp.ndarray,
        p_prev: jnp.ndarray,
        cms_memories: List[jnp.ndarray],
        cms_keys: List[jnp.ndarray],
        object_features: Optional[jnp.ndarray] = None,
        audio_features: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Unified forward pass with automatic batch detection.

        Accepts both batched and unbatched inputs. For explicit control
        and better performance, use forward_single() or forward_batch() directly.

        Args:
            x_obs: Observation [obs_dim] or [B, obs_dim]
            a_prev: Previous action [action_dim] or [B, action_dim]
            r_t: Reward (scalar, [1], or [B, 1])
            s_prev: Previous fast state [d_s] or [B, d_s]
            w_prev: Previous wave state [d_w] or [B, d_w]
            p_prev: Previous particle state [d_p] or [B, d_p]
            cms_memories: List of CMS memory matrices per level
            cms_keys: List of CMS key matrices per level
            object_features: Optional object features
            audio_features: Optional audio features [num_frames, audio_dim] or [B, num_frames, audio_dim]

        Returns:
            y_t: Output [output_dim] or [B, output_dim]
            info: Dictionary with intermediate states and attention weights
        """
        # Use centralized normalization from batch_utils
        (
            x_obs, a_prev, r_t,
            s_prev, w_prev, p_prev,
            cms_memories, cms_keys,
            object_features, was_unbatched
        ) = normalize_inputs(
            x_obs, a_prev, r_t,
            s_prev, w_prev, p_prev,
            cms_memories, cms_keys,
            object_features
        )

        # Normalize audio features if provided
        if audio_features is not None and audio_features.ndim == 2:
            audio_features = audio_features[None, :, :]

        # Run batched forward
        y_t, info = self.forward_batch(
            x_obs, a_prev, r_t,
            s_prev, w_prev, p_prev,
            cms_memories, cms_keys,
            object_features,
            audio_features
        )

        # Denormalize outputs if needed
        y_t, info = denormalize_outputs(y_t, info, was_unbatched)

        return y_t, info


def make_core_model(
    rng_key: jax.random.PRNGKey,
    obs_dim: int,
    action_dim: int,
    output_dim: int,
    config: Optional[CoreModelConfig] = None,
) -> Tuple[CoreModel, Dict[str, Any]]:
    """
    Create and initialize a CoreModel.

    Args:
        rng_key: JAX random key for initialization
        obs_dim: Observation dimension
        action_dim: Action dimension
        output_dim: Output dimension
        config: Model configuration (defaults to pi5_optimized)

    Returns:
        model: CoreModel instance
        params: Initialized parameters
    """
    if config is None:
        config = CoreModelConfig.pi5_optimized()

    model = CoreModel(config, obs_dim, action_dim, output_dim)

    # Create dummy inputs for initialization
    batch_size = 1
    dummy_obs = jnp.zeros((batch_size, obs_dim))
    dummy_action = jnp.zeros((batch_size, action_dim))
    dummy_reward = jnp.zeros((batch_size, 1))
    dummy_s = jnp.zeros((batch_size, config.d_s))
    dummy_w = jnp.zeros((batch_size, config.d_w))
    dummy_p = jnp.zeros((batch_size, config.d_p))
    dummy_cms_memories = [
        jnp.zeros((batch_size, size, dim)) for size, dim in zip(config.cms_sizes, config.cms_dims)
    ]
    dummy_cms_keys = [
        jnp.zeros((batch_size, size, config.d_k)) for size in config.cms_sizes
    ]

    # Optional features
    dummy_object_features = (
        jnp.zeros((batch_size, config.max_objects, config.object_dim))
        if config.use_object_features else None
    )
    dummy_audio_features = (
        jnp.zeros((batch_size, config.audio_max_frames, config.audio_dim))
        if config.use_audio_features else None
    )

    # Initialize parameters
    params = model.init(
        rng_key,
        dummy_obs,
        dummy_action,
        dummy_reward,
        dummy_s,
        dummy_w,
        dummy_p,
        dummy_cms_memories,
        dummy_cms_keys,
        object_features=dummy_object_features,
        audio_features=dummy_audio_features,
    )

    return model, params

