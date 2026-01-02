# JAX Models: WaveCore Seed Implementation

**Status:** Seed Phase (Bootstrap)  
**Framework:** JAX/Flax  
**Parameters:** 172,202

This directory contains the JAX/Flax implementation of the WaveCore seed model, which will evolve into the HOPE Agent Manager for production.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WAVECORE SEED MODEL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT ENCODER                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ x_obs ──► Dense ──► LayerNorm ──► ReLU ──► Dense ──► e_obs              ││
│  │ a_prev ─► Dense ──► e_action                                            ││
│  │ r_t ────► Dense ──► e_reward                                            ││
│  │                                                                          ││
│  │ e_t = Concat(e_obs, e_action, e_reward) ∈ ℝ^{d_e}                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                   │                                         │
│                                   ▼                                         │
│  CMS READ                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Query: q_t = W_q · [s_{t-1}; e_t]                                       ││
│  │                                                                          ││
│  │ Per-level attention:                                                     ││
│  │   α^(ℓ) = softmax(K^(ℓ) · q_t / √d_k)                                   ││
│  │   c^(ℓ) = α^(ℓ) · M^(ℓ)                                                 ││
│  │                                                                          ││
│  │ Hierarchical mixing:                                                     ││
│  │   c_t = Σ_ℓ β_ℓ · c^(ℓ)                                                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                   │                                         │
│                                   ▼                                         │
│  HOPE CORE (Fast/Wave/Particle)                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  FAST STATE (Reactive)           WAVE STATE (SSM)                       ││
│  │  ┌─────────────────────────┐    ┌─────────────────────────────────┐    ││
│  │  │ s_t = σ(W_s · [s_{t-1}; │    │ Mamba-like SSM:                  │    ││
│  │  │         e_t; c_t])      │    │ Δ = softplus(Linear(e_t))        │    ││
│  │  │                         │    │ B = Linear(e_t)                  │    ││
│  │  │ τ = 10ms (reflexes)     │    │ C = Linear(e_t)                  │    ││
│  │  └─────────────────────────┘    │                                  │    ││
│  │                                  │ w_t = Discretize(A, B, Δ) @ w_{t-1} ││
│  │  PARTICLE STATE (Local)         │       + B @ e_t                  │    ││
│  │  ┌─────────────────────────┐    │                                  │    ││
│  │  │ p_t = φ(W_p · [p_{t-1}; │    │ Long-range temporal memory       │    ││
│  │  │         e_t; s_t])      │    └─────────────────────────────────┘    ││
│  │  │                         │                                          ││
│  │  │ Nonlinear dynamics      │                                          ││
│  │  └─────────────────────────┘                                          ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                   │                                         │
│                                   ▼                                         │
│  OUTPUT DECODER                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ y_t = W_y · [s_t; c_t]                                                  ││
│  │                                                                          ││
│  │ Outputs: Action prediction, Value estimate, World model prediction     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `config.py` | `CoreModelConfig` dataclass with all hyperparameters |
| `core_model.py` | Main `CoreModel` class with encoder/CMS/core/decoder |
| `mamba_ssm.py` | `MambaLikeWave` selective state space model |
| `cms_jax.py` | JAX CMS memory (hierarchical read/write) |
| `export/inference_router.py` | Routes inference to JAX/Hailo/CPU backend |

---

## Configuration

```python
from continuonbrain.jax_models.config import CoreModelConfig

# Pi5-optimized (default for seed training)
config = CoreModelConfig(
    # State dimensions
    d_s=64,      # Fast state
    d_w=64,      # Wave state
    d_p=32,      # Particle state
    d_e=64,      # Encoded input
    d_k=32,      # Key dimension
    d_c=64,      # Context dimension
    
    # CMS hierarchy
    num_levels=3,
    cms_sizes=[64, 128, 256],    # Slots per level
    cms_dims=[128, 256, 512],    # Dimension per level
    cms_decays=[0.1, 0.05, 0.01], # Decay rates
    
    # Mamba SSM
    use_mamba_wave=True,
    mamba_state_dim=1,
    mamba_dt_min=1e-4,
    mamba_dt_scale=1.0,
    
    # Training
    learning_rate=1e-3,
    gradient_clip=10.0,
)

# Cloud TPU (larger model)
config = CoreModelConfig.cloud_optimized()
```

---

## Usage

### Initialization

```python
from continuonbrain.jax_models.core_model import make_core_model
import jax

rng = jax.random.PRNGKey(42)

model, variables = make_core_model(
    rng_key=rng,
    obs_dim=64,
    action_dim=7,
    output_dim=7,
    config=config,
)

params = variables['params']
```

### Forward Pass

```python
import jax.numpy as jnp

# Inputs
batch = 2
x_obs = jnp.zeros((batch, 64))      # Observation
a_prev = jnp.zeros((batch, 7))       # Previous action
r_t = jnp.zeros((batch, 1))          # Reward

# States
s_prev = jnp.zeros((batch, config.d_s))
w_prev = jnp.zeros((batch, config.d_w))
p_prev = jnp.zeros((batch, config.d_p))

# CMS memory
cms_memories = [jnp.zeros((batch, sz, dim)) 
                for sz, dim in zip(config.cms_sizes, config.cms_dims)]
cms_keys = [jnp.zeros((batch, sz, config.d_k)) 
            for sz in config.cms_sizes]

# Forward
y, info = model.apply(
    variables,
    x_obs, a_prev, r_t,
    s_prev, w_prev, p_prev,
    cms_memories, cms_keys,
)

# info contains intermediate states
print(f"Output: {y.shape}")                    # (2, 7)
print(f"Fast state: {info['fast_state'].shape}")  # (2, 64)
print(f"Wave state: {info['wave_state'].shape}")  # (2, 64)
```

### Training

```python
import optax

optimizer = optax.adam(config.learning_rate)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(params):
        y, _ = model.apply({'params': params}, *batch)
        return jnp.mean((y - target) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Gradient clipping
    grads = jax.tree_map(
        lambda g: jnp.clip(g, -config.gradient_clip, config.gradient_clip),
        grads
    )
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss
```

---

## Mamba SSM (Wave State)

The wave state uses a **Selective State Space Model** inspired by Mamba:

```
Standard SSM:
  h'(t) = A h(t) + B x(t)
  y(t) = C h(t) + D x(t)

Mamba Selective SSM:
  Δ = softplus(Linear(x))     # Input-dependent discretization
  B = Linear(x)               # Input-dependent input matrix
  C = Linear(x)               # Input-dependent output matrix
  
  Ā = exp(Δ A)
  B̄ = (Δ A)^{-1} (Ā - I) Δ B
  
  h_t = Ā h_{t-1} + B̄ x_t
  y_t = C h_t
```

### Key Properties

| Property | Value |
|----------|-------|
| **Complexity** | O(n) vs O(n²) for attention |
| **Memory** | Constant (no KV cache) |
| **Long-range** | SSM captures long dependencies |
| **Selective** | Input-dependent dynamics |

---

## CMS Memory

Hierarchical memory with **content-addressable read** and **salience-gated write**.

### Read Operation

```python
def cms_read(query, memories, keys):
    """
    query: [B, d_k]
    memories: [[B, N_0, d_0], [B, N_1, d_1], [B, N_2, d_2]]
    keys: [[B, N_0, d_k], [B, N_1, d_k], [B, N_2, d_k]]
    
    Returns: context [B, d_c], attentions [[B, N_0], ...]
    """
    contexts = []
    for M, K in zip(memories, keys):
        # Attention scores
        scores = einsum('bd,bnd->bn', query, K) / sqrt(d_k)
        attn = softmax(scores)
        
        # Retrieve
        c = einsum('bn,bnd->bd', attn, M)
        contexts.append(c)
    
    # Mix across levels
    mixed = hierarchical_mix(contexts)
    return mixed
```

### Write Operation

```python
def cms_write(memories, keys, content, key, salience, decay):
    """Write new memory with decay and salience gating."""
    for level, (M, K) in enumerate(zip(memories, keys)):
        # Decay existing
        M = M * decay[level]
        K = K * decay[level]
        
        if salience > threshold:
            # Find least salient slot
            idx = argmin(norm(M, axis=-1))
            
            # Write
            M[idx] = content
            K[idx] = key
    
    return memories, keys
```

---

## Checkpoints

### Seed Checkpoint Location

```
/opt/continuonos/brain/model/adapters/candidate/core_model_seed/
├── wavecore_seed.npz     # JAX weights
├── manifest.json         # Metadata
└── proof_of_learning.json # Eval results
```

### Manifest Schema

```json
{
  "version": "1.0.0",
  "phase": "seed",
  "timestamp": "2026-01-02T10:10:31.775877",
  "model_type": "wavecore_seed",
  "param_count": 172202,
  "scaffold": "google/gemma-3n-E2B-it (TEMPORARY)",
  "embedding": "google/embeddinggemma-300m (RETAINED)",
  "target": "hope_wavecore (non-transformers)",
  "training": {
    "final_loss": 0.0,
    "rlds_episodes": 4219
  }
}
```

---

## Next Steps: Production Evolution

### Transition Criteria

1. HOPE Eval ≥ 80%
2. Tool Router Accuracy ≥ 70%
3. WaveCore Stability (Lyapunov λ < 0.1)
4. Memory Consolidation validated

### Post-Seed Changes

- Gemma scaffold **deprecated**
- WaveCore becomes **primary LLM**
- Cloud TPU slow loop for **updates**
- OTA bundles for **distribution**

See [seed-to-hope-evolution.md](../../docs/seed-to-hope-evolution.md) for details.

