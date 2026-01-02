# HOPE Implementation: Hierarchical Orchestrated Predictive Engine

**Status:** Seed Phase  
**Framework:** PyTorch (CMS), JAX (Core)

This directory contains the Python/PyTorch implementation of HOPE components, particularly the Continuous Memory System (CMS).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HOPE ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         AGENT MANAGER                                   ││
│  │  Goal parsing │ Task decomposition │ Tool routing │ Response synthesis ││
│  └───────────────────────────────────┬─────────────────────────────────────┘│
│                                      │                                       │
│         ┌────────────────────────────┼────────────────────────────┐          │
│         │                            │                            │          │
│         ▼                            ▼                            ▼          │
│  ┌────────────┐              ┌────────────┐              ┌────────────┐     │
│  │ FAST LOOP  │              │ MID LOOP   │              │ SLOW LOOP  │     │
│  │ τ = 10ms   │◄────────────►│ τ = 100ms  │◄────────────►│ τ = 1s     │     │
│  │            │              │            │              │            │     │
│  │ Reflexes   │              │ Attention  │              │ Planning   │     │
│  │ Safety     │              │ Context    │              │ Goals      │     │
│  │ Motor      │              │ Integration│              │ Learning   │     │
│  └────────────┘              └────────────┘              └────────────┘     │
│         │                            │                            │          │
│         └────────────────────────────┼────────────────────────────┘          │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    CONTINUOUS MEMORY SYSTEM (CMS)                       ││
│  │                                                                          ││
│  │  Level 0: EPISODIC     Level 1: WORKING      Level 2: SEMANTIC         ││
│  │  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐         ││
│  │  │ τ = 100ms     │     │ τ = 10s       │     │ τ = ∞         │         ││
│  │  │ decay = 0.9   │────►│ decay = 0.99  │────►│ decay = 0.999 │         ││
│  │  │ 64 slots      │     │ 128 slots     │     │ 256 slots     │         ││
│  │  │               │     │               │     │               │         ││
│  │  │ Raw sensory   │     │ Task context  │     │ Knowledge     │         ││
│  │  │ Motor cmds    │     │ Active goals  │     │ Skills        │         ││
│  │  └───────────────┘     └───────────────┘     └───────────────┘         ││
│  │                                                                          ││
│  │  Operations:                                                             ││
│  │  • READ: Content-addressable attention retrieval                        ││
│  │  • WRITE: Salience-gated memory storage                                 ││
│  │  • CONSOLIDATE: Sleep-like pattern compression                          ││
│  │  • FORGET: Decay-based memory pruning                                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `brain.py` | `HOPEBrain` main class with column architecture |
| `cms.py` | Continuous Memory System (PyTorch) |
| `state.py` | `CMSMemory`, `MemoryLevel` state dataclasses |
| `config.py` | HOPE configuration |

---

## Continuous Memory System (CMS)

### Mathematical Specification

```
Memory State: M^(ℓ) ∈ ℝ^{N_ℓ × d_ℓ} for level ℓ ∈ {0, 1, ..., L-1}

READ Operation:
  q_t = Q_ψ(s_{t-1}, e_t)                    # Generate query
  α_t^(ℓ) = softmax(K^(ℓ) q_t / √d_k)       # Per-level attention
  c_t^(ℓ) = Σ_i α_{t,i}^(ℓ) M_i^(ℓ)         # Context retrieval
  c_t = Σ_ℓ β_t^(ℓ) U^(ℓ) c_t^(ℓ)           # Hierarchical mixing

WRITE Operation:
  w_t = W_φ(s_t, e_t, r_t)                   # Write weight
  v_t = V_θ(s_t, e_t)                        # Value to write
  M_t^(ℓ) = (1 - α_ℓ · w_t) ⊙ M_{t-1}^(ℓ) + α_ℓ · w_t · v_t^T

DECAY:
  M_t^(ℓ) = γ_ℓ · M_{t-1}^(ℓ)                # Exponential decay
  where γ_ℓ ∈ [0.9, 0.99, 0.999] for ℓ ∈ {0, 1, 2}

CONSOLIDATION:
  Pattern extraction: P = SVD(M^(ℓ), k=r)
  Schema compression: M^(ℓ+1) ← merge(M^(ℓ+1), compress(P))
```

### Implementation

```python
from continuonbrain.hope_impl.cms import CMSRead, CMSWrite

# Initialize
cms_read = CMSRead(
    d_s=64,         # Fast state dimension
    d_e=64,         # Encoded input dimension
    d_k=32,         # Key dimension
    d_c=64,         # Context output dimension
    num_levels=3,
    cms_dims=[32, 64, 128],
)

cms_write = CMSWrite(
    d_s=64,
    d_e=64,
    num_levels=3,
    cms_dims=[32, 64, 128],
)

# Read
query, context, attentions = cms_read(
    cms_memory,     # Current memory state
    s_prev,         # Previous fast state
    e_t,            # Encoded input
)

# Write
new_memory = cms_write(
    cms_memory,
    s_t,
    e_t,
    r_t,            # Reward (salience signal)
)
```

---

## HOPE Brain

### Column Architecture

Each HOPE column operates at a different timescale:

```python
from continuonbrain.hope_impl.brain import HOPEBrain, HOPEColumn

brain = HOPEBrain(
    embed_dim=64,
    num_columns=3,
    cms_levels=3,
)

# Process input through all columns
outputs = brain(
    observation=obs,
    prev_action=a_prev,
    reward=r,
    cms_state=cms,
)

# Each column contributes:
# - Fast: Reflexive responses (safety, motor)
# - Mid: Contextualized responses (attention, integration)
# - Slow: Goal-directed responses (planning, learning)
```

### Consolidation

```python
# Background consolidation (sleep-like)
brain.consolidate_memory(
    fast_to_mid=True,   # Compress episodic → working
    mid_to_slow=True,   # Compress working → semantic
    threshold=0.5,      # Salience threshold
)
```

---

## Memory Levels

### Level 0: Episodic (Fast)

| Property | Value |
|----------|-------|
| **Timescale** | ~100ms |
| **Decay** | 0.9 |
| **Slots** | 64 |
| **Content** | Raw sensory, motor commands, immediate context |
| **Update** | Every perception cycle |

### Level 1: Working (Mid)

| Property | Value |
|----------|-------|
| **Timescale** | ~10s |
| **Decay** | 0.99 |
| **Slots** | 128 |
| **Content** | Current task state, active goals, recent events |
| **Update** | Event-driven (significant changes) |

### Level 2: Semantic (Slow)

| Property | Value |
|----------|-------|
| **Timescale** | Persistent |
| **Decay** | 0.999 |
| **Slots** | 256 |
| **Content** | Learned skills, factual knowledge, identity |
| **Update** | Background consolidation |

---

## Integration with WaveCore

The PyTorch CMS integrates with the JAX WaveCore model:

```python
# HOPE CMS (PyTorch) for memory operations
cms = CMSMemory(num_levels=3, dims=[32, 64, 128])

# WaveCore (JAX) for state dynamics
from continuonbrain.jax_models.core_model import CoreModel

# Bridge: Convert PyTorch tensors to JAX arrays
import jax.numpy as jnp

def torch_to_jax(t):
    return jnp.array(t.detach().numpy())

def jax_to_torch(a):
    return torch.from_numpy(np.array(a))

# Combined inference
def hope_forward(obs, cms_state):
    # 1. CMS read (PyTorch)
    context = cms_read(cms_state, ...)
    
    # 2. WaveCore dynamics (JAX)
    y, info = wavecore.apply(params, torch_to_jax(obs), ...)
    
    # 3. CMS write (PyTorch)
    new_cms = cms_write(cms_state, jax_to_torch(info['fast_state']), ...)
    
    return y, new_cms
```

---

## Formal Specification

See [CMS_FORMAL_SPEC.md](../docs/CMS_FORMAL_SPEC.md) for the complete mathematical specification including:

- Convergence guarantees
- Stability analysis
- Capacity bounds
- Forgetting dynamics

---

## Next Steps

### Seed Phase Goals

1. Validate CMS read/write correctness
2. Verify decay dynamics
3. Test consolidation patterns
4. Integrate with WaveCore training

### Production Migration

- Port CMS to JAX for unified inference
- Enable TPU-accelerated memory operations
- Optimize for edge deployment

See [seed-to-hope-evolution.md](../../docs/seed-to-hope-evolution.md) for the full transition plan.
