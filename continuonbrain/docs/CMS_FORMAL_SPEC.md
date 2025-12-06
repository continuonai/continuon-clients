# CMS Formal Memory Update Rule

**Continuous Memory System (CMS): Mathematical Specification**

This document provides the complete mathematical formulation of the CMS memory update rule, including derivations, stability analysis, and computational complexity.

---

## 1. Overview

The **Continuous Memory System (CMS)** is a hierarchical, content-addressable memory structure with controlled decay and write operations. It implements a **discrete jump map** that updates memory at event times while maintaining:

- **Bounded memory growth** through exponential decay
- **Content-addressable retrieval** via attention mechanisms
- **Hierarchical organization** across multiple timescales
- **Stability guarantees** via Lyapunov analysis

---

## 2. State Representation

### 2.1 Memory Hierarchy

The CMS consists of **L+1 levels** (indexed ℓ = 0, 1, ..., L), ordered from fastest (episodic) to slowest (semantic):

**Level ℓ state:**
```
M_t^(ℓ) ∈ ℝ^(N_ℓ × d_ℓ)    Memory matrix (N_ℓ slots, d_ℓ dimensions)
K_t^(ℓ) ∈ ℝ^(N_ℓ × d_k)    Key matrix (for content addressing)
d_ℓ ∈ (0,1)                Decay coefficient
```

**Hierarchical properties:**
- **Timescale separation**: d_0 > d_1 > ... > d_L (faster levels decay more)
- **Capacity scaling**: N_0 ≤ N_1 ≤ ... ≤ N_L (slower levels have more slots)
- **Dimension scaling**: d_0 ≤ d_1 ≤ ... ≤ d_L (slower levels have richer representations)

### 2.2 Full CMS State

```
M_t = {M_t^(0), M_t^(1), ..., M_t^(L)}
K_t = {K_t^(0), K_t^(1), ..., K_t^(L)}
```

---

## 3. CMS Read Operation (Query)

### 3.1 Query Generation

Given fast state s_{t-1} and encoded input e_t, compute query vector:

```
q_t = Q_ψ(s_{t-1}, e_t) ∈ ℝ^(d_k)
```

where Q_ψ is a learned query network (typically MLP or linear projection).

### 3.2 Per-Level Attention

For each level ℓ, compute attention weights using scaled dot-product:

```
α_t^(ℓ) = softmax(K_t^(ℓ) q_t / √d_k) ∈ ℝ^(N_ℓ)
```

**Properties:**
- Σ_i α_{t,i}^(ℓ) = 1 (normalized distribution)
- α_{t,i}^(ℓ) ≥ 0 (non-negative)
- Temperature scaling via √d_k prevents saturation

### 3.3 Context Retrieval

Retrieve context from each level via weighted sum:

```
c_t^(ℓ) = Σ_{i=1}^{N_ℓ} α_{t,i}^(ℓ) M_{t,i}^(ℓ) ∈ ℝ^(d_ℓ)
```

Equivalently in matrix form:
```
c_t^(ℓ) = (α_t^(ℓ))^T M_t^(ℓ)
```

### 3.4 Hierarchical Mixing

Combine contexts across levels with learned mixing weights:

```
β_t = softmax(W_β [c_t^(0) || c_t^(1) || ... || c_t^(L)]) ∈ ℝ^(L+1)

c_t = Σ_{ℓ=0}^L β_t^(ℓ) U^(ℓ) c_t^(ℓ) ∈ ℝ^(d_c)
```

where:
- W_β ∈ ℝ^((L+1) × Σ_ℓ d_ℓ) is learned mixing matrix
- U^(ℓ) ∈ ℝ^(d_c × d_ℓ) are per-level projection matrices
- || denotes concatenation

**Output:** Mixed context c_t ∈ ℝ^(d_c) summarizing all memory levels

---

## 4. CMS Write Operation (Jump Map)

### 4.1 Event Signal Computation

For each level ℓ, compute event signal from lower-level activity:

```
z_t^(ℓ) = Z_ℓ(s_t, c_t^(ℓ-1), e_t) ∈ ℝ^(d_z)
```

where:
- For ℓ=0: z_t^(0) = Z_0(s_t, e_t) (no lower level)
- For ℓ>0: uses context from level ℓ-1 as input

**Interpretation:** z_t^(ℓ) represents "what happened" that should be written to level ℓ

### 4.2 Write Gate

Compute write gate controlling write strength:

```
g_t^(ℓ) = σ(W_g^(ℓ) z_t^(ℓ)) ∈ [0,1]
```

where σ is sigmoid activation.

**Interpretation:** 
- g_t^(ℓ) ≈ 0: minimal write (preserve existing memory)
- g_t^(ℓ) ≈ 1: strong write (update memory significantly)

### 4.3 Write Value and Key

Compute what to write and how to address it:

```
v_t^(ℓ) = V_ℓ(z_t^(ℓ)) ∈ ℝ^(d_ℓ)        Write value
k_t^(ℓ) = K_ℓ(z_t^(ℓ)) ∈ ℝ^(d_k)        Write key (optional)
```

### 4.4 Write Addressing

Compute write weights (where to write):

**Option 1: Content-based addressing**
```
α̃_t^(ℓ) = softmax(K_{t-1}^(ℓ) k_t^(ℓ) / √d_k) ∈ ℝ^(N_ℓ)
```

**Option 2: Least-recently-used (LRU) addressing**
```
α̃_t^(ℓ) = one_hot(argmin_i usage_t^(ℓ)[i])
```

**Option 3: Hybrid (content + LRU)**
```
α̃_t^(ℓ) = λ · content_weights + (1-λ) · lru_weights
```

### 4.5 Memory Update Rule

**Core update equation:**

```
M_t^(ℓ) = (1 - d_ℓ) M_{t-1}^(ℓ) + g_t^(ℓ) (α̃_t^(ℓ) ⊗ v_t^(ℓ))
```

where ⊗ denotes outer product: (α̃_t^(ℓ) ⊗ v_t^(ℓ)) ∈ ℝ^(N_ℓ × d_ℓ)

**Element-wise form:**
```
M_{t,i}^(ℓ) = (1 - d_ℓ) M_{t-1,i}^(ℓ) + g_t^(ℓ) α̃_{t,i}^(ℓ) v_t^(ℓ)
```

**Key update (if using content-based addressing):**
```
K_t^(ℓ) = (1 - d_ℓ) K_{t-1}^(ℓ) + g_t^(ℓ) (α̃_t^(ℓ) ⊗ k_t^(ℓ))
```

---

## 5. Mathematical Properties

### 5.1 Bounded Memory Growth

**Theorem 1 (Bounded Norm):** If ||v_t^(ℓ)|| ≤ V_max and g_t^(ℓ) ≤ 1, then:

```
||M_t^(ℓ)||_F ≤ max(||M_0^(ℓ)||_F, V_max√N_ℓ / d_ℓ)
```

**Proof:**

Taking Frobenius norm of the update equation:

```
||M_t^(ℓ)||_F² = ||(1-d_ℓ)M_{t-1}^(ℓ) + g_t^(ℓ)(α̃_t^(ℓ) ⊗ v_t^(ℓ))||_F²
```

By triangle inequality and properties of outer product:

```
≤ (1-d_ℓ)²||M_{t-1}^(ℓ)||_F² + 2(1-d_ℓ)g_t^(ℓ)||M_{t-1}^(ℓ)||_F||v_t^(ℓ)|| + g_t^(ℓ)²||v_t^(ℓ)||²
```

Since ||α̃_t^(ℓ)||₂ = 1 (normalized), the outer product has norm ||v_t^(ℓ)||.

At equilibrium (||M_t^(ℓ)||_F = ||M_{t-1}^(ℓ)||_F = M̄):

```
M̄² = (1-d_ℓ)²M̄² + 2(1-d_ℓ)g_t^(ℓ)M̄V_max + g_t^(ℓ)²V_max²
```

Solving for M̄:

```
M̄ ≤ V_max√N_ℓ / d_ℓ
```

**Interpretation:** Decay prevents unbounded growth; memory saturates at a level proportional to write magnitude and inversely proportional to decay rate.

### 5.2 Decay Dynamics

**Theorem 2 (Exponential Decay):** In the absence of writes (g_t^(ℓ) = 0), memory decays exponentially:

```
M_t^(ℓ) = (1-d_ℓ)^t M_0^(ℓ)
```

**Half-life:** Time for memory to decay to 50% of initial value:

```
t_{1/2} = log(0.5) / log(1-d_ℓ) ≈ 0.693 / d_ℓ  (for small d_ℓ)
```

**Example timescales:**
- d_0 = 0.1 → t_{1/2} ≈ 7 steps (episodic)
- d_1 = 0.05 → t_{1/2} ≈ 14 steps (working memory)
- d_2 = 0.01 → t_{1/2} ≈ 69 steps (semantic)

### 5.3 Write Saturation

**Theorem 3 (Saturation Bound):** For constant writes with g_t^(ℓ) = g and v_t^(ℓ) = v:

```
lim_{t→∞} M_t^(ℓ) = (g/d_ℓ)(α̃^(ℓ) ⊗ v)
```

**Proof:** At equilibrium M_t^(ℓ) = M_{t-1}^(ℓ) = M̄^(ℓ):

```
M̄^(ℓ) = (1-d_ℓ)M̄^(ℓ) + g(α̃^(ℓ) ⊗ v)
d_ℓ M̄^(ℓ) = g(α̃^(ℓ) ⊗ v)
M̄^(ℓ) = (g/d_ℓ)(α̃^(ℓ) ⊗ v)
```

**Interpretation:** Equilibrium memory is proportional to write strength and inversely proportional to decay rate.

---

## 6. Lyapunov Stability Analysis

### 6.1 Memory Energy Function

Define Lyapunov function for CMS:

```
V_mem(M_t) = Σ_{ℓ=0}^L λ_ℓ ||M_t^(ℓ)||_F²
```

where λ_ℓ > 0 are level-specific weights.

### 6.2 Energy Dissipation

**Theorem 4 (Dissipative Dynamics):** The change in memory energy satisfies:

```
ΔV_mem = V_mem(M_t) - V_mem(M_{t-1}) ≤ -γ V_mem(M_{t-1}) + C
```

for some γ > 0 (dissipation rate) and C ≥ 0 (write energy bound).

**Proof:**

For each level:

```
||M_t^(ℓ)||_F² = ||(1-d_ℓ)M_{t-1}^(ℓ) + g_t^(ℓ)(α̃_t^(ℓ) ⊗ v_t^(ℓ))||_F²
                ≤ (1-d_ℓ)²||M_{t-1}^(ℓ)||_F² + g_t^(ℓ)²||v_t^(ℓ)||²
                ≤ (1-2d_ℓ)||M_{t-1}^(ℓ)||_F² + V_max²
```

Summing over levels:

```
V_mem(M_t) ≤ (1-2d_min)V_mem(M_{t-1}) + (L+1)V_max²
```

where d_min = min_ℓ d_ℓ.

Setting γ = 2d_min and C = (L+1)V_max² gives the result.

**Interpretation:** Memory energy is dissipative (decays) with bounded input from writes, ensuring stability.

---

## 7. Connection to Existing Memory Architectures

### 7.1 Neural Turing Machines (NTM)

**Similarities:**
- Content-based addressing via attention
- Read/write operations
- External memory matrix

**Differences:**
- CMS has **exponential decay** (NTM memory persists)
- CMS has **hierarchical levels** (NTM has single memory)
- CMS write is **additive with decay** (NTM uses erase+add gates)

**CMS update vs NTM update:**

```
NTM:  M_t = M_{t-1} ⊙ (1 - w_t ⊗ e_t) + w_t ⊗ a_t
CMS:  M_t^(ℓ) = (1-d_ℓ)M_{t-1}^(ℓ) + g_t^(ℓ)(α̃_t^(ℓ) ⊗ v_t^(ℓ))
```

### 7.2 Differentiable Neural Computer (DNC)

**Similarities:**
- Content and location-based addressing
- Temporal memory links
- Usage tracking

**Differences:**
- CMS uses **decay** instead of explicit usage tracking
- CMS has **hierarchical timescales** (DNC has single memory)
- CMS is **simpler** (fewer addressing mechanisms)

### 7.3 Transformer Memory

**Similarities:**
- Attention-based retrieval
- Key-value structure

**Differences:**
- CMS has **bounded memory** via decay (Transformer context grows)
- CMS has **write operations** (Transformer is read-only during inference)
- CMS has **hierarchical organization** (Transformer has flat context)

---

## 8. Computational Complexity

### 8.1 CMS Read

**Per-level attention:**
```
α_t^(ℓ) = softmax(K_t^(ℓ) q_t / √d_k)
```

- Matrix-vector multiply: O(N_ℓ d_k)
- Softmax: O(N_ℓ)
- **Total per level:** O(N_ℓ d_k)

**Context retrieval:**
```
c_t^(ℓ) = (α_t^(ℓ))^T M_t^(ℓ)
```

- Weighted sum: O(N_ℓ d_ℓ)

**Hierarchical mixing:**
```
c_t = Σ_ℓ β_t^(ℓ) U^(ℓ) c_t^(ℓ)
```

- Per-level projection: O(d_c d_ℓ)
- Mixing: O(L d_c)
- **Total:** O(L d_c max_ℓ d_ℓ)

**Total CMS Read:** O(Σ_ℓ N_ℓ(d_k + d_ℓ) + L d_c max_ℓ d_ℓ)

### 8.2 CMS Write

**Event signal, gate, value, key:**
- Neural network forward passes: O(d_z²) each (assuming MLP)

**Write addressing:**
- Content-based: O(N_ℓ d_k) (same as read attention)

**Memory update:**
```
M_t^(ℓ) = (1-d_ℓ)M_{t-1}^(ℓ) + g_t^(ℓ)(α̃_t^(ℓ) ⊗ v_t^(ℓ))
```

- Decay: O(N_ℓ d_ℓ)
- Outer product + add: O(N_ℓ d_ℓ)
- **Total per level:** O(N_ℓ d_ℓ)

**Total CMS Write:** O(Σ_ℓ N_ℓ(d_k + d_ℓ) + L d_z²)

### 8.3 Memory Footprint

**Storage per level:**
```
Memory: N_ℓ × d_ℓ floats
Keys:   N_ℓ × d_k floats
```

**Total storage:**
```
Σ_{ℓ=0}^L N_ℓ(d_ℓ + d_k) floats
```

**Example (3 levels, FP32):**
- Level 0: 64 × (128 + 64) = 12,288 floats = 48 KB
- Level 1: 128 × (256 + 64) = 40,960 floats = 160 KB
- Level 2: 256 × (512 + 64) = 147,456 floats = 576 KB
- **Total:** ~784 KB

**Raspberry Pi 5 feasibility:** Easily fits in L2 cache (2MB), excellent for edge deployment.

---

## 9. Implementation Considerations

### 9.1 Numerical Stability

**Issue:** Decay can cause vanishing gradients during backpropagation.

**Solutions:**
1. **Gradient clipping:** Clip gradients to prevent explosion
2. **Residual connections:** Add skip connections around CMS
3. **Layer normalization:** Normalize memory before/after updates
4. **Careful initialization:** Initialize decay rates conservatively (d_ℓ ∈ [0.01, 0.1])

### 9.2 Sparse Writes

**Optimization:** Only write to CMS when g_t^(ℓ) exceeds threshold:

```python
if g_t[ℓ] > threshold:  # e.g., threshold = 0.1
    M_t[ℓ] = (1 - d[ℓ]) * M_{t-1}[ℓ] + g_t[ℓ] * outer(α̃_t[ℓ], v_t[ℓ])
else:
    M_t[ℓ] = (1 - d[ℓ]) * M_{t-1}[ℓ]  # Decay only
```

**Benefit:** Reduces computation when writes are weak.

### 9.3 Quantization

**Memory quantization:**
- Store M_t^(ℓ) in INT8 or FP16
- Dequantize during read operations
- Quantize after write operations

**Benefit:** 2-4× memory reduction, critical for Pi5 deployment.

### 9.4 Batching

**Challenge:** CMS state is sequential (M_t depends on M_{t-1}).

**Solution:** Batch across independent sequences, not time:

```python
# Batch dimension B, time dimension T
M_t = torch.zeros(B, N_ℓ, d_ℓ)  # Separate memory per batch item

for t in range(T):
    M_t = cms_write(M_{t-1}, ...)  # Vectorized across batch
```

---

## 10. Hyperparameter Tuning Guide

### 10.1 Decay Rates

**Principle:** Exponential spacing across levels

```python
d_0 = 0.1   # Episodic: ~7 step half-life
d_1 = 0.05  # Working:  ~14 step half-life
d_2 = 0.01  # Semantic: ~69 step half-life
```

**Tuning:**
- Increase d_ℓ if memory saturates too slowly
- Decrease d_ℓ if memory forgets too quickly

### 10.2 Memory Sizes

**Principle:** Larger capacity for slower levels

```python
N_0 = 64    # Episodic
N_1 = 128   # Working
N_2 = 256   # Semantic
```

**Tuning:**
- Increase N_ℓ if attention weights are too diffuse
- Decrease N_ℓ to reduce computation

### 10.3 Dimensions

**Principle:** Richer representations for slower levels

```python
d_0 = 128   # Episodic
d_1 = 256   # Working
d_2 = 512   # Semantic
```

**Tuning:**
- Increase d_ℓ if retrieval quality is poor
- Decrease d_ℓ to reduce memory footprint

---

## 11. Future Extensions

### 11.1 Adaptive Decay

Learn decay rates per level:

```
d_t^(ℓ) = σ(W_d^(ℓ) z_t^(ℓ)) ∈ (0,1)
```

**Benefit:** Memory adapts decay based on content importance.

### 11.2 Sparse Memory

Use sparse tensors for M_t^(ℓ):

```python
M_t[ℓ] = torch.sparse_coo_tensor(indices, values, size=(N_ℓ, d_ℓ))
```

**Benefit:** Scales to very large memory (N_ℓ > 10,000).

### 11.3 Episodic Replay

Periodically replay important memories:

```
M_t^(ℓ) ← M_t^(ℓ) + γ_replay M_important^(ℓ)
```

**Benefit:** Prevents catastrophic forgetting of critical experiences.

---

## 12. Summary

The CMS formal memory update rule provides:

✅ **Bounded, stable memory** via exponential decay  
✅ **Hierarchical organization** across timescales  
✅ **Content-addressable retrieval** via attention  
✅ **Efficient computation** suitable for edge devices  
✅ **Theoretical guarantees** via Lyapunov analysis  

**Core equation:**
```
M_t^(ℓ) = (1 - d_ℓ) M_{t-1}^(ℓ) + g_t^(ℓ) (α̃_t^(ℓ) ⊗ v_t^(ℓ))
```

This elegant formulation combines:
- **Decay** (1 - d_ℓ) for forgetting
- **Gating** g_t^(ℓ) for write control
- **Addressing** α̃_t^(ℓ) for where to write
- **Content** v_t^(ℓ) for what to write

Ready for implementation in the HOPE architecture! 🚀
