# ContinuonBrain Seed Model Capabilities

**Version:** 3.0.0  
**Status:** Production Ready  
**Date:** January 2026

## Quick Stats

| Metric | Value |
|--------|-------|
| **Parameters** | 3.4M |
| **Memory** | 14 MB |
| **Embedding** | EmbeddingGemma-300m (768-dim) |
| **Inference** | 231 steps/sec (4.3ms) |
| **Loss** | 0.011 |

The **Seed Model** is the universal initialization point for every robot in the Continuon ecosystem. It provides the foundational cognitive capabilities that all robots share, regardless of their hardware platform.

## Key Principles

| Principle | Description |
|-----------|-------------|
| **Universal** | Every robot starts from the same seed |
| **Hardware-Agnostic** | Runs on ARM, x64, RISC-V, quantum, neuromorphic |
| **Permanent** | Not deprecated—always the initialization point |
| **Evolvable** | Continuous learning builds on the seed foundation |
| **Golden Rule** | Must run on devices with <8GB RAM |

This document describes the advanced embodied AI capabilities implemented in the ContinuonBrain seed model.

---

## 1. Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTINUONBRAIN SEED MODEL                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                           INPUT LAYER                                   ││
│  │  Vision │ Depth │ Pose │ Audio │ Haptics │ Language │ Proprioception   ││
│  └───────────────────────────────┬─────────────────────────────────────────┘│
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         ENCODER BANK                                    ││
│  │  VisionCore │ AudioEncoder │ LanguageEncoder │ ProprioEncoder          ││
│  │       ↓            ↓               ↓                 ↓                  ││
│  │    z_vis       z_audio         z_lang           z_proprio              ││
│  └───────────────────────────────┬─────────────────────────────────────────┘│
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         WAVECORE (Mamba SSM)                            ││
│  │                                                                          ││
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐                       ││
│  │  │ Fast Loop  │   │ Mid Loop   │   │ Slow Loop  │                       ││
│  │  │ τ = 10ms   │──►│ τ = 100ms  │──►│ τ = 1s     │                       ││
│  │  │ Reflexes   │   │ Attention  │   │ Planning   │                       ││
│  │  └────────────┘   └────────────┘   └────────────┘                       ││
│  │                                                                          ││
│  │  State Space Model: dh/dt = Ah + Bx, y = Ch + Dx                        ││
│  │  Selective Mechanism: Δ = softplus(Linear(x))                           ││
│  │  Parameters: 172,202                                                     ││
│  └───────────────────────────────┬─────────────────────────────────────────┘│
│                                  │                                          │
│         ┌────────────────────────┼────────────────────────┐                 │
│         │                        │                        │                 │
│         ▼                        ▼                        ▼                 │
│  ┌────────────┐          ┌────────────┐          ┌────────────┐            │
│  │ CMS Memory │          │ Context    │          │ Decision   │            │
│  │ (3 levels) │◄────────►│ Graph      │◄────────►│ Trace      │            │
│  └────────────┘          └────────────┘          └────────────┘            │
│         │                        │                        │                 │
│         └────────────────────────┼────────────────────────┘                 │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         POLICY HEADS                                    ││
│  │  Arm Control │ Navigation │ Chat │ Gaze │ Expression                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. World Model (Next-Token Prediction)

The seed model learns a **predictive world model** that forecasts future states conditioned on actions.

### Key Features

| Feature | Description |
|---------|-------------|
| **State Prediction** | Predict `s_{t+1}` given `(s_t, a_t)` |
| **Reward Prediction** | Anticipate rewards before receiving them |
| **Termination Prediction** | Know when episodes will end |
| **Multi-Step Rollouts** | Imagine future trajectories for planning |

### Mathematical Formulation

```
World Model: M_θ(s_t, a_t) → (ŝ_{t+1}, r̂_t, d̂_t)

Where:
  s_t  = current state embedding
  a_t  = action taken
  ŝ_{t+1} = predicted next state
  r̂_t = predicted reward
  d̂_t = predicted done probability

Loss: L = ||s_{t+1} - ŝ_{t+1}||² + λ_r(r_t - r̂_t)² + λ_d BCE(d_t, d̂_t)
```

### Usage

```python
from continuonbrain.jax_models.core_model import CoreModel

# Initialize
model, params = make_core_model(rng, obs_dim=64, action_dim=7, output_dim=7)

# Forward pass returns predicted next state
y_pred, info = model.apply(params, obs, action, reward, s_prev, w_prev, p_prev, cms)

# info contains:
# - 'fast_state': s_t (reactive state)
# - 'wave_state': w_t (temporal state)
# - 'particle_state': p_t (local dynamics)
```

---

## 3. Context Graph Reasoning

The context graph maintains **relational knowledge** about entities and their relationships.

### Graph Structure

```python
@dataclass
class GraphNode:
    id: str                    # Unique identifier
    type: str                  # 'object', 'person', 'location', 'concept'
    embedding: np.ndarray      # 768-dim semantic embedding
    attributes: Dict[str, Any] # bbox, confidence, depth, etc.
    timestamp: datetime        # Last update time
    provenance: str           # Which observation created this

@dataclass  
class GraphEdge:
    source: str               # Source node ID
    target: str               # Target node ID
    relation: str             # 'on', 'near', 'holding', 'wants', etc.
    weight: float             # Attention-based salience
    timestamp: datetime
```

### Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| **Query** | Semantic search over nodes | "Find the red cup" |
| **Traverse** | Multi-hop relation following | "What is on the table?" |
| **Update** | Add/modify nodes from observation | New object detected |
| **Reason** | Apply rules for inference | "If A on B and B on C, then A above C" |

### Example

```python
from continuonbrain.core.context_graph_store import ContextGraphStore

graph = ContextGraphStore()

# Add observation
graph.add_observation(Observation(
    image=frame,
    detected_objects=[
        {'label': 'cup', 'bbox': [100, 200, 50, 50], 'confidence': 0.95},
        {'label': 'table', 'bbox': [0, 300, 640, 180], 'confidence': 0.99},
    ]
))

# Query
results = graph.query("Where is the cup?")
# Returns: [{'node': 'cup_1', 'relation': 'on', 'target': 'table_1', 'confidence': 0.92}]

# Multi-hop reasoning
trace = graph.reason("Can I reach the cup?")
# Returns reasoning trace with intermediate steps
```

---

## 4. Semantic Search (EmbeddingGemma-300m)

768-dimensional semantic embeddings enable **meaning-based retrieval**.

### Model Details

| Property | Value |
|----------|-------|
| **Model** | google/embeddinggemma-300m |
| **Dimensions** | 768 |
| **Max Tokens** | 512 |
| **Similarity** | Cosine |

### Search Pipeline

```
Query: "How do I make coffee?"
    ↓
EmbeddingGemma-300m → q ∈ ℝ^768
    ↓
Vector Index Search → top-k candidates
    ↓
Re-ranking → Final results

Results:
1. "Pour hot water over ground coffee beans" (0.89)
2. "Coffee maker instructions: fill reservoir..." (0.82)
3. "The barista demonstrated latte art" (0.71)
```

### Code

```python
from continuonbrain.services.embedding_gemma import get_embedding_model

encoder = get_embedding_model()

# Encode query
query_emb = encoder.encode_query("How does the arm move?")

# Encode documents
doc_embs = encoder.encode([
    "The robotic arm has 7 degrees of freedom",
    "Coffee is best served hot",
    "Arm movements are controlled by WaveCore",
])

# Compute similarities
similarities = np.dot(doc_embs, query_emb)
# [0.87, 0.12, 0.91]
```

---

## 5. Decision Traces (Explainability)

Every decision is logged with full **provenance** for debugging and learning.

### Trace Fields

```yaml
trace_id: "dt_20260102_100523_abc123"
timestamp: 2026-01-02T10:05:23.456Z
agent: "hope_agent_manager"

input_context:
  observation:
    image_hash: "sha256:abc..."
    pose: [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
  user_query: "Pick up the red cup"
  memory_context:
    - text: "Red cup is on table"
      salience: 0.92
      source: "observation_20260102_100520"

reasoning_steps:
  - step: 1
    operation: "goal_parsing"
    input: "Pick up the red cup"
    output: {action: "grasp", target: "red_cup"}
    
  - step: 2
    operation: "object_grounding"
    input: "red_cup"
    output: {object_id: "cup_42", location: [0.3, 0.5, 0.1]}
    
  - step: 3
    operation: "path_planning"
    input: {start: [0.0, 0.0, 0.5], goal: [0.3, 0.5, 0.1]}
    output: {waypoints: [[...], [...], [...]], duration: 2.5}

decision:
  action: "grasp"
  target: "cup_42"
  confidence: 0.94
  alternatives:
    - {action: "ask_clarify", confidence: 0.03}
    - {action: "point_to", confidence: 0.02}

outcome:
  success: true
  reward: 1.0
  execution_time_ms: 2847
```

### Usage

```python
from continuonbrain.core.decision_trace_logger import DecisionTraceLogger

logger = DecisionTraceLogger()

# Start trace
trace = logger.start_trace("hope_agent", observation)

# Log reasoning
logger.add_step(trace, ReasoningStep(
    operation="goal_parsing",
    input=user_query,
    output=parsed_goal,
))

# Record decision
logger.set_decision(trace, Decision(
    action="grasp",
    target="cup_42",
    confidence=0.94,
))

# Record outcome
logger.record_outcome(trace, Outcome(
    success=True,
    reward=1.0,
))
```

---

## 6. CMS Memory System

**Continuous Memory System** with 3 hierarchical levels.

### Configuration

```python
CMS_CONFIG = {
    'levels': [
        {
            'name': 'fast',
            'timescale': '100ms',
            'slots': 64,
            'dim': 32,
            'decay': 0.9,
            'content': 'Raw sensory, motor commands',
        },
        {
            'name': 'mid', 
            'timescale': '10s',
            'slots': 128,
            'dim': 64,
            'decay': 0.99,
            'content': 'Current task, active goals',
        },
        {
            'name': 'slow',
            'timescale': '∞',
            'slots': 256,
            'dim': 128,
            'decay': 0.999,
            'content': 'Skills, knowledge, identity',
        },
    ]
}
```

### Memory Operations

```python
# READ: Attention-weighted retrieval
context, attention = cms.read(query, memories, keys)

# WRITE: Salience-gated storage
memories, keys = cms.write(memories, keys, new_content, salience=0.8)

# CONSOLIDATE: Background pattern compression
cms.consolidate(fast_memories, mid_memories, slow_memories)
```

### Formal Specification

See [CMS_FORMAL_SPEC.md](./CMS_FORMAL_SPEC.md) for mathematical details:

```
Read:  c_t = Σ_ℓ β_ℓ · softmax(K^(ℓ) q / √d_k) · M^(ℓ)
Write: M_t^(ℓ) = (1 - α_ℓ) M_{t-1}^(ℓ) + α_ℓ · w_t · v_t^T
Decay: M_t^(ℓ) = γ_ℓ · M_{t-1}^(ℓ)
```

---

## 7. RLDS Training Data

The seed model trains on **RLDS (Reinforcement Learning Dataset Specification)** episodes.

### Current Dataset

| Metric | Value |
|--------|-------|
| **Episodes** | 4,219 |
| **Steps** | 91,535 |
| **Chat Episodes** | 870 |
| **HOPE Eval Episodes** | 1,749 |

### Episode Structure

```json
{
  "episode_id": "ep_20260102_100523",
  "steps": [
    {
      "observation": {
        "image": "base64:...",
        "pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        "proprioception": [0.5, 0.3, ...]
      },
      "action": {
        "type": "joint_velocity",
        "values": [0.1, 0.0, -0.1, ...]
      },
      "reward": 0.5,
      "metadata": {
        "trace_id": "dt_...",
        "agent": "hope_agent"
      }
    }
  ],
  "episode_metadata": {
    "origin": "chat_learn",
    "duration_sec": 45.2,
    "success": true
  }
}
```

---

## 8. File Locations

| Component | Path |
|-----------|------|
| **Core Model** | `continuonbrain/jax_models/core_model.py` |
| **CMS (JAX)** | `continuonbrain/jax_models/cms_jax.py` |
| **CMS (PyTorch)** | `continuonbrain/hope_impl/cms.py` |
| **Context Graph** | `continuonbrain/core/context_graph_store.py` |
| **Decision Traces** | `continuonbrain/core/decision_trace_logger.py` |
| **Embedding Model** | `continuonbrain/services/embedding_gemma.py` |
| **Experience Logger** | `continuonbrain/services/experience_logger.py` |
| **RLDS Logger** | `continuonbrain/rlds/chat_rlds_logger.py` |
| **Seed Checkpoint** | `/opt/continuonos/brain/model/adapters/candidate/core_model_seed/` |
| **RLDS Episodes** | `/opt/continuonos/brain/rlds/episodes/` |

---

## 9. Next Steps

### Seed → Production Transition Criteria

1. **HOPE Eval Score** ≥ 80%
2. **Tool Router Accuracy** ≥ 70%
3. **WaveCore Stability** (Lyapunov λ < 0.1)
4. **Memory Consolidation** validated

### Post-Seed Architecture

- **Primary LLM**: HOPE Agent Manager (WaveCore, non-transformers)
- **Updates**: Cloud TPU slow loop → OTA bundles
- **Gemma**: Demoted to fallback only

See [seed-to-hope-evolution.md](../../docs/seed-to-hope-evolution.md) for full details.

