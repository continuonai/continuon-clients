# Seed Model: Universal Robot Initialization

**Version:** 1.2.0  
**Status:** Core Architecture  
**Date:** 2026-01-02

## Overview

The **Seed Model** is the universal initialization point for every robot in the Continuon ecosystem. It is not a temporary bootstrap—it is a permanent, hardware-agnostic core that:

- **Initializes** every new robot that connects to the ecosystem
- **Runs on any chip**: ARM, x64, RISC-V, quantum, neuromorphic
- **Provides** the foundation for all higher-level capabilities
- **Evolves** through continuous learning while maintaining core stability

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEED MODEL: UNIVERSAL FOUNDATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  New Robot Connects                                                          │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         SEED MODEL                                      ││
│  │                   (Hardware-Agnostic Core)                              ││
│  │                                                                          ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    ││
│  │  │ ARM (Pi5)   │  │ x64 (PC)    │  │ RISC-V      │  │ Quantum     │    ││
│  │  │ Jetson      │  │ Server      │  │ Edge        │  │ Future      │    ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    ││
│  │                                                                          ││
│  │  Universal Capabilities:                                                 ││
│  │  • World Model (next-token prediction)                                  ││
│  │  • Context Graph (relational reasoning)                                 ││
│  │  • Semantic Search (768-dim embeddings)                                 ││
│  │  • Decision Traces (explainability)                                     ││
│  │  • CMS Memory (multi-timescale)                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    CONTINUOUS EVOLUTION                                  ││
│  │  Seed → Experience → Local Learning → Cloud Aggregation → OTA Update   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

The ContinuonBrain seed model demonstrates advanced embodied AI capabilities through a unified architecture that integrates:

- **World Models** for predictive next-token generation
- **Context Graphs** for relational reasoning
- **Semantic Search** via EmbeddingGemma-300m
- **Decision Traces** for explainable behavior
- **Multi-Timescale Memory** (CMS) for temporal abstraction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEED MODEL CAPABILITIES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ World Model  │  │ Context      │  │ Semantic     │  │ Decision     │    │
│  │ Prediction   │  │ Graph        │  │ Search       │  │ Traces       │    │
│  │              │  │ Reasoning    │  │              │  │              │    │
│  │ Next-token   │  │ Entity/      │  │ EmbeddingGemma│  │ Explainable  │    │
│  │ + action     │  │ Relation     │  │ 768-dim      │  │ Provenance   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │            │
│         └─────────────────┴─────────────────┴─────────────────┘            │
│                                    │                                        │
│                                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                    WaveCore (Mamba SSM + Spectral)                     ││
│  │                    172K params | O(n) complexity                       ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                    │                                        │
│                                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                    CMS Memory (3-level hierarchical)                   ││
│  │                    Fast (100ms) | Mid (10s) | Slow (∞)                 ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. World Model: Predictive Next-Token Generation

The seed model implements a **generative world model** that predicts future states, enabling:

- **Action-conditioned prediction**: Given current state `s_t` and action `a_t`, predict `s_{t+1}`
- **Multi-step rollouts**: Plan trajectories in latent space
- **Counterfactual reasoning**: "What if I had done X instead?"

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORLD MODEL PREDICTION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Observation (o_t)                                               │
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Encoder: VQ-VAE / Vision Transformer                         ││
│  │ o_t → z_t (discrete tokens or continuous embedding)          ││
│  └─────────────────────────────────────────────────────────────┘│
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ WaveCore (Mamba SSM)                                         ││
│  │ h_t = SSM(h_{t-1}, z_t, a_{t-1})                             ││
│  │                                                               ││
│  │ State evolution: dh/dt = Ah + Bx                             ││
│  │ Selective gating: Δ = softplus(Linear(x))                    ││
│  └─────────────────────────────────────────────────────────────┘│
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Next-Token Prediction Head                                    ││
│  │ p(z_{t+1} | h_t, a_t) = softmax(W_z · h_t)                   ││
│  │ p(r_t | h_t) = Reward prediction                              ││
│  │ p(done | h_t) = Episode termination                           ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# continuonbrain/jax_models/core_model.py

class WorldModelHead(nn.Module):
    """Next-token prediction for world modeling."""
    config: CoreModelConfig
    
    @nn.compact
    def __call__(self, h_t: jnp.ndarray, a_t: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Combine hidden state with action
        ha = jnp.concatenate([h_t, a_t], axis=-1)
        
        # Predict next latent
        z_next_logits = nn.Dense(self.config.num_vq_vocab)(ha)
        
        # Predict reward
        reward_pred = nn.Dense(1)(ha)
        
        # Predict done
        done_logits = nn.Dense(2)(ha)
        
        return {
            'z_next_logits': z_next_logits,
            'reward_pred': reward_pred,
            'done_logits': done_logits,
        }
```

### Training Objective

```python
# World model loss (next-token + reward + done)
L_world = L_reconstruction + λ_r * L_reward + λ_d * L_done

# Where:
# L_reconstruction = CrossEntropy(z_next_pred, z_next_true)
# L_reward = MSE(r_pred, r_true)
# L_done = CrossEntropy(done_pred, done_true)
```

---

## 2. Context Graph Reasoning

The seed model maintains a **context graph** for relational reasoning about entities, objects, and their relationships.

### Graph Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT GRAPH                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Nodes (Entities):                                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Object  │  │ Person  │  │ Location│  │ Concept │            │
│  │ "cup"   │  │ "user"  │  │ "table" │  │ "hot"   │            │
│  │ emb[768]│  │ emb[768]│  │ emb[768]│  │ emb[768]│            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                  │
│  Edges (Relations):                                              │
│       │            │            │            │                  │
│       └──── on ────┼──── near ──┘            │                  │
│                    │                         │                  │
│                    └─── wants ───────────────┘                  │
│                                                                  │
│  Edge Attributes:                                                │
│  - type: spatial | causal | semantic | temporal                  │
│  - weight: attention-based salience                              │
│  - provenance: which observation created this edge               │
│  - timestamp: when last updated                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Graph Operations

```python
# continuonbrain/core/context_graph_store.py

class ContextGraphStore:
    """Graph-based reasoning for embodied agents."""
    
    def add_observation(self, obs: Observation) -> None:
        """Update graph from visual/sensor observation."""
        # Extract entities
        entities = self.vision_core.detect_objects(obs.image)
        
        # Add/update nodes
        for entity in entities:
            node = GraphNode(
                id=entity.track_id,
                type=entity.category,
                embedding=self.embed(entity.label),
                attributes={
                    'bbox': entity.bbox,
                    'confidence': entity.score,
                    'depth': entity.depth,
                },
                timestamp=obs.timestamp,
            )
            self.graph.upsert_node(node)
        
        # Infer spatial relations
        for e1, e2 in itertools.combinations(entities, 2):
            relation = self.infer_spatial_relation(e1, e2)
            if relation:
                self.graph.add_edge(e1.id, e2.id, relation)
    
    def query(self, question: str) -> List[GraphNode]:
        """Semantic graph traversal for question answering."""
        query_emb = self.embed(question)
        
        # Find relevant nodes
        candidates = self.graph.semantic_search(query_emb, k=10)
        
        # Expand via relations
        expanded = self.graph.expand_neighbors(candidates, hops=2)
        
        return self.rank_by_relevance(expanded, query_emb)
    
    def reason(self, goal: str) -> ReasoningTrace:
        """Multi-hop reasoning over graph."""
        trace = ReasoningTrace()
        
        # Parse goal into subgoals
        subgoals = self.parse_goal(goal)
        
        for subgoal in subgoals:
            # Find relevant context
            context = self.query(subgoal)
            
            # Apply reasoning rules
            conclusions = self.apply_rules(context, subgoal)
            
            trace.add_step(subgoal, context, conclusions)
        
        return trace
```

### Graph Attention for Reasoning

```python
# Multi-head graph attention for relation reasoning
class GraphAttention(nn.Module):
    """Graph attention for context reasoning."""
    
    @nn.compact
    def __call__(self, nodes: jnp.ndarray, edges: jnp.ndarray, query: jnp.ndarray):
        # nodes: [N, d], edges: [E, 2], query: [d]
        
        # Compute attention scores
        Q = nn.Dense(64)(query)
        K = nn.Dense(64)(nodes)
        V = nn.Dense(64)(nodes)
        
        # Edge-aware attention
        attn = jnp.einsum('d,nd->n', Q, K) / jnp.sqrt(64)
        attn = nn.softmax(attn)
        
        # Aggregate
        context = jnp.einsum('n,nd->d', attn, V)
        
        return context, attn
```

---

## 3. Semantic Search

The seed model uses **EmbeddingGemma-300m** for 768-dimensional semantic embeddings, enabling:

- **Experience retrieval**: Find similar past interactions
- **Knowledge grounding**: Connect observations to learned concepts
- **Cross-modal search**: Match text queries to visual memories

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC SEARCH PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query: "Where did I put my keys?"                               │
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ EmbeddingGemma-300m                                          ││
│  │ "Where did I put my keys?" → q ∈ ℝ^768                       ││
│  └─────────────────────────────────────────────────────────────┘│
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Vector Index (FAISS / ScaNN)                                  ││
│  │ top_k = argmax_i cos(q, e_i)                                 ││
│  │                                                               ││
│  │ Indexed memories:                                             ││
│  │ - RLDS episodes (91K steps)                                   ││
│  │ - Conversation history                                        ││
│  │ - Visual observations                                         ││
│  │ - Context graph nodes                                         ││
│  └─────────────────────────────────────────────────────────────┘│
│       ↓                                                          │
│  Retrieved: [                                                    │
│    "You placed keys on the kitchen counter (0.89)",              │
│    "Keys were last seen near the door (0.76)",                   │
│    "Conversation about losing keys yesterday (0.71)",            │
│  ]                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# continuonbrain/services/embedding_gemma.py

class EmbeddingGemmaEncoder:
    """768-dim semantic embeddings for search and retrieval."""
    
    DEFAULT_MODEL_ID = "google/embeddinggemma-300m"
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode search query with query-specific prompt."""
        prompt = f"search_query: {query}"
        return self.model.encode([prompt])[0]
    
    def encode_document(self, doc: str) -> np.ndarray:
        """Encode document/memory for indexing."""
        prompt = f"search_document: {doc}"
        return self.model.encode([prompt])[0]
    
    def encode_batch(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Batch encode for efficiency."""
        prefix = "search_query: " if is_query else "search_document: "
        prompts = [prefix + t for t in texts]
        return self.model.encode(prompts)

# Usage in experience logger
class ExperienceLogger:
    def search_conversations(self, query: str, max_results: int = 5) -> List[Dict]:
        query_embedding = self.encoder.encode_query(query)
        
        # Search vector index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            max_results
        )
        
        # Return ranked results with relevance scores
        return [
            {
                **self.memories[idx],
                'relevance': 1 - distances[0][i],  # Convert distance to similarity
            }
            for i, idx in enumerate(indices[0])
        ]
```

---

## 4. Decision Traces

The seed model logs **decision traces** for explainable AI, enabling:

- **Provenance tracking**: Why did the agent take this action?
- **Debugging**: Trace failures back to root causes
- **Learning from mistakes**: Identify and correct decision patterns

### Trace Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION TRACE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  trace_id: "dt_20260102_100523_abc123"                          │
│  timestamp: 2026-01-02T10:05:23.456Z                            │
│  agent: "hope_agent_manager"                                     │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Input Context                                              │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ observation: { image: "...", depth: "...", pose: [...] }   │  │
│  │ user_query: "Pick up the red cup"                          │  │
│  │ memory_context: [                                          │  │
│  │   { text: "Red cup is on table", salience: 0.92 },         │  │
│  │   { text: "User prefers left hand", salience: 0.71 },      │  │
│  │ ]                                                          │  │
│  │ graph_context: [                                           │  │
│  │   { node: "red_cup", relation: "on", target: "table" },    │  │
│  │ ]                                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Reasoning Steps                                            │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ 1. Goal parsing: "pick up" → GRASP action                  │  │
│  │ 2. Object grounding: "red cup" → object_id: "cup_42"       │  │
│  │ 3. Spatial reasoning: cup at (0.3, 0.5, 0.1) in base_link  │  │
│  │ 4. Path planning: 5-waypoint trajectory generated          │  │
│  │ 5. Safety check: PASSED (no obstacles, reachable)          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Decision                                                   │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ action: "grasp"                                            │  │
│  │ target: "cup_42"                                           │  │
│  │ confidence: 0.94                                           │  │
│  │ alternatives: [                                            │  │
│  │   { action: "ask_clarify", confidence: 0.03 },             │  │
│  │   { action: "point_to", confidence: 0.02 },                │  │
│  │ ]                                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Outcome                                                    │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ success: true                                              │  │
│  │ reward: 1.0                                                │  │
│  │ feedback: null                                             │  │
│  │ error: null                                                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# continuonbrain/core/decision_trace_logger.py

@dataclass
class DecisionTrace:
    trace_id: str
    timestamp: datetime
    agent: str
    input_context: InputContext
    reasoning_steps: List[ReasoningStep]
    decision: Decision
    outcome: Optional[Outcome] = None

class DecisionTraceLogger:
    """Log decision traces for explainability and learning."""
    
    def start_trace(self, agent: str, observation: Observation) -> DecisionTrace:
        return DecisionTrace(
            trace_id=f"dt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
            timestamp=datetime.now(),
            agent=agent,
            input_context=InputContext(observation=observation),
            reasoning_steps=[],
            decision=None,
        )
    
    def add_reasoning_step(self, trace: DecisionTrace, step: ReasoningStep) -> None:
        trace.reasoning_steps.append(step)
    
    def set_decision(self, trace: DecisionTrace, decision: Decision) -> None:
        trace.decision = decision
    
    def record_outcome(self, trace: DecisionTrace, outcome: Outcome) -> None:
        trace.outcome = outcome
        
        # Log to RLDS for learning
        self.log_to_rlds(trace)
        
        # Update context graph
        self.context_graph.add_decision_edge(trace)
    
    def log_to_rlds(self, trace: DecisionTrace) -> None:
        """Convert trace to RLDS episode step."""
        step = {
            'observation': trace.input_context.to_dict(),
            'action': trace.decision.action,
            'reward': trace.outcome.reward if trace.outcome else 0.0,
            'metadata': {
                'trace_id': trace.trace_id,
                'reasoning_steps': [s.to_dict() for s in trace.reasoning_steps],
                'confidence': trace.decision.confidence,
            }
        }
        self.rlds_logger.log_step(step)
```

---

## 5. Multi-Timescale Memory (CMS)

The **Continuous Memory System** provides hierarchical temporal abstraction:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CMS HIERARCHY                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 0: FAST (Episodic)                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ τ = 100ms | decay = 0.9 | slots = 64                        ││
│  │ Content: Raw sensory, motor commands, immediate context     ││
│  │ Update: Every perception cycle                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                         │                                        │
│                         ▼ consolidation                          │
│  Level 1: MID (Working)                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ τ = 10s | decay = 0.99 | slots = 128                        ││
│  │ Content: Current task state, active goals, recent events    ││
│  │ Update: Event-driven (significant changes)                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                         │                                        │
│                         ▼ consolidation                          │
│  Level 2: SLOW (Semantic)                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ τ = ∞ | decay = 0.999 | slots = 256                         ││
│  │ Content: Learned skills, persistent knowledge, identity     ││
│  │ Update: Sleep-like consolidation (background)                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Memory Operations:                                              │
│  ─────────────────                                               │
│  READ:  c_t = CMS_Read(M, query)                                │
│         Attention-weighted retrieval across all levels           │
│                                                                  │
│  WRITE: M_t = (1-α)M_{t-1} + α·new_memory                       │
│         Content-addressable write with salience gating           │
│                                                                  │
│  CONSOLIDATE: M_slow = consolidate(M_fast, M_mid)               │
│         Pattern compression, schema extraction                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation (JAX)

```python
# continuonbrain/jax_models/cms_jax.py

class CMSMemory(nn.Module):
    """JAX implementation of Continuous Memory System."""
    config: CMSConfig
    
    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,       # [B, d_query]
        memories: List[jnp.ndarray],  # [[B, N_l, d_l], ...]
        keys: List[jnp.ndarray],      # [[B, N_l, d_k], ...]
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """Read from hierarchical memory."""
        
        contexts = []
        attentions = []
        
        for level, (M, K) in enumerate(zip(memories, keys)):
            # Compute attention
            # scores[b, n] = query[b] · K[b, n] / √d_k
            scores = jnp.einsum('bd,bnd->bn', query, K) / jnp.sqrt(self.config.d_k)
            attn = nn.softmax(scores, axis=-1)
            
            # Retrieve context
            # c[b, d] = Σ_n attn[b, n] * M[b, n, d]
            c = jnp.einsum('bn,bnd->bd', attn, M)
            
            contexts.append(c)
            attentions.append(attn)
        
        # Hierarchical mixing
        # Learn to weight different timescales
        stacked = jnp.stack(contexts, axis=1)  # [B, L, d]
        mix_weights = nn.softmax(
            nn.Dense(self.config.num_levels)(query), axis=-1
        )  # [B, L]
        
        mixed = jnp.einsum('bl,bld->bd', mix_weights, stacked)
        
        return mixed, attentions
    
    def write(
        self,
        memories: List[jnp.ndarray],
        keys: List[jnp.ndarray],
        new_content: jnp.ndarray,
        new_key: jnp.ndarray,
        salience: float,
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """Write to memory with decay and salience gating."""
        
        updated_memories = []
        updated_keys = []
        
        for level, (M, K) in enumerate(zip(memories, keys)):
            decay = self.config.decays[level]
            
            # Find least salient slot
            saliences = jnp.linalg.norm(M, axis=-1)  # [B, N]
            min_idx = jnp.argmin(saliences, axis=-1)  # [B]
            
            # Decay existing memories
            M_decayed = M * decay
            K_decayed = K * decay
            
            # Write new content if salient enough
            if salience > self.config.write_threshold:
                # Replace least salient slot
                M_new = M_decayed.at[:, min_idx].set(new_content)
                K_new = K_decayed.at[:, min_idx].set(new_key)
            else:
                M_new = M_decayed
                K_new = K_decayed
            
            updated_memories.append(M_new)
            updated_keys.append(K_new)
        
        return updated_memories, updated_keys
```

---

## 6. Advanced Embodied AI Concepts

### 6.1 Action-Perception Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBODIED ACTION-PERCEPTION LOOP               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐                                    ┌─────────┐     │
│  │ Sensors │───────────────────────────────────►│ Encoder │     │
│  └─────────┘                                    └────┬────┘     │
│      ▲                                               │          │
│      │                                               ▼          │
│      │                                    ┌──────────────────┐  │
│      │                                    │ World Model      │  │
│      │                                    │ State: h_t       │  │
│      │                                    │ Predict: ẑ_{t+1} │  │
│      │                                    └────────┬─────────┘  │
│      │                                             │            │
│      │         ┌───────────────────────────────────┘            │
│      │         │                                                │
│      │         ▼                                                │
│      │  ┌────────────┐     ┌─────────────┐     ┌────────────┐  │
│      │  │ CMS Memory │◄───►│ HOPE Core   │◄───►│ Context    │  │
│      │  │ Retrieval  │     │ Reasoning   │     │ Graph      │  │
│      │  └────────────┘     └──────┬──────┘     └────────────┘  │
│      │                            │                             │
│      │                            ▼                             │
│      │                     ┌────────────┐                       │
│      │                     │ Policy     │                       │
│      │                     │ π(a|s,g)   │                       │
│      │                     └─────┬──────┘                       │
│      │                           │                              │
│      │                           ▼                              │
│  ┌───┴───┐                ┌────────────┐                        │
│  │ Robot │◄───────────────│ Actuators  │                        │
│  │ Body  │                └────────────┘                        │
│  └───────┘                                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Intrinsic Motivation

```python
# Curiosity-driven exploration
class IntrinsicMotivation:
    """Compute intrinsic rewards for exploration."""
    
    def compute_curiosity(self, state: jnp.ndarray, next_state: jnp.ndarray) -> float:
        """Prediction error as curiosity signal."""
        predicted = self.forward_model(state)
        error = jnp.mean((predicted - next_state) ** 2)
        return float(error)
    
    def compute_empowerment(self, state: jnp.ndarray) -> float:
        """Mutual information between actions and outcomes."""
        # I(a; s' | s) - how much control does the agent have?
        return self.empowerment_estimator(state)
    
    def compute_novelty(self, state: jnp.ndarray) -> float:
        """Distance from visited states."""
        distances = self.memory.query_nearest(state, k=5)
        return float(jnp.mean(distances))
```

### 6.3 Skill Learning

```python
# Hierarchical skill learning
class SkillLibrary:
    """Learned reusable behaviors."""
    
    skills: Dict[str, Skill] = {
        'grasp': Skill(
            preconditions=['object_visible', 'arm_reachable'],
            policy=GraspPolicy(),
            postconditions=['object_held'],
        ),
        'place': Skill(
            preconditions=['object_held', 'target_visible'],
            policy=PlacePolicy(),
            postconditions=['object_placed'],
        ),
        'navigate': Skill(
            preconditions=['target_known'],
            policy=NavigatePolicy(),
            postconditions=['at_target'],
        ),
    }
    
    def select_skill(self, goal: str, context: Dict) -> Optional[Skill]:
        """Select appropriate skill for goal."""
        for name, skill in self.skills.items():
            if skill.matches_goal(goal) and skill.check_preconditions(context):
                return skill
        return None
```

---

## 7. Hardware-Agnostic Architecture

The seed model is designed to run on **any compute platform**—from edge devices to quantum processors.

### Supported Architectures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HARDWARE PORTABILITY MATRIX                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Architecture      │ Runtime        │ Accelerator      │ Status            │
│  ──────────────────┼────────────────┼──────────────────┼─────────────────  │
│  ARM64 (Pi5)       │ JAX CPU        │ Hailo-8 NPU      │ ✅ Primary        │
│  ARM64 (Jetson)    │ JAX CUDA       │ Tensor Cores     │ ✅ Supported      │
│  x86_64 (PC)       │ JAX CPU/CUDA   │ NVIDIA GPU       │ ✅ Supported      │
│  x86_64 (Cloud)    │ JAX TPU        │ TPU v4/v5        │ ✅ Supported      │
│  RISC-V            │ Portable C     │ Custom NPU       │ 🔶 Planned        │
│  Apple Silicon     │ JAX Metal      │ ANE              │ 🔶 Planned        │
│  Quantum           │ Pennylane/JAX  │ QPU              │ 🔮 Future         │
│  Neuromorphic      │ Lava/Loihi     │ Intel Loihi 2    │ 🔮 Future         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Portability Principles

1. **Pure JAX Core**: All core computations use JAX, which compiles to XLA for any backend
2. **No Hardware-Specific Code in Model**: Hardware abstraction at the inference router level
3. **Graceful Degradation**: Falls back to CPU when accelerators unavailable
4. **Quantization-Ready**: Weights can be quantized for edge deployment

### Backend Selection

```python
# Automatic backend selection based on available hardware
from continuonbrain.jax_models.export.inference_router import InferenceRouter

router = InferenceRouter()

# Automatically selects best available backend:
# 1. TPU (if on Cloud TPU VM)
# 2. CUDA (if NVIDIA GPU available)
# 3. Hailo (if Hailo-8 NPU detected)
# 4. CPU (always available fallback)

output = router.infer(observation)
print(f"Backend used: {router.active_backend}")  # e.g., "hailo", "cuda", "cpu"
```

### Edge Deployment

```python
# Optimized for edge devices (Pi5, Jetson, etc.)
from continuonbrain.jax_models.config import CoreModelConfig

# Automatic config based on detected hardware
config = CoreModelConfig.for_device()  # Detects and optimizes

# Manual override for specific hardware
config_pi5 = CoreModelConfig.pi5_optimized()      # 172K params, fits in 512MB
config_jetson = CoreModelConfig.jetson_optimized() # Larger model for Jetson
config_cloud = CoreModelConfig.cloud_optimized()   # Full model for TPU
```

### Seed Model Initialization Flow

Every new robot, regardless of hardware, starts with the same seed:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEW ROBOT INITIALIZATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Robot boots for first time                                               │
│       │                                                                      │
│       ▼                                                                      │
│  2. Hardware detection (RuntimeContext)                                      │
│       │  Detects: CPU arch, RAM, GPU/NPU, sensors                           │
│       ▼                                                                      │
│  3. Download/verify seed model                                               │
│       │  From: continuon.cloud/seed/v1.0.0/                                 │
│       │  Checksum: verified against manifest                                │
│       ▼                                                                      │
│  4. Select optimal config for hardware                                       │
│       │  Pi5 → pi5_optimized (172K params)                                  │
│       │  Jetson → jetson_optimized (1M params)                              │
│       │  Cloud → cloud_optimized (10M params)                               │
│       ▼                                                                      │
│  5. Initialize WaveCore + CMS + Context Graph                               │
│       │  All capabilities ready                                              │
│       ▼                                                                      │
│  6. Robot operational with full seed capabilities                           │
│       │                                                                      │
│       ▼                                                                      │
│  7. Begin experience collection → local learning → evolution                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cross-Platform Serialization

```python
# Seed model weights are platform-agnostic
import numpy as np

# Save (any platform)
weights = {'params': jax.tree_util.tree_map(np.array, params)}
np.savez('seed_model.npz', **weights)

# Load (any platform)
loaded = np.load('seed_model.npz', allow_pickle=True)
params = jax.tree_util.tree_map(jnp.array, dict(loaded))
```

---

## 8. Training Pipeline

### Continuous Learning (Post-Initialization)

After seed initialization, robots learn continuously:

| Phase | Location | Data | Frequency |
|-------|----------|------|-----------|
| **Experience Collection** | Device | RLDS episodes | Continuous |
| **Local Learning** | Device | Fast/Mid loops | Every session |
| **Cloud Aggregation** | TPU | Aggregated RLDS | Daily/Weekly |
| **OTA Distribution** | Cloud → Device | Updated weights | On-demand |

### Seed Model (Core - Never Deprecated)

| Component | Status | Details |
|-----------|--------|---------|
| **WaveCore** | ✅ 172K params | Mamba SSM, O(n) complexity |
| **CMS** | ✅ 3 levels | Fast/Mid/Slow with decay |
| **Scaffold** | 🔶 Gemma 3n | Chat generation (evolves with updates) |
| **Embeddings** | ✅ 768-dim | EmbeddingGemma (permanent) |
| **RLDS** | ✅ 4,219 episodes | 91K steps available |

### Production Phase (Post-Seed)

```
Device RLDS ──► Cloud TPU Slow Loop ──► OTA Bundle ──► Device Install
                     │
                     ▼
              WaveCore Production Weights
              (Gemma scaffold deprecated)
```

---

## 8. References

- [WaveCore Spec](./wavecore-spec.md)
- [HOPE/CMS VLA](./hope-cms-vla.md)
- [CMS Formal Spec](../continuonbrain/docs/CMS_FORMAL_SPEC.md)
- [RCAN Protocol](./rcan-protocol.md)
- [Training Plan](./training-plan.md)
