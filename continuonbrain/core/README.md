# Core Module: Context Graph & Decision Traces

This module implements the core cognitive infrastructure for the ContinuonBrain seed model.

---

## Components

| Component | File | Description |
|-----------|------|-------------|
| **Context Graph** | `context_graph_store.py` | Relational knowledge graph for entity reasoning |
| **Decision Traces** | `decision_trace_logger.py` | Explainable AI logging with provenance |
| **Settings** | `settings_store.py` | Persistent configuration storage |

---

## Context Graph

The context graph maintains **relational knowledge** about entities, objects, and their relationships.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTEXT GRAPH                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  NODES (Entities)                                                            │
│  ────────────────                                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ type: object│  │ type: person│  │ type: loc   │  │ type: concept│        │
│  │ id: cup_42  │  │ id: user_1  │  │ id: table_1 │  │ id: hot      │        │
│  │ emb: [768]  │  │ emb: [768]  │  │ emb: [768]  │  │ emb: [768]   │        │
│  │ bbox: [...]  │  │ face: [...]  │  │ bounds: [...] │  │              │        │
│  │ conf: 0.95  │  │ conf: 0.88  │  │ conf: 0.99  │  │ salience: 0.7│        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│  EDGES (Relations)                                                          │
│  ─────────────────                                                          │
│         │                │                │                │               │
│         └───── on ───────┼──── near ──────┘                │               │
│                          │                                 │               │
│                          └───── wants ─────────────────────┘               │
│                                                                              │
│  Edge Types:                                                                 │
│  • spatial: on, near, above, below, inside, touching                       │
│  • causal: caused, enables, prevents                                        │
│  • semantic: is_a, part_of, similar_to                                      │
│  • temporal: before, after, during                                          │
│  • possession: has, owns, holds                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Node Schema

```python
@dataclass
class GraphNode:
    id: str                       # Unique identifier (e.g., "cup_42")
    type: str                     # Entity type (object/person/location/concept)
    embedding: np.ndarray         # 768-dim semantic embedding
    attributes: Dict[str, Any]    # Type-specific attributes
    timestamp: datetime           # Last update time
    provenance: str               # Source observation/trace
    salience: float               # Attention weight (0-1)

# Example attributes by type
object_attrs = {
    'label': 'coffee cup',
    'bbox': [100, 200, 50, 50],
    'confidence': 0.95,
    'depth': 0.8,
    'color': 'red',
}

person_attrs = {
    'name': 'user',
    'face_embedding': [...],
    'role': 'owner',
    'last_seen': '2026-01-02T10:00:00',
}

location_attrs = {
    'name': 'kitchen table',
    'bounds': [0, 0, 100, 60],
    'type': 'furniture',
}
```

### Edge Schema

```python
@dataclass
class GraphEdge:
    source: str                   # Source node ID
    target: str                   # Target node ID
    relation: str                 # Relation type
    weight: float                 # Confidence/salience
    attributes: Dict[str, Any]    # Relation-specific data
    timestamp: datetime           # When created/updated
    provenance: str               # Source observation
```

### Operations

```python
from continuonbrain.core.context_graph_store import ContextGraphStore

graph = ContextGraphStore(db_path='/opt/continuonos/brain/context_graph.db')

# Add observation (updates nodes and infers relations)
graph.add_observation(Observation(
    timestamp=datetime.now(),
    image=frame,
    detected_objects=[
        {'label': 'cup', 'bbox': [100, 200, 50, 50], 'confidence': 0.95},
        {'label': 'table', 'bbox': [0, 300, 640, 180], 'confidence': 0.99},
    ],
    depth_map=depth,
))

# Semantic query
results = graph.query("Where is the red cup?")
# Returns: [
#   {'node': 'cup_42', 'relation': 'on', 'target': 'table_1', 'confidence': 0.92}
# ]

# Multi-hop traversal
path = graph.traverse(start='user_1', goal='cup_42', max_hops=3)
# Returns: user_1 → near → table_1 → on → cup_42

# Relational reasoning
conclusions = graph.reason(
    query="What can I reach from here?",
    rules=[
        "if X near Y and Y reachable, then X reachable",
        "if X on Y and Y visible, then X visible",
    ]
)
```

### Spatial Relation Inference

```python
def infer_spatial_relation(obj1: DetectedObject, obj2: DetectedObject) -> Optional[str]:
    """Infer spatial relation between two objects."""
    
    # Bounding box overlap
    iou = compute_iou(obj1.bbox, obj2.bbox)
    
    # Vertical relationship
    if obj1.bbox.bottom < obj2.bbox.top:
        return 'above'
    elif obj1.bbox.top > obj2.bbox.bottom:
        return 'below'
    
    # Containment
    if is_contained(obj1.bbox, obj2.bbox):
        return 'inside'
    
    # Proximity (using depth)
    distance = np.linalg.norm(obj1.position - obj2.position)
    if distance < 0.1:
        return 'touching'
    elif distance < 0.5:
        return 'near'
    
    return None
```

---

## Decision Traces

Every decision is logged with **full provenance** for explainability, debugging, and learning.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DECISION TRACE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  METADATA                                                                    │
│  ────────                                                                    │
│  trace_id: dt_20260102_100523_abc123                                        │
│  timestamp: 2026-01-02T10:05:23.456Z                                        │
│  agent: hope_agent_manager                                                   │
│  session_id: sess_xyz789                                                     │
│                                                                              │
│  INPUT CONTEXT                                                               │
│  ─────────────                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ observation:                                                             ││
│  │   image_hash: sha256:abc...                                              ││
│  │   pose: [x, y, z, qx, qy, qz, qw]                                       ││
│  │   detected_objects: [{label: cup, bbox: [...], conf: 0.95}]             ││
│  │                                                                          ││
│  │ user_query: "Pick up the red cup"                                        ││
│  │                                                                          ││
│  │ memory_context: (from CMS)                                               ││
│  │   - "Red cup is on the kitchen table" (salience: 0.92)                  ││
│  │   - "User prefers left hand" (salience: 0.71)                           ││
│  │                                                                          ││
│  │ graph_context: (from Context Graph)                                      ││
│  │   - cup_42 --on--> table_1                                               ││
│  │   - table_1 --near--> user_1                                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  REASONING STEPS                                                             │
│  ───────────────                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Step 1: GOAL_PARSING                                                     ││
│  │   input: "Pick up the red cup"                                           ││
│  │   output: {action: GRASP, target: "red cup"}                             ││
│  │   confidence: 0.98                                                       ││
│  │   duration_ms: 12                                                        ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ Step 2: OBJECT_GROUNDING                                                 ││
│  │   input: "red cup"                                                       ││
│  │   candidates: [cup_42 (0.95), cup_17 (0.23)]                            ││
│  │   output: {object_id: cup_42, location: [0.3, 0.5, 0.1]}                ││
│  │   confidence: 0.95                                                       ││
│  │   duration_ms: 45                                                        ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ Step 3: REACHABILITY_CHECK                                               ││
│  │   input: {object: cup_42, arm: right}                                    ││
│  │   output: {reachable: true, ik_solution: [...]}                          ││
│  │   confidence: 1.0                                                        ││
│  │   duration_ms: 23                                                        ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ Step 4: PATH_PLANNING                                                    ││
│  │   input: {start: current_pose, goal: grasp_pose}                        ││
│  │   output: {waypoints: [[...], [...], [...]], duration: 2.5s}            ││
│  │   confidence: 0.99                                                       ││
│  │   duration_ms: 156                                                       ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ Step 5: SAFETY_CHECK                                                     ││
│  │   input: {trajectory: [...], scene: current_scene}                      ││
│  │   output: {safe: true, collisions: [], violations: []}                  ││
│  │   confidence: 1.0                                                        ││
│  │   duration_ms: 34                                                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  DECISION                                                                    │
│  ────────                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ action: GRASP                                                            ││
│  │ target: cup_42                                                           ││
│  │ parameters: {approach: overhead, gripper: 0.05}                         ││
│  │ confidence: 0.94                                                         ││
│  │                                                                          ││
│  │ alternatives:                                                            ││
│  │   - {action: ASK_CLARIFY, confidence: 0.03}                             ││
│  │   - {action: POINT_TO, confidence: 0.02}                                ││
│  │   - {action: WAIT, confidence: 0.01}                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  OUTCOME (after execution)                                                   │
│  ───────                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ success: true                                                            ││
│  │ reward: 1.0                                                              ││
│  │ execution_time_ms: 2847                                                  ││
│  │ error: null                                                              ││
│  │ feedback: null                                                           ││
│  │ rlds_step_id: step_20260102_100526_def456                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Usage

```python
from continuonbrain.core.decision_trace_logger import DecisionTraceLogger, ReasoningStep

logger = DecisionTraceLogger(storage_path='/opt/continuonos/brain/traces/')

# Start a new trace
trace = logger.start_trace(
    agent="hope_agent_manager",
    observation=current_observation,
    user_query="Pick up the red cup",
)

# Add memory context
trace.add_memory_context(cms_read_results)

# Add graph context
trace.add_graph_context(context_graph.query(user_query))

# Log reasoning steps
logger.add_step(trace, ReasoningStep(
    operation="goal_parsing",
    input={"query": user_query},
    output={"action": "GRASP", "target": "red cup"},
    confidence=0.98,
    duration_ms=12,
))

logger.add_step(trace, ReasoningStep(
    operation="object_grounding",
    input={"target": "red cup"},
    output={"object_id": "cup_42", "location": [0.3, 0.5, 0.1]},
    confidence=0.95,
    duration_ms=45,
))

# Set final decision
logger.set_decision(trace, Decision(
    action="GRASP",
    target="cup_42",
    confidence=0.94,
    parameters={"approach": "overhead", "gripper": 0.05},
    alternatives=[
        {"action": "ASK_CLARIFY", "confidence": 0.03},
    ],
))

# Record outcome after execution
logger.record_outcome(trace, Outcome(
    success=True,
    reward=1.0,
    execution_time_ms=2847,
))

# Trace is automatically persisted and linked to RLDS
```

### Querying Traces

```python
# Find traces by criteria
failed_grasps = logger.query(
    action="GRASP",
    success=False,
    time_range=(yesterday, today),
)

# Analyze failure patterns
for trace in failed_grasps:
    print(f"Trace: {trace.trace_id}")
    print(f"  Target: {trace.decision.target}")
    print(f"  Error: {trace.outcome.error}")
    print(f"  Reasoning steps: {len(trace.reasoning_steps)}")
    
    # Find which step had lowest confidence
    weakest = min(trace.reasoning_steps, key=lambda s: s.confidence)
    print(f"  Weakest step: {weakest.operation} ({weakest.confidence:.2f})")
```

---

## Integration

### Context Graph ↔ CMS Memory

```python
# Graph nodes influence memory salience
for node in context_graph.get_active_nodes():
    cms.update_salience(node.embedding, node.salience)

# Memory context informs graph queries
memory_context = cms.read(query_embedding)
graph_results = context_graph.query(query, context=memory_context)
```

### Decision Traces ↔ RLDS

```python
# Each trace creates an RLDS step
def trace_to_rlds_step(trace: DecisionTrace) -> Dict:
    return {
        'observation': trace.input_context.to_dict(),
        'action': trace.decision.action,
        'reward': trace.outcome.reward if trace.outcome else 0.0,
        'info': {
            'trace_id': trace.trace_id,
            'reasoning_steps': [s.to_dict() for s in trace.reasoning_steps],
            'confidence': trace.decision.confidence,
            'alternatives': trace.decision.alternatives,
        }
    }
```

---

## File Locations

| File | Description |
|------|-------------|
| `context_graph_store.py` | Graph storage and operations |
| `decision_trace_logger.py` | Trace logging and querying |
| `settings_store.py` | Persistent settings |
| `/opt/continuonos/brain/context_graph.db` | SQLite graph database |
| `/opt/continuonos/brain/traces/` | Decision trace storage |

