# Context Graphs: Wave–Particle Bridge for Situation-Aware Memory

Context graphs bridge raw memory (RLDS episodes, CMS spans, and HOPE state) to situation-aware intelligence by fusing the **particle** layer (symbolic graph) with the **wave** layer (dense embeddings and decay-aware state). They align with the HOPE/CMS dual design described in [wave_particle_rollout.md](./wave_particle_rollout.md) and the CMS rules in [CMS_FORMAL_SPEC.md](./CMS_FORMAL_SPEC.md).

## Dual Layers

- **Particle layer (symbolic graph):** nodes for entities, tools, intents, goals, policies, events/episodes, and assertions; typed edges (causal, temporal, membership, policy, tool-use) carrying provenance. Policies and permissions live here as first-class edges.
- **Wave layer (dense fields):** embeddings per node/edge/episode, uncertainty/belief scores, salience + decay curves, and a **context state vector** that mirrors the HOPE wave state for downstream planners.
- **Fusion principle:** every edge is **typed + scored + contextualized** with scope (time range, location, audience/permissions), provenance (episode + step, CMS span), confidence, salience function/decay parameters, and an embedding used for wave-side scoring.

## Retrieval Flow (planner-facing)

1. **Wave prefilter:** embed the query (goal/tool/intent span) and retrieve nearest neighbors from the vector index (node + edge embeddings, optionally episode centroids).
2. **Particle expansion with constraints:** expand the symbolic neighborhood subject to time windows, location/scene tags, permissions/policies, confidence thresholds, and “same-agent/actor” guards.
3. **Wave re-rank + stitch:** re-score the expanded set with embeddings + salience/decay, then stitch a **context subgraph** and an updated **context state vector** handed to the planner (HOPE fast path).

## Schema Sketch (implementable)

Storage: Graph DB (e.g., Neo4j/SQLite with adjacency tables) + vector index (FAISS/SQLite JSON index). Keep IDs stable across RLDS/CMS writes.

```yaml
Node:
  id: "node/<uuid>"
  type: "entity|tool|goal|intent|concept|policy|context_session"
  name: "Long-tail entity or tool name"
  attributes:
    tags: ["location:lab", "agent:robot1"]
    cms_span_ids: ["cms/span/123"]        # maps CMS atoms/spans into the graph
    rlds_episode_ids: ["rlds/episode/abc"]
  embedding: [float, ...]
  belief: {score: 0.82, updated_at: "2025-01-05T12:00:00Z"}

Episode:
  id: "episode/<uuid>"
  start_time: "2025-01-05T12:00:00Z"
  end_time: "2025-01-05T12:10:00Z"
  location: "lab/arm-bench"
  nodes: ["node/entity/arm", "node/context_session/xyz"]
  edges: ["edge/<uuid>", ...]
  embedding: [float, ...]

Edge:
  id: "edge/<uuid>"
  source: "node/<uuid>"
  target: "node/<uuid>"
  type: "causal|temporal|membership|policy|tool_use|assertion"
  scope:
    time_range: ["2025-01-05T12:00:00Z", "2025-01-05T12:01:00Z"]
    location: "lab/arm-bench"
    audience: ["agent:robot1", "role:operator"]
  provenance:
    episode_id: "episode/<uuid>"
    step_index: 42
    cms_span_id: "cms/span/123"
  assertion:
    predicate: "used_tool"
    value: "screwdriver"
  confidence: 0.77
  embedding: [float, ...]
  salience:
    score: 0.64
    decay_fn: "exp"
    half_life_s: 3600
    last_updated: "2025-01-05T12:02:00Z"
  policy:
    allow_roles: ["operator"]
    deny_roles: ["visitor"]

ContextSession:             # anchors the active workspace for planning
  id: "context_session/<uuid>"
  goals: ["node/goal/assemble-arm"]
  tools: ["node/tool/screwdriver", "node/tool/gripper"]
  active_entities: ["node/entity/arm", "node/entity/operator"]
  intents: ["node/intent/fix_joint"]
  context_state_vector: [float, ...]     # wave-compatible state for HOPE planner
  constraints:
    permissions: ["role:operator"]
    time_window: ["2025-01-05T12:00:00Z", "2025-01-05T12:30:00Z"]
```

## Learning & Updates

- **Particle-side:** entity resolution/merging, new edges with provenance, contradiction tracking (conflicting assertions held with confidence + provenance tags), and policy edge tightening when denials occur.
- **Wave-side:** embedding refreshes (batch or online), salience/decay parameter learning per node/edge type, link prediction to propose new candidate edges, and belief/uncertainty updates that gate planner use.
- **Completeness requirement:** every query returns a context subgraph and context state vector, explicitly marking unknowns/hypotheses and citing provenance + confidence for each edge/node.

## Mapping to CMS + HOPE Artifacts

- CMS atoms/spans/episodes map to node attributes (`cms_span_ids`) and provenance on edges; decays mirror CMS decay parameters.
- HOPE wave–particle paths stay synchronized: the context state vector follows the wave path, while symbolic edges feed the particle path. See [wave_particle_rollout.md](./wave_particle_rollout.md) for rollout dynamics.
- CMS memory objects (atoms → spans → episodes → semantic concepts) populate nodes/episodes; the graph provides the bridge from raw CMS writes to planner-ready context.
