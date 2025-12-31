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

## Decision Receipts at Commit Surfaces

Decision receipts capture **write-time provenance** whenever a stateful change is proposed or committed (e.g., code merge, dataset promotion, policy toggle, config rollout). Each receipt records:

- Referenced inputs (episodes, CMS spans, metrics, design docs), linked by stable IDs.
- Policies/constraints evaluated, including allow/deny rules, safety/PII gates, budget caps, and owner/audience scopes.
- Exception paths invoked (temporary waivers, degraded-mode approvals) and the justification for using them.
- Approvals and actors (requester, reviewer, automated checker signatures).
- Action taken (apply/rollback/defer) and the observed outcome (success/failure, residual risk, follow-up tasks).

Receipts are treated as **particle-side provenance** that is immediately dual-written into the wave layer:

- Particle: `DecisionReceipt` nodes with typed edges to referenced evidence (episodes, spans, policies, config versions), actors, approvals, actions, and outcomes. Policy edges carry the evaluated scope and the result (allow/deny).
- Wave: embeddings over the whole receipt (and per-edge when available), confidence sourced from check outcomes, salience + decay tuned to the task window (e.g., promotion freezes decay slower than experimental toggles). Receipts contribute to wave prefilters as provenance anchors for later queries.

### Bottom-Up Ontology Stance

- **Bottom-up (preferred):** start from captured traces/receipts, then infer minimal entities/relations that repeatedly appear in similar trajectories (e.g., “pi5_hailo_promotion” receipts repeatedly connect `policy:pii_cleared` → `action:promote` → `outcome:success`). Keep the schema thin and let recurrence drive new node/edge types.
- **Top-down (contrast):** fixed, prescriptive taxonomy (Palantir-style) that predefines entities/relations before observing traces. This is avoided for Continuon Brain runtime scaffolding to prevent premature lock-in; only promote new types after receipts show recurring structure.

### Receipt Schema and Graph Ingestion (lightweight)

```yaml
DecisionReceipt:
  id: "receipt/<uuid>"
  task: "deploy_pi5_wavecore_loop"
  actor: "user/ops1"
  approvals: ["user/reviewer_a", "check/ci_suite@v2"]
  referenced_inputs:
    episodes: ["rlds/episode/abc"]
    cms_spans: ["cms/span/123"]
    configs: ["config/wavecore/v1.2"]
  policies_checked: [{id: "policy/pii_cleared", result: "allow"}, {id: "policy/budget", result: "allow"}]
  exceptions: [{id: "waiver/temp_net_degradation", justification: "lab wifi outage"}]
  action: "promote_candidate"
  outcome: {status: "success", notes: "latency stable", follow_ups: ["validate on-device drift"]}
  timestamps: {requested: "2025-01-05T12:00:00Z", committed: "2025-01-05T12:07:00Z"}
```

Ingestion into the dual-layer graph:

- Create a `DecisionReceipt` node with an embedding over the concatenated fields; attach salience/decay based on scope (`task`, `audience`, and time window).
- Add edges to evidence (`ref_input` → episode/span/config), actors/approvers, applicable policies (typed `policy_result` edges), actions, and outcomes. Each edge carries provenance to the receipt ID and the original source (e.g., CI log URL, CMS span ID).
- When the receipt references CMS spans or RLDS episodes, reuse their embeddings for the wave layer and add a composed embedding for the receipt to improve prefilter recall.

### Retrieval Hooks

- Query receipts by task/user/time/permission (e.g., `task=deploy_pi5_wavecore_loop AND audience~role:operator AND committed_within=7d`) to assemble a policy-aware history.
- Use the wave prefilter on receipt embeddings to find similar decisions, then expand particle edges to surface which inputs/policies drove past approvals or denials.
- For active planning, include recent receipts in the context state vector so the planner respects the latest approvals/denials without re-running heavy checks.

### Implementation Guardrails

- Keep instrumentation minimal: emit receipts at commit surfaces only, and reuse existing IDs/logs instead of rebuilding the world from scratch.
- Respect PII/safety constraints: do not ingest or list receipts unless associated episodes/spans have `pii_cleared=true`, `pending_review=false`, and redactions noted (`pii_redacted=true` when applicable).
- Stage evolution: **v1** structured logging only; **v2** promote into the dual-layer graph with embeddings/decay; **v3** enable learned relation suggestions (link prediction) gated by policy edges and human approval.
