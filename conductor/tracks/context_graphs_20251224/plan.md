# Implementation Plan: Context Graphs

This plan follows the established TDD-driven workflow, prioritizing local persistence with SQLite while allowing for in-memory experimentation.

## Phase 1: Core Module & Schema (SQLite Storage)
Implement the fundamental data structures and persistence layer for the Context Graph.

- [x] **Task: Define Core Data Models**
  - Create `continuonbrain/core/context_graph_models.py` with Pydantic/dataclass models for `Node`, `Edge`, `Episode`, and `ContextSession`.
- [x] **Task: Implement SQLite Storage Engine**
  - Write tests for `SQLiteContextStore` in `continuonbrain/tests/test_context_store.py`.
  - Implement the store in `continuonbrain/core/context_graph_store.py` with tables for nodes, edges, and vector metadata.
- [x] **Task: In-Memory Bridge (Experimentation)**
  - Implement a `NetworkXContextStore` that can swap with the SQLite engine for high-speed experimentation.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Core Module & Schema (SQLite Storage)' (Protocol in workflow.md)

## Phase 2: Wave & Particle Retrieval Logic
Implement the fusion retrieval flow (Wave prefilter + Particle expansion).

- [x] **Task: Vector Search Integration**
  - Integrate with the existing LiteRT/Gemma embedding pipeline to generate vectors for nodes.
  - Implement a basic vector search (K-NN) within the SQLite store.
- [x] **Task: Symbolic Expansion (Particle Layer)**
  - Write tests for graph traversal with constraints (time, confidence, type).
  - Implement depth-limited traversal in `continuonbrain/core/context_retriever.py`.
- [x] **Task: Salience & Decay Re-ranking**
  - Implement the decay functions (exponential/linear) for edge scoring based on CMS timestamps.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Wave & Particle Retrieval Logic' (Protocol in workflow.md)

## Phase 3: HOPE & CMS Integration
Connect the graph to live data streams and the decision planner.

- [x] **Task: CMS/RLDS Listeners**
  - Create a `GraphIngestor` that subscribes to CMS span creation and RLDS episode finalization.
  - Map CMS atoms to graph nodes and causal links to edges.
- [x] **Task: HOPE Planner Hook**
  - Update `continuonbrain/hope_impl` to call the `ContextRetriever` during the mid-loop planning phase.
  - Verify the planner receives a valid `ContextStateVector`.
- [x] Task: Conductor - User Manual Verification 'Phase 3: HOPE & CMS Integration' (Protocol in workflow.md)

## Phase 4: Visualization & Debug API
Expose the graph state to the Web Dashboard.

- [x] **Task: Robot API Endpoints**
  - Add `/context/graph` and `/context/search` endpoints to `robot_api_server.py`.
- [x] **Task: Web Dashboard Visualization**
  - Create a simple D3.js or similar visualization in the `studio_server` / debug dashboard templates to render the current context session.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Visualization & Debug API' (Protocol in workflow.md)
