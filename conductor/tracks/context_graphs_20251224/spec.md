# Specification: Context Graphs Implementation

## Overview
Implement the "Context Graphs" module within the `continuonbrain` to act as a Wave–Particle bridge for situation-aware memory. This module will fuse symbolic graph structures (Particle layer) with dense embeddings and decay-aware state (Wave layer), aligning with the HOPE/CMS architecture.

## Functional Requirements
- **Dual-Layer Graph Storage:**
    - Primary implementation using **SQLite** for persistence (adjacency tables for nodes/edges).
    - Support for an **In-Memory persistence mode** (e.g., NetworkX + lightweight vector store) to facilitate fast experimentation.
- **Situation-Aware Retrieval:**
    - Implement the "Wave prefilter" (vector search), "Particle expansion" (symbolic traversal with constraints), and "Wave re-rank" (salience/decay scoring) flow.
- **Data Ingestion:**
    - Implement listeners to automatically populate the graph from incoming **RLDS episodes** and **CMS events/spans**.
- **System Integration:**
    - **API Integration:** Expose graph retrieval and visualization data via new endpoints in `robot_api_server.py`.
    - **Planner Hook:** Integrate the retrieval flow into the **HOPE planner** to provide context subgraphs and state vectors during decision-making.
- **Embedding Generation:**
    - Primary: Reuse the existing **Gemma/LiteRT** Local LLM for vector embeddings.
    - Secondary: Fallback to a lightweight model (e.g., `all-MiniLM-L6-v2`) if resources are constrained.
- **User Interface:**
    - Main: Integrate controls and visualization into the **Web-based Debug Dashboard**.
    - Optional: Expose simpler UI hooks for the **Flutter Companion App**.

## Non-Functional Requirements
- **Offline-First:** Ensure all storage and embedding generation runs locally on the Raspberry Pi 5.
- **Performance:** Retrieval flow must be optimized to meet HOPE "Mid-Loop" timing requirements (1-10s range).
- **Extensibility:** The schema must support typed edges with provenance, confidence, and salience attributes as defined in `context_graphs.md`.

## Acceptance Criteria
- [ ] ContextGraph module initialized with SQLite/In-memory storage options.
- [ ] Graph successfully ingests RLDS/CMS data into typed nodes/edges.
- [ ] Vector search retrieves relevant nodes based on query embeddings.
- [ ] HOPE planner receives a valid "Context Subgraph" and "Context State Vector".
- [ ] Debug Dashboard displays a visual representation of the current context graph.

## Out of Scope
- Large-scale graph database deployment (e.g., Neo4j).
- Real-time "Fast-Loop" (50ms) graph updates (focusing on Mid/Slow loop integration).
