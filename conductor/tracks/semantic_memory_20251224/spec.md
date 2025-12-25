# Specification: Semantic Memory Retrieval & Deduplication

## Overview
This track upgrades the `ExperienceLogger` from basic keyword matching to semantic search using the `all-MiniLM-L6-v2` transformer model. It also introduces active deduplication to keep the robot's learned knowledge clean and manageable, and integrates semantic relevance directly into the Brain's decision-making and visualization layers.

## Functional Requirements
- **Semantic Encoding:**
    - Integrate `sentence-transformers` into the `ExperienceLogger`.
    - Download and cache the `all-MiniLM-L6-v2` model (80MB) locally.
    - Implement a `LazyEncoder` pattern to ensure the model is only loaded into memory when first needed.
- **Active Gating & Deduplication:**
    - Implement semantic similarity checks *before* storing new conversations.
    - If a new query matches an existing memory with >0.95 similarity, increment the existing memory's `hit_count` and update its `last_accessed` timestamp instead of creating a new file.
- **Enhanced Retrieval:**
    - Replace Jaccard similarity with Cosine Similarity for memory retrieval.
    - Implement a "Semantic Neighborhood" query that returns the top-K semantically related memories.
- **Decision Integration:**
    - Expose the semantic similarity score as a `relevance_confidence` metric.
    - Wire this score into the hierarchical fallback logic: if a highly relevant memory is found, increase the priority of the HOPE/Memory response over a fresh LLM call.
- **Brain Studio Visualization:**
    - Add a "Knowledge Map" panel to the dashboard that visualizes learned memories as semantic clusters.
    - Automatically assign category tags (e.g., "Safety", "Motion") based on embedding proximity to predefined category centroids.

## Non-Functional Requirements
- **Hardware Efficiency:** Model loading and inference must respect the Pi 5's memory limits.
- **Latency:** Semantic retrieval (encoding + search) must complete in <150ms to maintain real-time responsiveness.
- **Persistence:** Ensure embeddings are stored alongside raw JSON conversations to avoid re-encoding on every reboot.

## Acceptance Criteria
- [ ] Robot successfully answers "How do I move?" using a memory learned from "Explain the motion process."
- [ ] Identical or near-identical queries (95%+ similarity) do not result in duplicate files in the `experiences/` directory.
- [ ] Brain Studio displays a clustered map of learned knowledge with automatic topic tagging.
- [ ] Hierarchical fallback correctly prioritizes semantically relevant memories.

## Out of Scope
- Multi-modal embeddings (vision/audio).
- Fine-tuning the embedding model on the edge.
