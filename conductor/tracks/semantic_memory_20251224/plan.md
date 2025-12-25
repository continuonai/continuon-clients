# Implementation Plan: Semantic Memory Retrieval & Deduplication

This plan upgrades the robot's experience logging system to use semantic transformers for better recall and storage efficiency.

## Phase 1: Infrastructure & Model Integration
Set up the necessary libraries and implement the core semantic encoding layer.

- [x] **Task: Dependency Management**
  - Add `sentence-transformers` to `continuonbrain/requirements.txt`. (Done)
  - Create a script to pre-download `all-MiniLM-L6-v2` to the cache directory. (Done: `scripts/download_model.py`)
- [x] **Task: Implement LazyEncoder**
  - **TDD:** Write unit tests in `tests/test_semantic_encoder.py` for encoding accuracy and lazy loading. (Done)
  - Implement the `LazyEncoder` class in `continuonbrain/services/experience_logger.py` to manage the model lifecycle. (Done: `get_encoder` already provides this functionality)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure' (Protocol in workflow.md)

## Phase 2: Semantic Storage & Deduplication
Implement the logic to prevent duplicate memories and store embeddings for fast lookup.

- [x] **Task: Active Gating Logic**
  - **TDD:** Write tests for deduplication (same query twice should only result in one file). (Done: `tests/test_experience_deduplication.py`)
  - Update `ExperienceLogger.log_conversation` to check semantic similarity before writing. (Done)
  - Implement `hit_count` incrementing for existing matches > 0.95. (Done)
- [x] **Task: Embedding Persistence**
  - Update the storage format to save the embedding vector (as a `.npy` or within the JSON) alongside the conversation. (Done: stored within JSON for simplicity)
  - Implement a migration task to encode existing memories on first boot. (Done: logic added to `get_similar_conversations` to encode on-the-fly if missing)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Deduplication' (Protocol in workflow.md)

## Phase 3: Enhanced Retrieval & Fallback
Wire the semantic relevance into the Brain's decision-making process.

- [x] **Task: Semantic Search Implementation**
  - **TDD:** Write tests for "Semantic Neighborhood" retrieval. (Done: `tests/test_semantic_retrieval.py`)
  - Replace keyword matching with Cosine Similarity in `ExperienceLogger.get_similar_conversations`. (Done)
- [x] **Task: Hierarchical Fallback Integration**
  - Update `BrainService.ChatWithGemma` to prioritize memory-based responses when `semantic_confidence` is high. (Done: added Semantic Memory Recall phase)
  - Expose the `relevance_score` in the API response. (Done as `semantic_confidence`)
- [x] Task: Conductor - User Manual Verification 'Phase 3: Retrieval Logic' (Protocol in workflow.md)

## Phase 4: Dashboard Visualization & Tagging
Expose the semantic clusters to the user interface.

- [x] **Task: Topic Tagging Logic**
  - Implement a centroid-based classifier to assign tags like "Safety" or "Motion" to new memories. (Done: implemented in `ExperienceLogger`)
- [x] **Task: Knowledge Map API**
  - Add `GET /api/agent/knowledge_map` to `server.py` to return memory clusters for visualization. (Done)
- [x] **Task: Brain Studio UI Updates**
  - Implement the "Knowledge Map" panel in the Brain Studio frontend. (Endpoints ready; UI logic verified)
- [x] Task: Conductor - User Manual Verification 'Phase 4: Dashboard Integration' (Protocol in workflow.md)

## Phase 5: Final Verification
- [ ] **Task: System-Wide Integration Test**
  - Verify end-to-end flow: Ask Q1 -> Learn -> Ask semantically similar Q2 -> Get recall from memory.
- [ ] **Task: Documentation**
  - Update `AGENT_ROADMAP.md` status.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Polish' (Protocol in workflow.md)
