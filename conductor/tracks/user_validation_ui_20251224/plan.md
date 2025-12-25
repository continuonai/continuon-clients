# Implementation Plan: User Validation Feedback UI

This plan implements a user-driven feedback loop to identify and prioritize high-quality learned memories ("Gold Data").

## Phase 1: Feedback Persistence Layer
Establish the dedicated storage for user validations.

- [x] **Task: Feedback Database Schema**
  - Create `continuonbrain/core/feedback_store.py` with an `SQLiteFeedbackStore` class. (Done)
  - Define table: `feedback` (conversation_id TEXT PRIMARY KEY, is_validated BOOLEAN, correction TEXT, timestamp TEXT). (Done)
- [x] **Task: Repository Integration**
  - [x] **TDD:** Write unit tests in `tests/test_feedback_store.py` for CRUD operations. (Done)
  - [x] Integrate `SQLiteFeedbackStore` into `ExperienceLogger` initialization. (Done)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Persistence' (Protocol in workflow.md)

## Phase 2: Validation API & Server Integration
Expose endpoints for the UI to submit and query validation status.

- [x] **Task: Implement Validation Endpoints**
  - [x] Add `POST /api/agent/validate` to `continuonbrain/api/server.py`. (Done)
  - [x] Add `GET /api/agent/validation_summary` to provide aggregate metrics (e.g., % validated). (Done)
- [x] **Task: Update ExperienceLogger Methods**
  - [x] **TDD:** Verify that `validate_conversation` correctly updates the SQLite store. (Done)
  - [x] Ensure `ExperienceLogger.get_statistics` includes validation rates from the database. (Done)
- [x] Task: Conductor - User Manual Verification 'Phase 2: API' (Protocol in workflow.md)

## Phase 3: Prioritized & Gated Retrieval
Update the hierarchical response logic to respect user feedback.

- [x] **Task: Gated Recall Logic**
  - [x] **TDD:** Write tests in `tests/test_gated_recall.py` ensuring unvalidated or rejected memories are never used for `hope_brain` direct answers. (Done)
  - [x] Update `HOPEAgent.can_answer` to check the validation status of retrieved memories. (Note: implemented in BrainService.ChatWithGemma gating instead)
- [x] **Task: Priority Boosting Implementation**
  - [x] Update `ExperienceLogger.get_similar_conversations` to apply a +0.2 relevance bonus to memories where `is_validated` is True. (Done)
  - [x] Ensure the final sort order respects this bonus. (Done)
- [x] Task: Conductor - User Manual Verification 'Phase 3: Retrieval' (Protocol in workflow.md)

## Phase 4: Brain Studio UI Updates
Add the interactive elements to the dashboard.

- [x] **Task: Chat Feedback Buttons**
  - [x] Update the Brain Studio chat template to include "Thumbs Up/Down" buttons on assistant messages. (Done: implemented in `renderChatMessage`)
  - [x] Implement the JS logic to call `POST /api/agent/validate` on click. (Done: `validateResponse` added)
- [x] **Task: Knowledge Map Visual Distinction**
  - [x] Update the `GET /api/agent/knowledge_map` response to include validation status. (Done: updated `ExperienceLogger.search_conversations`)
  - [x] Modify the dashboard visualization to highlight "Gold Data" nodes (e.g., using a different color or border). (Done: updated `context_graph.html`)
- [x] Task: Conductor - User Manual Verification 'Phase 4: UI' (Protocol in workflow.md)

## Phase 5: Final Verification
- [x] **Task: System-Wide Integration Test**
  - [x] Verify flow: Ask Q -> Get LLM Answer -> Thumbs Up -> Ask similar Q -> Get direct HOPE recall. (Done: `tests/test_validation_feedback_flow.py`)
  - [x] Verify flow: Ask Q -> Get Answer -> Thumbs Down -> Verify Q is never recalled as a direct HOPE answer. (Done)
- [x] Task: Conductor - User Manual Verification 'Phase 5: Final Validation' (Protocol in workflow.md)
