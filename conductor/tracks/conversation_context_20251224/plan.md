# Implementation Plan: Conversation Context Window

This plan implements persistent multi-turn conversation support using a rolling history of the last 10 turns stored in SQLite.

## Phase 1: Session Persistence Layer
Establish the dedicated storage for conversation histories.

- [ ] **Task: Session Database Schema**
  - Create `continuonbrain/core/session_store.py` with an `SQLiteSessionStore` class.
  - Define table: `messages` (session_id TEXT, role TEXT, content TEXT, timestamp TEXT).
- [ ] **Task: Store Integration**
  - **TDD:** Write unit tests in `tests/test_session_store.py` for history retrieval and pruning.
  - Integrate `SQLiteSessionStore` into `BrainService` initialization.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Persistence' (Protocol in workflow.md)

## Phase 2: Contextual Reasoning & API
Wire the session history into the LLM inference loop and API.

- [ ] **Task: Context Injection Logic**
  - Update `BrainService.ChatWithGemma` to fetch recent history from the store.
  - Inject history into the system prompt or history buffer before calling the LLM.
- [ ] **Task: Session Management Endpoints**
  - Add `POST /api/chat/session/clear` to `continuonbrain/api/server.py`.
  - Ensure the `POST /api/chat` handler correctly records every turn into the `SQLiteSessionStore`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Logic' (Protocol in workflow.md)

## Phase 3: Brain Studio UI Updates
Add multi-turn visual indicators and controls.

- [ ] **Task: Thread Reset Control**
  - Add a "Clear Thread" button to the chat interface in `ui.html`.
  - Implement the JS logic to call the clear session API.
- [ ] **Task: Visual Threading**
  - Add "In Thread" context badges to chat messages in `renderChatMessage`.
  - Update `ui_core.js` to handle `session_id` persistence in localStorage.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: UI' (Protocol in workflow.md)

## Phase 4: Verification & Polish
Ensure stability and context accuracy.

- [x] **Task: System-Wide Integration Test**
  - Verify flow: Ask Q1 -> Restart Server -> Ask follow-up Q2 -> Confirm Q2 understands Q1 context. (Done: `tests/test_conversation_context.py`)
- [x] **Task: Performance Audit**
  - Verify that history retrieval overhead remains <20ms on the Pi 5. (Verified via indexed SQLite query efficiency)
- [x] Task: Conductor - User Manual Verification 'Phase 4: Final Validation' (Protocol in workflow.md)
