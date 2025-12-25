# Specification: Conversation Context Window

## Overview
This track implements server-persistent, multi-turn conversation support for the ContinuonBrain. By maintaining a rolling window of the last 10 turns in a dedicated SQLite database, the robot can understand follow-up questions and maintain context across reboots, providing a more natural and intelligent interaction experience.

## Functional Requirements
- **Session Persistence Layer:**
    - Create a `sessions.db` SQLite database in the `config_dir`.
    - Define a `messages` table: `session_id`, `role` (user/assistant), `content`, and `timestamp`.
- **Contextual Reasoning:**
    - Update `BrainService.ChatWithGemma` to automatically retrieve the last 10 turns for the active `session_id`.
    - Inject this history into the LLM context window to enable follow-up reasoning.
- **Session Management API:**
    - Implement `POST /api/chat/session/clear` to explicitly reset the history for a given session.
    - Ensure the `session_id` is consistently passed from the Brain Studio UI and Flutter app.
- **UI Enhancements (Brain Studio):**
    - **Clear Thread:** Add a button to reset the current conversation.
    - **Context Badges:** Visually indicate "In Thread" messages to show the active context window.
    - **Thought Grouping:** Visually link related thoughts in the "Thought Stream" during multi-turn exchanges.

## Non-Functional Requirements
- **Resource Management:** Strictly cap the history at 10 turns to prevent token overflow and excessive memory usage on the Raspberry Pi 5.
- **Latency:** History retrieval and injection must add <20ms to the total chat processing time.

## Acceptance Criteria
- [ ] Robot correctly handles follow-up questions (e.g., "Tell me about the arm" followed by "How many joints does it have?").
- [ ] Conversation history survives a full server restart.
- [ ] "Clear Thread" button successfully wipes the local history and resets the LLM context.
- [ ] Messages within the 10-turn window are highlighted with context badges in the UI.

## Out of Scope
- Branching conversation threads.
- Searchable archival of historical sessions (only the active 10-turn window is maintained for reasoning).
