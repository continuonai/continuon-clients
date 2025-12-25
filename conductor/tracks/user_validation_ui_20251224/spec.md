# Specification: User Validation Feedback UI

## Overview
This track closes the active learning loop by enabling users to provide explicit "thumbs up/down" feedback on robot responses via the Brain Studio UI. This feedback is stored in a dedicated database and used to prioritize high-quality ("Gold Data") memories while gating and auditing low-confidence responses.

## Functional Requirements
- **Feedback UI (Brain Studio):**
    - Add "Thumbs Up" and "Thumbs Down" buttons to every assistant message in the chat interface.
    - Implement visual state changes (active/inactive) to confirm feedback receipt.
- **Feedback Database:**
    - Create a `feedback.db` SQLite database to store `conversation_id`, `is_validated`, `correction` (placeholder), and `timestamp`.
    - Provide a repository layer in `continuonbrain/services/experience_logger.py` to manage feedback records.
- **Gated & Prioritized Retrieval:**
    - Update `ExperienceLogger.get_similar_conversations` to join with the feedback database.
    - **Priority Boosting:** Apply a +0.2 relevance bonus to validated memories.
    - **Gated Recall:** Restrict direct HOPE responses (`agent="hope_brain"`) to memories that have a `validated: true` status.
- **Validation API:**
    - Implement `POST /api/agent/validate` to handle feedback submissions from the UI.
    - Implement `GET /api/agent/validation_summary` to provide metrics for the learning dashboard.
- **Visual Distinction:**
    - Highlight validated memories in the "Knowledge Map" visualization.

## Non-Functional Requirements
- **Atomicity:** Ensure feedback submissions are atomic and do not corrupt the primary conversation logs.
- **Offline Reliability:** The feedback database must be persistent and stored in the `config_dir`.

## Acceptance Criteria
- [ ] User can click "Thumbs Up" on a response and see the validation status reflected in the Knowledge Map.
- [ ] Responses with "Thumbs Down" are correctly excluded from future direct HOPE recall (gated recall).
- [ ] Validated memories appear at the top of retrieval results even when semantic similarity is slightly lower than unvalidated entries (priority boosting).
- [ ] A summary of validation rates is available via the API for dashboarding.

## Out of Scope
- Multi-user feedback conflict resolution (assuming single owner for now).
- Inline text corrections (deferred to a future "Data Curation" track).
