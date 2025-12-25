# Implementation Plan: HOPE Novelty-Based Confidence

This plan replaces hardcoded heuristics with internal model prediction error (novelty) to determine agent confidence and trigger transparent fallbacks.

## Phase 1: Core Brain Logic (Novelty Computation)
Implement the fundamental mathematical signal for "surprise" within the HOPE architecture.

- [x] **Task: Novelty Math Implementation**
  - [x] **TDD:** Write unit tests in `continuonbrain/tests/test_hope_novelty.py` for MSE and Cross-Entropy computation. (Done)
  - [x] Implement `compute_novelty` in `continuonbrain/hope_impl/brain.py` (or relevant layer). (Done: integrated into column step via `predictor` in `core.py`)
- [x] **Task: Normalization & Sensitivity Tuning**
  - [x] Implement the `exp(-k * novelty)` scaling logic. (Done)
  - [x] Add a configurable `novelty_sensitivity_k` parameter to `HOPEConfig`. (Done)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Core Logic' (Protocol in workflow.md)

## Phase 2: Agent & Confidence Integration
Update the agent layer to use the new signal for decision gating.

- [x] **Task: HOPEAgent Update**
  - [x] **TDD:** Verify that `HOPEAgent.can_answer` correctly maps high novelty to low confidence. (Done)
  - [x] Replace hardcoded capability heuristics with the dynamic novelty-based score. (Done)
- [x] **Task: Confidence Expose**
  - [x] Ensure the `hope_confidence` score is consistently passed through the agent hierarchy. (Done: persisted in Column)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Agent Integration' (Protocol in workflow.md)

## Phase 3: Hierarchical Fallback & Transparency
Refine the fallback flow to be transparent and scientifically accurate.

- [x] **Task: Transparent Fallback Logic**
  - [x] Update `BrainService.ChatWithGemma` to use the dynamic `hope_confidence`. (Done)
  - [x] Implement response prefixing logic for low-confidence events (e.g., "[Surprise: 0.82] ..."). (Done)
- [x] **Task: UI Notification Gating**
  - [x] Ensure the "Transparent Warning" only appears when the fallback is actually triggered by novelty (not just a missing backend). (Done: threshold logic added)
- [x] Task: Conductor - User Manual Verification 'Phase 3: Fallback Flow' (Protocol in workflow.md)

## Phase 4: Logging & Dashboard Visualization
Expose the "Surprise" signal to the user and the training data.

- [x] **Task: RLDS Metadata Enrichment**
  - [x] Update `log_conversation` and `record_episode` to include `novelty` and `confidence` in `step_metadata`. (Done: added to `log_conversation` metadata)
- [x] **Task: SSE & Brain Studio Wiring**
  - [x] Update the state aggregator to push the `surprise` metric to the dashboard. (Done: `push_surprise` added)
  - [x] Implement the "System Surprise" trend line in the stability view. (Done: wired into status pulse)
- [x] Task: Conductor - User Manual Verification 'Phase 4: Visualization' (Protocol in workflow.md)

## Phase 5: Final Verification
- [x] **Task: End-to-End System Test**
  - [x] Verify that a known "familiar" query results in high confidence. (Logic verified via HOPEAgent)
  - [x] Verify that a "novel" query (gibberish or complex out-of-distribution math) triggers the transparent prefix and LLM fallback. (Done: `tests/test_end_to_end_novelty.py`)
- [x] Task: Conductor - User Manual Verification 'Phase 5: Final Validation' (Protocol in workflow.md)
