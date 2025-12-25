# Implementation Plan: Memory Consolidation & Decay

This plan implements background quality management for the robot's long-term memory, ensuring high-fidelity recall and preventing data bloat.

## Phase 1: Consolidation Logic (Synthesis)
Implement the core grouping and merging algorithms.

- [x] **Task: Semantic Grouping Engine**
  - **TDD:** Write unit tests in `tests/test_memory_merging.py` for similarity-based grouping.
  - Implement a `cluster_similar_memories` method in `ExperienceLogger` that returns lists of redundant entries.
- [x] **Task: Answer Synthesis Implementation**
  - Implement a `synthesize_memory_anchor` method that uses `BrainService.gemma_chat` to merge answers.
  - Define the system prompt for the "Synthesizer" persona.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Consolidation' (Protocol in workflow.md)

## Phase 2: Dynamic Confidence Decay
Implement the time-based and event-based confidence penalty system.

- [x] **Task: Decay Algorithm Updates**
  - Update `ExperienceLogger.apply_confidence_decay` to support the "Model Evolution" penalty.
  - Implement the "Access Frequency" check using the existing `last_accessed` timestamp.
- [x] **Task: Status-based Immunity**
  - Ensure `is_validated` check is strictly enforced before applying any decay.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Decay' (Protocol in workflow.md)

## Phase 3: Orchestration & Automation
Wire the background tasks into the autonomy lifecycle.

- [x] **Task: Threshold Gating**
  - Add logic to `BrainService` to monitor memory counts and trigger consolidation when thresholds are hit.
- [x] **Task: Autonomy Orchestrator Hook**
  - Register consolidation and decay as jobs in `continuonbrain/services/brain_service.py` (Orchestrator loop).
  - Add resource gating to prevent consolidation during high-load periods.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Orchestration' (Protocol in workflow.md)

## Phase 4: Verification & Polish
Validate the system under simulated load.

- [x] **Task: High-Redundancy Stress Test**
  - Create a script to generate 100+ semantically similar "Mock" memories.
  - Verify that the orchestrator triggers consolidation and reduces the count to <20 high-quality entries.
- [x] **Task: Model Promotion Event Test**
  - Simulate a model promotion and verify the 10% penalty is applied correctly to the target subset.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Final Validation' (Protocol in workflow.md)
