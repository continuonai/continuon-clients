# Plan: HOPE v1 Seed Model Training on Pi 5 (8GB)

## Phase 1: Alignment and Data Partitioning
- [x] Task: Audit RLDS schema alignment with Cloud/Flutter.
    - [x] Subtask: Review `docs/rlds-schema.md` and verify compatibility with `continuonai/` (Flutter) and Cloud ingestion requirements.
    - [x] Subtask: Update `rlds/` utilities if discrepancies are found.
- [x] Task: Implement multi-scale windowing loaders.
    - [x] Subtask: Implement `FastWindowDataset` (50-100ms) and `MidTrajectoryDataset` (1-10s).
    - [x] Subtask: Add unit tests to verify correct slicing.
- [x] Task: Conductor - User Manual Verification 'Alignment and Data Partitioning' (Protocol in workflow.md)

## Phase 2: Sequential Training Orchestrator
- [x] Task: Create `SequentialTrainer` in `continuonbrain/trainer/`.
    - [x] Subtask: Implement the load-train-save-unload lifecycle for individual components.
    - [x] Subtask: Add memory monitoring and safety guards for the 8GB limit.
    - [x] Subtask: Integrate `OfflineSearchLoader` for "thought" episodes.
- [x] Task: Conductor - User Manual Verification 'Sequential Training' (Protocol in workflow.md)

## Phase 3: Loop Closure and API Integration
- [x] Task: Update `run_trainer.py` to support sequential mode.
- [x] Task: Implement Hot-Reload in `BrainService`.
    - [x] Subtask: Allow the Brain Runtime to swap model weights from candidate checkpoints without restart.
- [x] Task: Expose API and Web UI controls.
    - [x] Subtask: Add `POST /api/training/sequential` to `continuonbrain/api/server.py`.
    - [x] Subtask: Update the Robot Editor (Web UI) to show real-time training progress and reflex metrics.
- [x] Task: Conductor - User Manual Verification 'Loop Closure and API Integration' (Protocol in workflow.md)

## Phase 4: End-to-End Validation
- [x] Task: Perform "Seed" training cycle on real Pi 5 hardware.
- [x] Task: Demonstrate functional reflex/inference via API and Web Browser.
- [x] Task: Document memory, thermal, and convergence metrics.
- [x] Task: Conductor - User Manual Verification 'End-to-End Validation' (Protocol in workflow.md)
