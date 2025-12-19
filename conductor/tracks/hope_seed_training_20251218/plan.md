# Plan: HOPE v1 Seed Model Training on Pi 5 (8GB)

## Phase 1: Alignment and Data Partitioning
- [ ] Task: Audit RLDS schema alignment with Cloud/Flutter.
    - [ ] Subtask: Review `docs/rlds-schema.md` and verify compatibility with `continuonai/` (Flutter) and Cloud ingestion requirements.
    - [ ] Subtask: Update `rlds/` utilities if discrepancies are found.
- [ ] Task: Implement multi-scale windowing loaders.
    - [ ] Subtask: Implement `FastWindowDataset` (50-100ms) and `MidTrajectoryDataset` (1-10s).
    - [ ] Subtask: Add unit tests to verify correct slicing.
- [ ] Task: Conductor - User Manual Verification 'Alignment and Data Partitioning' (Protocol in workflow.md)

## Phase 2: Sequential Training Orchestrator
- [ ] Task: Create `SequentialTrainer` in `continuonbrain/trainer/`.
    - [ ] Subtask: Implement the load-train-save-unload lifecycle for individual components.
    - [ ] Subtask: Add memory monitoring and safety guards for the 8GB limit.
    - [ ] Subtask: Integrate `OfflineSearchLoader` for "thought" episodes.
- [ ] Task: Conductor - User Manual Verification 'Sequential Training' (Protocol in workflow.md)

## Phase 3: Loop Closure and API Integration
- [ ] Task: Update `run_trainer.py` to support sequential mode.
- [ ] Task: Implement Hot-Reload in `BrainService`.
    - [ ] Subtask: Allow the Brain Runtime to swap model weights from candidate checkpoints without restart.
- [ ] Task: Expose API and Web UI controls.
    - [ ] Subtask: Add `POST /api/training/sequential` to `continuonbrain/api/server.py`.
    - [ ] Subtask: Update the Robot Editor (Web UI) to show real-time training progress and reflex metrics.
- [ ] Task: Conductor - User Manual Verification 'Loop Closure and API Integration' (Protocol in workflow.md)

## Phase 4: End-to-End Validation
- [ ] Task: Perform "Seed" training cycle on real Pi 5 hardware.
- [ ] Task: Demonstrate functional reflex/inference via API and Web Browser.
- [ ] Task: Document memory, thermal, and convergence metrics.
- [ ] Task: Conductor - User Manual Verification 'End-to-End Validation' (Protocol in workflow.md)
