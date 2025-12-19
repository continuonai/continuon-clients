# Plan: Consolidate Brain Runtime and Establish HOPE Seed Training Pipeline

## Phase 1: Audit and Pruning
- [ ] Task: Audit `continuonbrain` directory for redundant server scripts.
    - [ ] Subtask: List all files in `continuonbrain` and identify multiple entry points (e.g., `server.py`, `robot_api_server.py`).
    - [ ] Subtask: Determine the canonical server script and mark others for deletion.
- [ ] Task: Audit `continuonbrain` for redundant training scripts.
    - [ ] Subtask: Identify overlapping training logic in `trainer/`, `scripts/`, and root level.
    - [ ] Subtask: Mark deprecated training scripts for deletion.
- [ ] Task: Remove identified redundant files.
    - [ ] Subtask: Delete the files marked in previous steps.
    - [ ] Subtask: Verify that the project still builds/runs (if possible) or that no critical dependencies were broken.
- [ ] Task: Conductor - User Manual Verification 'Audit and Pruning' (Protocol in workflow.md)

## Phase 2: Runtime Unification
- [ ] Task: Define the Canonical Runtime Entry Point.
    - [ ] Subtask: Create or refactor a single `startup.py` (or equivalent) that initializes HAL, Robot API, and HOPE loops.
    - [ ] Subtask: Ensure it correctly detects hardware (Pi 5 vs. Mock).
- [ ] Task: Consolidate Hardware Abstraction Layer (HAL) initialization.
    - [ ] Subtask: Ensure all sensor/actuator startup logic is centralized.
- [ ] Task: Conductor - User Manual Verification 'Runtime Unification' (Protocol in workflow.md)

## Phase 3: HOPE Seed Training Pipeline
- [ ] Task: Establish Canonical HOPE Training Script.
    - [ ] Subtask: Refactor `trainer/` to have a single, clear entry point for training the seed model (Fast/Mid loops).
    - [ ] Subtask: Ensure it consumes RLDS data correctly.
- [ ] Task: Verify JAX/Mamba integration.
    - [ ] Subtask: Check that the training script correctly utilizes JAX and Mamba components as defined in the Tech Stack.
- [ ] Task: Conductor - User Manual Verification 'HOPE Seed Training Pipeline' (Protocol in workflow.md)

## Phase 4: Documentation
- [ ] Task: Update `continuonbrain/README.md`.
    - [ ] Subtask: specific instructions on how to run the unified runtime and training pipeline.
- [ ] Task: Conductor - User Manual Verification 'Documentation' (Protocol in workflow.md)
