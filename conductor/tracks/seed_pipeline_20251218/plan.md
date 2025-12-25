# Plan: Consolidate Brain Runtime and Establish HOPE Seed Training Pipeline

## Phase 1: Audit and Pruning
- [x] Task: Audit `continuonbrain` directory for redundant server scripts.
    - [x] Subtask: List all files in `continuonbrain` and identify multiple entry points (e.g., `server.py`, `robot_api_server.py`).
    - [x] Subtask: Determine the canonical server script and mark others for deletion.
- [x] Task: Audit `continuonbrain` for redundant training scripts.
    - [x] Subtask: Identify overlapping training logic in `trainer/`, `scripts/`, and root level.
    - [x] Subtask: Mark deprecated training scripts for deletion.
- [x] Task: Remove identified redundant files.
    - [x] Subtask: Delete the files marked in previous steps. (Files cleared with deprecation notices)
    - [x] Subtask: Verify that the project still builds/runs (if possible) or that no critical dependencies were broken.
- [x] Task: Conductor - User Manual Verification 'Audit and Pruning' (Protocol in workflow.md)

## Phase 2: Runtime Unification
- [x] Task: Define the Canonical Runtime Entry Point.
    - [x] Subtask: Create or refactor a single `startup.py` (or equivalent) that initializes HAL, Robot API, and HOPE loops. (Done: `startup_manager.py` is the canonical entry point)
    - [x] Subtask: Ensure it correctly detects hardware (Pi 5 vs. Mock). (Done: HardwareDetector used in services)
- [x] Task: Consolidate Hardware Abstraction Layer (HAL) initialization.
    - [x] Subtask: Ensure all sensor/actuator startup logic is centralized. (Done: `BrainService` centralizes HAL via `ArmEpisodeRecorder`)
- [x] Task: Conductor - User Manual Verification 'Runtime Unification' (Protocol in workflow.md)

## Phase 3: HOPE Seed Training Pipeline
- [x] Task: Establish Canonical HOPE Training Script.
    - [x] Subtask: Refactor `trainer/` to have a single, clear entry point for training the seed model (Fast/Mid loops). (Done: `run_trainer.py` is the unified script)
    - [x] Subtask: Ensure it consumes RLDS data correctly. (Done: JAX and PyTorch paths support RLDS)
- [x] Task: Verify JAX/Mamba integration.
    - [x] Subtask: Check that the training script correctly utilizes JAX and Mamba components as defined in the Tech Stack. (Done: `jax_models` used in `run_trainer.py`)
- [x] Task: Conductor - User Manual Verification 'HOPE Seed Training Pipeline' (Protocol in workflow.md)

## Phase 4: Documentation
- [x] Task: Update `continuonbrain/README.md`.
    - [x] Subtask: specific instructions on how to run the unified runtime and training pipeline. (Done: README updated with canonical entry points)
- [x] Task: Conductor - User Manual Verification 'Documentation' (Protocol in workflow.md)
- [x] Task: Address UI metrics fetch errors and server crash in `/api/runtime/control_loop`. (Fixed: Corrected function signatures and error handling in server.py and robot_modes.py)
