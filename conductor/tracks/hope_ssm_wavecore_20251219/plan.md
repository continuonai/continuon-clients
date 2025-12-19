# Plan: HOPE Loop SSM & WaveCore Development (Seed Brain)

## Phase 1: WaveCore Library Maturation
- [ ] **Task: Refactor WaveCore Prototype**
    - [ ] Move `toy_wave_model.py` logic into a structured package: `continuonbrain/wavecore/models/`, `continuonbrain/wavecore/layers/`.
    - [ ] Implement a `Config` class to manage model hyperparameters (d_model, n_layers, vocab_size).
- [ ] **Task: Hybrid Block Enhancements**
    - [ ] Improve the `SpectralBlock` and `HybridBlock` implementations to ensure stable gradients during longer training.
    - [ ] Add support for optional sliding-window attention within the hybrid architecture.
- [ ] **Task: Unit Testing & Verification**
    - [ ] Write unit tests for individual WaveCore layers to verify output shapes and numerical stability.
    - [ ] Run a 100-step "Sanity Training" on a standard Pi 5 to ensure no memory leaks.
- [ ] **Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)**

## Phase 2: Training Infrastructure & Synthetic Data
- [ ] **Task: Synthetic Logic Generator**
    - [ ] Implement a `SyntheticDataPipeline` capable of streaming math, logic, and sequence patterns.
    - [ ] Add support for "Curriculum Levels" (Easy -> Hard) in the data generation logic.
- [ ] **Task: Self-Play Feedback Loop**
    - [ ] Create a training wrapper that allows the model to generate its own "experiences" for reinforcement-style training.
    - [ ] Implement a basic reward/penalty scoring system for synthetic tasks.
- [ ] **Task: Teacher-Student Hooks**
    - [ ] Implement a `DistillationLoss` class to compare WaveCore outputs with a provided "Teacher" signal.
    - [ ] Create a mock Teacher interface for testing the distillation logic without a full LLM.
- [ ] **Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)**

## Phase 3: Seed Model Training & Scaling
- [ ] **Task: Incremental Scaling Runner**
    - [ ] Create a script that sweeps through increasing model sizes (d_model=64 -> 512) to find the Pi 5's stability limit.
    - [ ] Log RAM usage and thermal metrics during the sweep using `system_health.py`.
- [ ] **Task: Robust Checkpointing System**
    - [ ] Implement a `CheckpointManager` that saves weights, optimizer state, and training metadata.
    - [ ] Add a "Auto-Resume" feature to restart training after interruptions.
- [ ] **Task: Training Dashboard Integration**
    - [ ] Export training loss, gradient norms, and current curriculum level to the Brain Studio visualization feed.
- [ ] **Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)**

## Phase 4: Verification & Identity Validation
- [ ] **Task: Seed Model Evaluation**
    - [ ] Run a standard "Intuition Test" on the trained Seed Model (logic puzzles, sequence completion).
    - [ ] Document the model's performance curves relative to its size and training time.
- [ ] **Task: Export & Readiness**
    - [ ] Verify the model can be successfully exported to a format ready for inference optimization (e.g., TorchScript or ONNX).
    - [ ] Ensure all code adheres to the project style guide and passes `pytest`.
- [ ] **Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)**
