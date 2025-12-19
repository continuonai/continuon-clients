# Specification: HOPE Loop SSM & WaveCore Development (Seed Brain)

## Overview
This track focuses on the development, training, and integration of the **State Space Models (SSMs)** that power the Fast, Mid, and Slow loops of the HOPE architecture. Utilizing the `WaveCore` pipeline, we will train a unified **Seed Brain Model** directly on the Raspberry Pi 5. The primary goal is to build the robot's "intuition" and "identity" through synthetic data and self-play, leveraging a "One Brain" architecture that can later be specialized for different loops.

## User Goals
- **Establish Identity:** Train a local "Seed" model that exhibits basic consistent behaviors (intuition) without reliance on cloud models.
- **Validate WaveCore:** Evolve the existing `WaveCore` prototype into a production-grade training pipeline capable of supporting HOPE's multi-loop requirements.
- **Edge-Native Training:** Prove that meaningful learning (Mid-Loop adaptation) can occur directly on the Raspberry Pi 5 hardware.

## Functional Requirements
### 1. WaveCore Pipeline Maturation
- **Architecture Upgrade:** Refactor `toy_wave_model.py` into a modular `WaveCore` library supporting configurable `d_model`, state expansion, and hybrid (SSM + Attention) blocks.
- **Loop Specialization:** Define configuration profiles for Fast (Latency-optimized), Mid (Context-optimized), and Slow (Capacity-optimized) variants within the unified architecture.

### 2. Data & Training Infrastructure
- **Synthetic Generator:** Create a data stream of deterministic logic patterns (math, sequences) as the primary "Grade School" curriculum.
- **Self-Play Loop:** Implement a feedback loop where the model's outputs act as inputs for the next step, rewarded by simple stability or goal-completion metrics.
- **Distillation Hooks:** Add interfaces to optionally use an onboard LLM (e.g., Gemma) as a "Teacher" for synthetic label generation (Teacher-Student setup).

### 3. Seed Model Training
- **Incremental Scaling:** Implement a training script that starts small (Toy config) and automatically scales model size to find the Pi 5's stability limit.
- **Checkpointing:** Robust saving/loading of model weights to support long-running, interruptible training sessions on the edge.

## Non-Functional Requirements
- **Resource Constraints:** Training must not exceed 6GB RAM usage (leaving 2GB for OS/overhead) on the Pi 5.
- **Thermal Management:** The training loop must respect thermal throttling indicators from `system_health.py`.
- **Modularity:** The model architecture must be exportable to TFLite/XNNPACK for future inference optimization.

## Acceptance Criteria
- [ ] `WaveCore` is refactored into a proper library with unit tests for its blocks.
- [ ] A "Seed Model" can be trained on the Pi 5 using synthetic data for >1 hour without crashing or overheating.
- [ ] The trained model demonstrates "better than random" performance on a hold-out synthetic validation set.
- [ ] The system supports a "Distillation Mode" where a mock Teacher signal can guide the training loss.

## Out of Scope
- Cloud-based training (Slow Loop offloading).
- Large-scale Human Teleoperation data ingestion (RLDS) for this specific seed phase.
- Final high-performance inference optimization (C++ rewrites/Assembly), though exportability is required.
