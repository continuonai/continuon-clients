# Specification: HOPE v1 Seed Model Training on Pi 5 (8GB)

## Overview
Implement the core training program for the HOPE v1 seed model, specifically optimized for the Raspberry Pi 5 with 8GB RAM. This track focuses on closing the data loop (RLDS -> Trainer -> Model Update -> Inference) while respecting strict memory and thermal constraints.

## Functional Requirements
1. **Sequential Component Training:** Implement a training manager that loads, trains, and saves one HOPE component (e.g., VQ-VAE, Fast Loop SSM, Mid Loop SSM) at a time to stay under the 8GB RAM limit.
2. **Multi-Scale Data Partitioning:** Implement data loaders that slice RLDS episodes into windows appropriate for Fast (50-100ms), Mid (1-10s), and strategic (Minutes) scales.
3. **Offline Search Integration:** Support training on "imagined" rollouts pre-generated offline and stored as standardized RLDS episodes.
4. **Loop Closure Integration:** Ensure the trainer can update model weights that are then immediately usable by the `BrainService` for inference.
5. **Unified Entry Point:** Refactor or extend `run_trainer.py` to support these Pi-specific sequential training optimizations.
6. **Inference Mode Demo:** Demonstrate that the trained model works in inference mode on the same hardware, accessible via API calls and the Web UI (Robot Editor).
7. **Cloud/Flutter Alignment:** Verify and ensure that RLDS schema usage aligns with the ContinuonAI Flutter platform and ContinuonCloud backend requirements for ingestion and storage.

## Non-Functional Requirements
1. **Memory Efficiency:** Training operations must not exceed 8GB, leaving enough overhead for the OS and background robot services.
2. **Stability:** Training must be resumable and resilient to Pi 5 thermal throttling.
3. **Scientific Documentation:** Maintain precise logs for loss, convergence, and loop-closure metrics.

## Acceptance Criteria
1. **Loop Closure:** Successful flow: RLDS data -> Trainer -> Candidate Checkpoint -> Brain Runtime -> Valid Inference Result.
2. **Convergence:** Average loss decreases consistently over 100 steps for at least one component (e.g., Fast Loop).
3. **Functional Reflex:** A trained "Fast Loop" model demonstrates simple tracking or reaching behavior in a mock/sim environment.
4. **No OOM Crashes:** Complete a component training cycle without memory failures.
5. **API/UI Access:** Training and inference controls are functional via the Web UI.

## Out of Scope
- Full "Slow Loop" cloud-scale training infrastructure.
- Optimizations specific to 16GB hardware (to be addressed in future tracks).
