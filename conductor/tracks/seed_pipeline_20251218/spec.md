# Specification: Consolidate Brain Runtime and Establish HOPE Seed Training Pipeline

## Context
The `continuonbrain` directory currently contains overlapping implementations for server startup, training scripts, and model handling. This fragmentation confuses the development of the HOPE (Hierarchical Optimizer, Perpetual Engine) seed model. We need to audit, prune, and unify these components into a single, canonical pipeline.

## Goals
1.  **Audit & Prune:** Identify and remove deprecated or redundant code (e.g., multiple server entry points, conflicting training scripts).
2.  **Unify Runtime:** Establish a single entry point for the Brain Runtime that correctly initializes the Hardware Abstraction Layer (HAL) and exposes the Robot API.
3.  **Canonical Training Pipeline:** define and implement a standard procedure for training the HOPE seed model (Fast/Mid loops) using JAX/Mamba.
4.  **Documentation:** Update internal documentation to reflect the consolidated architecture.

## Requirements
- The "One Brain, Many Shells" architecture must be preserved.
- The consolidated runtime must support the Raspberry Pi 5 (8GB) as the primary target.
- Training scripts must align with the RLDS data contract.
- Maintain the "Scientific & Precise" documentation style.

## Out of Scope
- Implementation of the "Slow Loop" cloud training infrastructure (this is a separate track).
- New feature development for the Android XR or Flutter apps.
