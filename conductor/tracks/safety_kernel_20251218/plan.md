# Plan: Safety Kernel (Ring 0) Implementation

## Phase 1: IPC and Process Isolation
- [ ] Task: Create the `SafetyKernel` process skeleton.
    - [ ] Subtask: Implement `continuonbrain/kernel/safety_kernel.py` as a standalone entry point.
    - [ ] Subtask: Set up a Unix Socket or ZMQ "Actuation Stream" listener.
    - [ ] Subtask: Write a test to verify the Kernel can receive and acknowledge dummy commands while running in a separate process.
- [ ] Task: Conductor - User Manual Verification 'IPC and Process Isolation' (Protocol in workflow.md)

## Phase 2: The Constitution (Deterministic Logic)
- [ ] Task: Implement Kinematic and Power Guard logic.
    - [ ] Subtask: Create `Constitution` class to handle joint limit and velocity clipping.
    - [ ] Subtask: Implement battery voltage and thermal check intercepts.
- [ ] Task: Implement Environmental Constraints.
    - [ ] Subtask: Add support for "No-Go" zone polygons (Static Obstacles).
    - [ ] Subtask: Write unit tests verifying that out-of-bounds commands are detected.
- [ ] Task: Conductor - User Manual Verification 'The Constitution' (Protocol in workflow.md)

## Phase 3: Graduated Response System
- [ ] Task: Implement Corrective Clipping (Level 1).
- [ ] Task: Implement Hard Halt and Degraded Recovery (Level 2 & 3).
    - [ ] Subtask: Create a "Home" position routine that the Kernel can trigger autonomously.
    - [ ] Subtask: Set up the "Stderr" stream to notify the Userland Brain of violations.
- [ ] Task: Conductor - User Manual Verification 'Graduated Response' (Protocol in workflow.md)

## Phase 4: Shell Integration and Boot
- [ ] Task: Implement Shell-specific config loading.
    - [ ] Subtask: Create `continuonbrain/kernel/configs/pi5_arm_safety.json`.
    - [ ] Subtask: Update `startup_manager.py` to launch the Safety Kernel process *before* the Brain Server.
- [ ] Task: Refactor `BrainService` to use the Kernel.
    - [ ] Subtask: Redirect all `Actuator` calls from direct methods to the Kernel's IPC stream.
- [ ] Task: Conductor - User Manual Verification 'Shell Integration' (Protocol in workflow.md)

## Phase 5: End-to-End Validation
- [ ] Task: Verify Ring 0 survival (Kill the Brain, trigger physical E-stop).
- [ ] Task: Benchmarking (Ensure < 5ms latency).
- [ ] Task: Conductor - User Manual Verification 'Validation' (Protocol in workflow.md)
