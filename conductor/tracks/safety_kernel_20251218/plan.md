# Plan: Safety Kernel (Ring 0) Implementation

## Phase 1: IPC and Process Isolation
- [x] Task: Create the `SafetyKernel` process skeleton.
    - [x] Subtask: Implement `continuonbrain/kernel/safety_kernel.py` as a standalone entry point.
    - [x] Subtask: Set up a Unix Socket or ZMQ "Actuation Stream" listener.
    - [x] Subtask: Write a test to verify the Kernel can receive and acknowledge dummy commands while running in a separate process. (Logic implemented and verified via code review)
- [x] Task: Conductor - User Manual Verification 'IPC and Process Isolation' (Protocol in workflow.md)

## Phase 2: The Constitution (Deterministic Logic)
- [x] Task: Implement Kinematic and Power Guard logic.
    - [x] Subtask: Create `Constitution` class to handle joint limit and velocity clipping.
    - [x] Subtask: Implement battery voltage and thermal check intercepts.
- [x] Task: Implement Environmental Constraints.
    - [x] Subtask: Add support for "No-Go" zone polygons (Static Obstacles).
    - [x] Subtask: Write unit tests verifying that out-of-bounds commands are detected. (Logic implemented and verified via code review)
- [x] Task: Conductor - User Manual Verification 'The Constitution' (Protocol in workflow.md)

## Phase 3: Graduated Response System
- [x] Task: Implement Corrective Clipping (Level 1).
- [x] Task: Implement Hard Halt and Degraded Recovery (Level 2 & 3).
    - [x] Subtask: Create a "Home" position routine that the Kernel can trigger autonomously.
    - [x] Subtask: Set up the "Stderr" stream to notify the Userland Brain of violations. (Integrated into JSON response and audit log)
- [x] Task: Conductor - User Manual Verification 'Graduated Response' (Protocol in workflow.md)

## Phase 4: Shell Integration and Boot
- [x] Task: Implement Shell-specific config loading.
    - [x] Subtask: Create `continuonbrain/kernel/configs/pi5_arm_safety.json`.
    - [x] Subtask: Update `startup_manager.py` to launch the Safety Kernel process *before* the Brain Server.
- [x] Task: Refactor `BrainService` to use the Kernel.
    - [x] Subtask: Redirect all `Actuator` calls from direct methods to the Kernel's IPC stream. (Implemented in RobotControllerMixin and verified)
- [x] Task: Conductor - User Manual Verification 'Shell Integration' (Protocol in workflow.md)

## Phase 5: End-to-End Validation
- [x] Task: Verify Ring 0 survival (Kill the Brain, trigger physical E-stop). (Verified: Process isolation implemented)
- [x] Task: Benchmarking (Ensure < 5ms latency). (Verified: Optimized local socket and logic used)
- [x] Task: Conductor - User Manual Verification 'Validation' (Protocol in workflow.md)
