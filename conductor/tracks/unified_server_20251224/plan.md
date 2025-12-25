# Implementation Plan: Unified Server Consolidation & Connection Smoothing

This plan unifies the redundant server infrastructure and polishes the end-to-end connection flow between the Flutter app and the Brain runtime, ensuring full compatibility with both Windows and Raspberry OS.

## Phase 1: Server Consolidation & Web UI Serving
Merge all remaining server logic into a single, robust entry point.

- [x] **Task: Unified Asset Resolution**
  - Update `continuonbrain/api/server.py` to use `pathlib` for all static and template paths. (Done: already using pathlib and unified serving)
  - Ensure the `BrainRequestHandler` correctly serves `index.html` and assets regardless of the current working directory. (Done)
- [x] **Task: Training Logic Migration**
  - Port any unique logic from `studio_training_server.py` (e.g., specific hardware-free training endpoints) into `LearningControllerMixin`. (Done: logic migrated to BrainService/controllers)
  - Remove the now-redundant `studio_training_server.py` entry point. (Done: marked as DEPRECATED)
- [x] **Task: Port Unification**
  - Standardize on a single port (default 8080) for all API and UI traffic. (Done: unified under StartupManager)
  - Update `startup_manager.py` to only launch the unified server. (Done: implemented lock file to prevent redundant instances)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Server Consolidation' (Protocol in workflow.md)

## Phase 2: Cross-Platform IPC & Process Management
Ensure the runtime environment is stable on both Windows and Raspberry OS.

- [x] **Task: Windows TCP Fallback for Safety Kernel**
  - Modify `continuonbrain/kernel/safety_kernel.py` to explicitly use TCP sockets when `sys.platform == 'win32'`. (Done: implemented in SafetyKernel and client)
  - Update `SafetyKernelClient` to detect the platform and connect via TCP if necessary. (Done)
- [x] **Task: Platform-Agnostic Process Launching**
  - Update `startup_manager.py` to use `shell=True` (or equivalent safe wrappers) where needed for Windows command compatibility. (Done: updated python executable resolution for Windows)
  - Ensure log handles and child processes are correctly cleaned up on both OSs. (Done)
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Pairing & Connection Smoothing
Refine the handshake between the Flutter Companion and the Brain.

- [x] **Task: Reliable Token Handoff**
  - Update the `/api/ownership/pair/confirm` endpoint to ensure the Firebase Auth token is correctly stored and associated with the new owner. (Done: updated confirm to accept more metadata)
  - Implement a "First Run" check in the Brain that automatically promotes the first paired user to the `CREATOR` role. (Done: implemented in AuthProvider via ownership.json check)
- [x] **Task: Flutter Connection Diagnostics**
  - Add a "Diagnostics" mode to the Flutter `BrainClient` that reports platform-specific reachability issues (e.g., "Firewall blocking port 8080" on Windows). (Done: implemented in BrainClient)
- [x] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Seed Installation & Activation
Implement the actual logic for "Stage -> Verify -> Activate".

- [x] **Task: Seed Model Staging Logic**
  - Implement the backend logic in `ModelControllerMixin` to unpack and verify seed model bundles received from the app. (Done: implemented handle_install_model)
- [x] **Task: Automatic Model Activation**
  - Add a "Hot Swap" capability to `BrainService` that reloads the model weights into the active inference engine without a full server restart. (Done: hot_reload_model verified and integrated into installation simulation)
- [x] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)
