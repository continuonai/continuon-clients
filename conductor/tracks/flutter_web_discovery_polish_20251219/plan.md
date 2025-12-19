# Plan: Flutter Web Discovery & UX Polish

## Phase 1: UX Refinement & Manual Path
- [ ] **Task: Web Diagnostic Banner**
    - [ ] Add a `PlatformInfo` helper to detect Web environment.
    - [ ] Implement a `WebDiscoveryNotice` widget in the `RobotListScreen`.
- [ ] **Task: Prominent Manual Connection UI**
    - [ ] Redesign the "Add Robot" flow to be more accessible on the main discovery screen (not hidden in a dialog).
    - [ ] Add a "Connect via IP" fast-path.
- [ ] **Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)**

## Phase 2: persistent Access (Quick Connect)
- [ ] **Task: Sidebar Quick-Connect Bar**
    - [ ] Add a text field to the `ContinuonDrawer` or `ContinuonLayout` for immediate IP entry.
    - [ ] Implement auto-connection logic for the Quick-Connect bar.
- [ ] **Task: Connection Diagnostics**
    - [ ] Improve the error reporting in `BrainClient` to distinguish between "CORS blocked" and "Host unreachable".
- [ ] **Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)**

## Phase 3: Final Polish & Documentation
- [ ] **Task: Update Discovery Documentation**
    - [ ] Add a "Connecting from a Browser" section to `PI5_EDGE_BRAIN_INSTRUCTIONS.md`.
- [ ] **Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)**
