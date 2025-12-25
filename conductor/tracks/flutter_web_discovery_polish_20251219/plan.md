# Plan: Flutter Web Discovery & UX Polish

## Phase 1: UX Refinement & Manual Path
- [x] **Task: Web Diagnostic Banner**
    - [x] Add a `PlatformInfo` helper to detect Web environment. (Done: `continuonai/lib/utils/platform_info.dart`)
    - [x] Implement a `WebDiscoveryNotice` widget in the `RobotListScreen`. (Done: `_buildWebDiscoveryNotice` added)
- [x] **Task: Prominent Manual Connection UI**
    - [x] Redesign the "Add Robot" flow to be more accessible on the main discovery screen (not hidden in a dialog). (Done: `_buildProminentAddRobot` added)
    - [x] Add a "Connect via IP" fast-path. (Done: `_quickConnect` implemented)
- [x] **Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)**

## Phase 2: persistent Access (Quick Connect)
- [x] **Task: Sidebar Quick-Connect Bar**
    - [x] Add a text field to the `ContinuonDrawer` or `ContinuonLayout` for immediate IP entry. (Done: `ContinuonDrawer` updated)
    - [x] Implement auto-connection logic for the Quick-Connect bar. (Done: `_handleQuickConnect` implemented)
- [x] **Task: Connection Diagnostics**
    - [x] Improve the error reporting in `BrainClient` to distinguish between "CORS blocked" and "Host unreachable". (Done: `ConnectionDiagnostic` and `runDiagnostics` added)
- [x] **Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)**

## Phase 3: Final Polish & Documentation
- [x] **Task: Update Discovery Documentation**
    - [x] Add a "Connecting from a Browser" section to `PI5_EDGE_BRAIN_INSTRUCTIONS.md`. (Done)
- [x] **Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)**
