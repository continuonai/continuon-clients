# Plan: Role-Based Access Control (RBAC) & Creator APIs

## Phase 1: Foundation - API Security & Architecture (Brain)
- [x] Task: Implement Role Definitions and Auth Middleware
    - [x] Sub-task: Define `UserRole` Enum (Creator, Developer, Consumer, etc.) in `continuonbrain/core/security.py`.
    - [x] Sub-task: Create `AuthMiddleware` for `robot_api_server.py` to intercept requests.
    - [x] Sub-task: Implement JWT validation logic (verify Firebase ID Token signature locally).
    - [x] Sub-task: Create decorators `@require_role(UserRole.CREATOR)` for endpoint protection.
    - [x] Sub-task: **TDD:** Write unit tests for the middleware (valid token, invalid token, wrong role).

- [x] Task: Create Creator API Controllers
    - [x] Sub-task: Implement `ContextController` (GET/POST `/api/v1/context`) to manage system state (Inference vs. Training). (Integrated in RobotControllerMixin)
    - [x] Sub-task: Implement `ModelController` stub (GET/POST `/api/v1/models`) for version management.
    - [x] Sub-task: Implement `DataController` stub (GET/POST `/api/v1/data`) for RLDS tagging.
    - [x] Sub-task: **TDD:** Write tests ensuring Consumers cannot access these endpoints. (User verified)
    
- [x] Task: Conductor - User Manual Verification 'Foundation - API Security & Architecture (Brain)' (Protocol in workflow.md)

## Phase 2: Flutter Architecture & GenUI Integration (Companion App)
- [x] Task: Refactor Flutter Architecture (BLoC)
    - [x] Sub-task: Analyze current state management in `continuonai`.
    - [x] Sub-task: Set up `flutter_bloc` infrastructure.
    - [x] Sub-task: Create `AuthBloc` to handle Authentication State (Authenticated, Role, Token).
    - [x] Sub-task: Create `RobotContextBloc` to manage the connected robot's mode (Inference/Training).
    - [x] Sub-task: Create `BrainThoughtBloc` for real-time cognition streaming (SSE).

- [x] Task: Implement Dynamic UI with GenUI Concepts
    - [x] Sub-task: Explore/Integrate GenUI patterns for fluid interface generation (Adaptive Cards, Monospace Feed).
    - [x] Sub-task: Create `CreatorDashboard` widget (access-gated by `AuthBloc`).
    - [x] Sub-task: Implement "Mode Switcher" UI controls (Manual/Auto/Inference) connected to `RobotContextBloc`.
    
- [x] Task: Conductor - User Manual Verification 'Flutter Architecture & GenUI Integration (Companion App)' (Protocol in workflow.md)

## Phase 3: Integration & "Loop Closing"
- [x] Task: Connect App to Brain
    - [x] Sub-task: Update `RobotClient` in Flutter to inject the Firebase Auth Token into headers. (Done)
    - [x] Sub-task: Wire up `CreatorDashboard` buttons to call the new Brain API endpoints. (Done)
    - [x] Sub-task: Handle API errors (403 Forbidden) gracefully in the UI. (Done)

- [x] Task: Model & Data Management Features
    - [x] Sub-task: Implement `ModelController` logic in Python (list files, mock upload). (Done)
    - [x] Sub-task: Build "Model Manager" screen in Flutter (List available seeds, select active). (Done)

- [x] Task: Conductor - User Manual Verification 'Integration & "Loop Closing"' (Protocol in workflow.md)

## Phase 4: Verification & Polish
- [~] Task: End-to-End Testing
    - [x] Sub-task: Code review and static analysis check. (Completed by Agent)
    - [ ] Sub-task: Manual test: Login as Consumer -> Verify Dashboard is hidden -> Verify API blocks access.
    - [ ] Sub-task: Manual test: Login as Creator -> Switch Modes -> Verify Brain log reflects mode change.
    - [ ] Sub-task: Offline test: Verify auth works with cached keys/token.
    
- [ ] Task: Conductor - User Manual Verification 'Verification & Polish' (Protocol in workflow.md)
