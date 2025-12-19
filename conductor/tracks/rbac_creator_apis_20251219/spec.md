# Specification: Role-Based Access Control (RBAC) & Creator APIs

## 1. Overview
This track implements the "Creator Mode" backend infrastructure on the ContinuonBrain and the corresponding management UI within the **ContinuonAI Companion App (Flutter)**. The goal is to centralize robot management (Training, Inference, Model Ops) in the Flutter app by exposing robust, role-secured APIs on the Brain. This ensures a unified experience where the app automatically adapts its interface based on the user's authenticated role.

## 2. Goals
1.  **API-First Architecture:** Expose all "Creator" capabilities (Seed management, Training toggles, Data tagging) via secure REST/gRPC endpoints on the Brain.
2.  **Secure RBAC Backend:** Implement middleware in the Brain's `robot_api_server` to validate Firebase ID Tokens and enforce permission boundaries (Creator vs. Consumer).
3.  **Unified Flutter Management:** Update the `continuonai` (Companion App) to serve as the primary console. It will authenticate with the Brain and dynamically render "Creator Tools" (Studio, Training Dashboard) only for authorized users.
4.  **Context Switching:** Enable the Flutter app to switch the Brain between **Inference**, **Manual HITL**, and **Autonomous (HOPE)** modes via API.

## 3. User Roles (Enforced by Brain API)
*   **Creator:** Super Admin. Full API access (Context switching, Model I/O, Data management).
*   **Developer:** Access to Training/Debug APIs.
*   **Consumer:** Restricted to `read-only` status and standard `control` endpoints.
*   **Lessee/Renter:** Restricted usage.
*   **Enterprise Fleet:** Telemetry access.

## 4. Functional Requirements

### 4.1. Brain API (Python - `continuonbrain`)
*   **Auth Middleware:**
    *   Validate Firebase ID Tokens (Bearer Token).
    *   Extract Claims/Roles.
    *   **Enforcement:** Return `403 Forbidden` for Consumer requests to Creator endpoints.
*   **New API Controllers:**
    *   `AuthController`: Handle token verification and session state.
    *   `ContextController`: Endpoints to set system mode (`/mode/inference`, `/mode/train/manual`, `/mode/train/autonomous`).
    *   `ModelController`: Endpoints to `/upload`, `/list`, and `/activate` seed models.
    *   `DataController`: Endpoints to `/episodes/list` and `/episodes/tag`.

### 4.2. Companion App (Flutter - `continuonai`)
*   **Auth Integration:** Ensure the app passes the user's Firebase Token in the header of all requests to the Robot Brain (Local IP or Remote).
*   **Dynamic UX:**
    *   **State Management:** Store the current `UserRole` in the app state.
    *   **Creator Dashboard:** A new dedicated section for "Brain Management" visible only to Creators.
    *   **Mode Toggles:** UI switches to trigger the `ContextController` endpoints with visual feedback of the robot's current state.
*   **GenUI & Architecture:** Explore Flutter GenUI packages for fluid controls and follow formal BLoC architecture for state management.

## 5. Non-Functional Requirements
*   **Offline Capability:** The Flutter app must be able to authenticate and control the robot locally via LAN even if the internet is down (using cached credentials/keys where possible).
*   **Security:** The Brain must strictly validate roles; the UI is just a presentation layer. Hiding a button in Flutter is not security—blocking the API call in Python is.

## 6. Out of Scope
*   "Cloud" side of the Enterprise Dashboard.
*   Modification of the Android XR Shell (`apps/continuonxr`).
