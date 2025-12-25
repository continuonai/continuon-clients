# Specification: Unified Server Consolidation & Connection Smoothing

## Overview
This track consolidates the redundant web server processes into a single **Unified API Server** and refines the robot connection/seed installation workflow. The goal is to provide a robust, cross-platform (Windows & Raspberry OS) runtime that serves both the API and the emergency Web UI, while ensuring the Flutter app can seamlessly discover, pair with, and install seed models on the robot.

## Functional Requirements
- **Unified API Server:**
    - Consolidate all logic from `studio_training_server.py` and deprecated server scripts into `continuonbrain/api/server.py`.
    - Serve the Brain Studio frontend (templates and static assets) directly from the primary service port.
    - Support all operating modes (Inference, Training, Manual) and user roles (Creator, Developer, Consumer) via integrated RBAC.
- **Connection & Pairing Refinement:**
    - Ensure the mDNS/QR pairing flow reliably transfers Firebase Auth tokens and ownership metadata to the Brain.
    - Implement a "smooth" first-connection experience that transitions from Guest to Creator role without manual intervention.
- **Seed Installation Workflow:**
    - Implement the logic to automatically stage, verify, and activate seed models when requested by the Flutter app.
- **Cross-Platform Compatibility:**
    - **IPC Fallback:** Implement and verify TCP-based communication for the Safety Kernel IPC on Windows, while retaining Unix Sockets for Linux.
    - **Path Agnostic:** Ensure all file operations use `pathlib` for consistency across Windows and Linux.
    - **Process Control:** Update `startup_manager.py` to use platform-agnostic process management for the server and kernel.

## Non-Functional Requirements
- **High Availability:** The Web UI must serve as a reliable "emergency" interface if the Flutter app is unavailable.
- **Performance:** Asset serving and API response times must remain performant on Raspberry Pi 5 hardware.
- **Zero-Config Discovery:** mDNS must function reliably on local subnets for both CLI and App clients.

## Acceptance Criteria
- [ ] A single server process handles both API requests and Web UI serving on port 8080/8081.
- [ ] The runtime starts and functions correctly on both Windows 11 and Raspberry Pi OS.
- [ ] The Safety Kernel correctly intercepts and validates commands on Windows via TCP.
- [ ] The Flutter app completes a full "Discover -> Pair -> Install Seed -> Control" loop with no manual IP entry.
- [ ] RBAC correctly gates the emergency Web UI based on the authenticated role.

## Out of Scope
- Global/Cloud discovery (outside local subnet).
- Non-seed model training (Cloud Slow Loop).
