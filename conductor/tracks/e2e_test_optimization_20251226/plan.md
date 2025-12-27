# Implementation Plan: End-to-End Test & Entry Point Optimization

This plan outlines the steps to implement a comprehensive E2E test suite and optimize the Brain's boot configuration as specified in the track specification.

## Phase 1: Boot Optimization & HAL Refinement
Focuses on the "One Brain" startup logic and how it handles hardware detection on non-robot platforms (Windows).

- [x] **Task: Research current entry point configuration**
    - Audit `continuonbrain/startup_manager.py` and configuration files to identify optimization opportunities for shell-to-brain handoff.
- [x] **Task: Refactor Entry Point Management**
    - Implement a more flexible/declarative way to define and launch brain modules at boot.
- [x] **Task: Implement Robust "Hardware Detect" HAL**
    - Update `continuonbrain/hal/` to ensure production drivers fail gracefully to mocks when running on local dev machines (Windows).
- [x] **Task: Write Unit Tests for Startup & HAL Fallback**
- [x] **Task: Implement Refactored Startup Logic**
- [ ] **Task: Conductor - User Manual Verification 'Phase 1: Boot Optimization & HAL Refinement' (Protocol in workflow.md)**

## Phase 2: Layered Verification Suite (Network & Content)
Focuses on the first two layers of the E2E verification: API reachability and HTML artifact inspection.

- [x] **Task: Create E2E Test Harness Structure**
    - Setup a dedicated test file (e.g., `tests/e2e/test_full_stack_smoke.py`) using `pytest` or similar.
- [x] **Task: Implement Layer 1: Network Reachability**
    - Write tests to ping Brain API, Web Server, and Flutter Web endpoints.
- [x] **Task: Implement Layer 2: Content Inspection**
    - Write tests to verify HTML responses contain necessary Flutter/Web scripts and assets.
- [ ] **Task: Conductor - User Manual Verification 'Phase 2: Layered Verification Suite (Network & Content)' (Protocol in workflow.md)**

## Phase 3: Browser Automation (Layer 3)
Integrates headless browser testing to verify the UI actually renders and functions on the client side.

- [x] **Task: Setup Playwright/Selenium Environment**
    - Add necessary dependencies and ensure browser binaries are available for Windows.
- [x] **Task: Implement Layer 3: Headless UI Smoke Test**
    - Write a test that launches the Flutter Web App, waits for initialization, and verifies a key UI element.
- [x] **Task: Integrate UI Test into Main E2E Harness**
- [ ] **Task: Conductor - User Manual Verification 'Phase 3: Browser Automation (Layer 3)' (Protocol in workflow.md)**

## Phase 4: Integration & Documentation
Finalizing the end-to-end flow and ensuring it's easy for developers to run.

- [x] **Task: Final End-to-End Run**
    - Execute the full suite: Startup Brain -> Network Check -> Content Check -> Browser Check -> Teardown.
- [x] **Task: Document E2E Test Procedures**
    - Update `README.md` or a dedicated test doc with instructions on how to run the full stack test.
- [ ] **Task: Conductor - User Manual Verification 'Phase 4: Integration & Documentation' (Protocol in workflow.md)**
