# Implementation Plan: System-wide Verification

This plan follows the established TDD-driven workflow, prioritizing scripted validation before moving to long-term stability and automated integration.

## Phase 1: Scripted Subsystem Validation
Create the primary test harness to exercise all recently implemented features.

- [ ] **Task: Master Verification Script Development**
  - Create `verify_full_system.py` using `requests` and `subprocess`.
  - Implement a `SystemHarness` class to manage server/kernel lifecycles.
- [ ] **Task: Safety Kernel Ring 0 Integrity**
  - Sub-task: Verify joint clipping for out-of-bounds commands.
  - Sub-task: Verify E-stop persistence when Brain process is interrupted.
- [ ] **Task: RBAC & Authentication Logic**
  - Sub-task: Test 401/403 responses for various mock tokens.
  - Sub-task: Verify Creator access to sensitive endpoints (/api/admin/*).
- [ ] **Task: Context Graph Ingestion & Search**
  - Sub-task: Trigger RLDS save and verify graph node creation.
  - Sub-task: Verify vector search returns expected semantic neighbors.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Scripted Subsystem Validation' (Protocol in workflow.md)

## Phase 2: Stability & Stress Testing
Verify the system remains reliable under continuous load.

- [ ] **Task: Long-run Stability Simulation**
  - Sub-task: Conduct 1-hour continuous run in Mock Hardware mode.
  - Sub-task: Log memory and CPU trends via `resource_monitor.py`.
- [ ] **Task: Curiosity Loop Stress Test**
  - Sub-task: Run high-frequency Teacher-Student (Gemma-Gemini) exchanges.
  - Sub-task: Monitor SSE stream for thought/loop event consistency.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Stability & Stress Testing' (Protocol in workflow.md)

## Phase 3: Automated Integration Suite
Formalize verification into a reusable pytest suite.

- [ ] **Task: Pytest Integration Harness**
  - Create `continuonbrain/tests/test_integration_full.py`.
  - Implement fixtures for clean-room server/kernel startup.
- [ ] **Task: Achieve Coverage Targets**
  - Ensure core logic in `kernel/`, `api/`, and `wavecore/` exceeds 80% coverage.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Automated Integration Suite' (Protocol in workflow.md)

## Phase 4: Reporting & Dashboard Integration
Expose verification results to the user-facing dashboard.

- [ ] **Task: JSON Audit Report Generation**
  - Implement `system_audit_report.json` export at the end of the verify script.
- [ ] **Task: Brain Studio Capability Matrix Wiring**
  - Sub-task: Add `/api/system/audit` endpoint to `server.py`.
  - Sub-task: Update `RobotListScreen` (Flutter) and `ui.html` to render the Capability Matrix from the audit file.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Reporting & Dashboard Integration' (Protocol in workflow.md)
