# Specification: System-wide Verification & Stability Harness

## Overview
Perform a comprehensive, multi-phase verification of the unified ContinuonBrain runtime. This track ensures that all recently implemented subsystems (Safety Kernel, RBAC, WaveCore, Context Graphs, and Brain Studio) operate harmoniously and reliably under both scripted and stressed conditions.

## Functional Requirements
- **Phase A: Master Scripted Testing**
    - Implement a `master_system_verify.py` script to exercise all API endpoints.
    - Verify **Safety Kernel** command interception and joint clipping.
    - Verify **RBAC** enforcement using mock tokens (Creator/Developer/Consumer).
    - Exercise **Context Graph** ingestion and vector search retrieval.
- **Phase C: Stability Stress Test**
    - Conduct a 1-hour continuous simulation in "Mock Hardware" mode.
    - Run a high-frequency **Curiosity Loop** (Teacher-Student) to monitor long-term stability.
    - Track and log thermal/resource metrics via `system_health.py` during the run.
- **Phase B: Automated Integration Suite**
    - Set up a `pytest`-driven integration suite for clean-room verification.
    - Target 80%+ coverage for core logic in `kernel/`, `api/`, and `wavecore/`.
- **Reporting & Visualization**
    - Generate a structured `system_audit_report.json` containing detailed pass/fail data.
    - Ensure verification results dynamically populate the **Brain Studio "Capability Matrix"** and "AGI Readiness" panels.

## Non-Functional Requirements
- **CI Compatibility:** Verification must be capable of running entirely in "Mock Hardware" mode without physical sensors.
- **Latency Monitoring:** Explicitly measure and verify that Safety Kernel validation overhead remains below 5ms.
- **Auditability:** Every test step must be logged with a high-resolution timestamp.

## Acceptance Criteria
- [ ] Master verification script completes successfully for all 5 subsystems.
- [ ] 1-hour stress test shows no memory leaks or unhandled process crashes.
- [ ] RBAC correctly returns `403 Forbidden` for unauthorized token/endpoint combinations.
- [ ] Safety Kernel successfully clips at least 3 distinct "illegal" joint configurations.
- [ ] Brain Studio displays an updated "Capability Matrix" based on the `system_audit_report.json`.
