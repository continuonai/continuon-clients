# Specification: Safety Kernel (Ring 0)

## Overview
Implement a dedicated "Safety Kernel" to act as the deterministic gatekeeper (Ring 0) for all robotic actuation. This component isolates safety enforcement from the main AI reasoning loop (Userland), ensuring that the robot respects physical laws, safety norms, and user constraints even if the primary "Brain" process hangs or hallucinates.

## Functional Requirements
1.  **Process Isolation:** The Safety Kernel must run as a separate, high-priority process, communicating with the Brain via IPC (Unix Sockets/ZMQ).
2.  **Deterministic Enforcement:** Intercept all actuator commands and validate them against a "Constitution" of hard-coded rules.
3.  **The Constitution:** Enforce the following pillars:
    *   **Kinematic Limits:** Joint angles, velocity, and torque caps.
    *   **Static Obstacles:** Self-collision avoidance and fixed environmental buffers.
    *   **Human Presence:** E-stop triggers based on proximity sensors.
    *   **Power/Thermal:** Low-voltage and thermal throttling cutoffs.
    *   **User Statutes:** "No-Go" zones configurable via the ContinuonAI app.
4.  **Graduated Response:**
    *   **Level 1 (Minor):** Corrective Clipping (adjust command to safe bounds).
    *   **Level 2 (Major):** Hard Halt (cut power).
    *   **Level 3 (Recovery):** Degraded Recovery (move to Home pose) + User Alert.
5.  **Shell Awareness:** Load hardware-specific safety configurations at boot time (`safety_config.json`), while supporting runtime updates.

## Non-Functional Requirements
1.  **Latency:** Validation overhead must be negligible (< 5ms) to support real-time control loops.
2.  **Reliability:** The Kernel process must auto-restart immediately if it crashes (Systemd watchdog).
3.  **Auditability:** All safety violations must be logged to a persistent "Black Box" stream.

## Acceptance Criteria
1.  **Isolation Test:** Verify the Kernel keeps running and enforcing E-stop even if `BrainService` is killed.
2.  **Constitution Test:** Verify that an unsafe command (e.g., joint over-limit) is successfully clipped (Level 1) or halted (Level 2).
3.  **Integration Test:** Verify seamless IPC communication between `BrainService` (Userland) and `SafetyKernel` (Ring 0).
