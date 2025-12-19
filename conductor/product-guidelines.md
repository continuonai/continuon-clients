# Product Guidelines: Continuon AI

## Core Personality
- **Scientific & Precise:** Communication is technically accurate, prioritizing transparency over conversational fluff. The system is honest about its "Brain" confidence levels and reasoning.
- **Helpful & Minimalist:** While precise, the interface is designed to be an efficient assistant, removing friction and providing clear, minimal paths to action.

## Visual Identity & Design
- **Multi-Mode Interfaces:**
    - **Technical Dashboard (Developer Mode):** High data density, real-time telemetry graphs, and debug overlays that visualize the "Searcher's" internal simulations.
    - **Clean & Consumer-Friendly (Operator Mode):** Focused hero interactions and ample whitespace for accessibility and ease of use.
- **Design Ethos:** Prioritize utility and data visualization to build trust through clarity.

## Ecosystem & Collaboration
- **Hybrid Openness:**
    - **Closed Core:** The central AI Brain and RLDS data pipeline are maintained as a closed, secure environment to ensure performance and integrity.
    - **Open Skills:** 3rd-party makers are empowered to build and contribute specialized "Skills" and hardware adapters (Shells) through a standardized development kit.

## Safety & Ethics
- **Layered Safety Architecture:**
    - **Hardware-Level Limits (Ring 0):** Physical constraints (force, speed, joint limits) and data streaming protocols are enforced at the lowest architectural level, independent of the AI's predictions.
    - **Transparency of Intent:** The system must visualize its "imagined" outcomes, allowing users to understand the "why" behind any invented solution.
- **Manual Oversight:** High-stakes actions require explicit human confirmation.

## Data Sovereignty & Privacy
- **Local-First Processing:** PII scrubbing and initial data curation are performed locally on the Pi 5 before any manual upload.
- **User Ownership:** Users retain full control over their RLDS episodes, with the ability to export, redact, or delete data from the system at any time.
- **Data Integrity:** Episode provenance is signed and verified to ensure training data remains untampered from edge to cloud.
