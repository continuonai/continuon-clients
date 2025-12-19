# Specification: Brain Visualization Dashboard & Skill-Teaching Roadmap

## Overview
This track implements a comprehensive visualization infrastructure within **Brain Studio** (`studio_server.py`) to monitor the robot's internal cognitive states and orchestrate an autonomous, phased "Skill-Teaching" curriculum. The goal is to move from passive observation to active, autonomous capability acquisition, starting with deterministic logic and progressing to global knowledge retrieval.

## User Goals
- **Visualize Cognition:** Gain real-time insight into the Brain's reasoning (System 2 thinking), HOPE loop transitions, and attention focus.
- **Monitor Progress:** Track the autonomous acquisition of skills via a dashboard-driven curriculum.
- **Audit Tool Usage:** Inspect how the Brain invokes external tools (Calculator, Wikipedia) during problem-solving.

## Functional Requirements
### 1. Brain Studio Visualization (Phase 1)
- **Cognitive Feed:** Real-time streaming of the internal "Thought" stream (LLM reasoning steps).
- **HOPE Loop Monitor:** Visual indicator of the active loop (Fast/Reflex, Mid/Tactical, Slow/Strategic).
- **Tool Invocation Log:** A dedicated dashboard panel showing tool calls, inputs, and results.
- **Metrics Infrastructure:** Support for secondary feeds (CPU/Memory, Sensory data, Hardware status) using a modular data-feed architecture.

### 2. Autonomous Skill-Teaching Curriculum (Phase 2)
- **Phase A (Deterministic Logic):** Integration and verification of the `Calculator` tool and simple Python snippet execution.
- **Phase B (Environment Awareness):** Integration of file system tools (listing directories, reading permitted config files).
- **Phase C (Knowledge Retrieval):** Integration of a `Wikipedia` retrieval tool for factual grounding.
- **Phase D (Advanced Connectivity):** Framework for broader Web Search and external REST API interaction.
- **Curriculum Manager:** A background process that executes "Lesson" scenarios to verify the Brain can successfully use its new skills.

## Non-Functional Requirements
- **Low Latency:** The "Cognitive Feed" must update within 200ms of state changes.
- **Modular Tooling:** Skills must be implemented as pluggable "Brain Tools" that can be easily enabled/disabled.
- **Offline-First:** All tools (except Wikipedia/Web) must function without external internet access.

## Acceptance Criteria
- [ ] Brain Studio displays a live "Reasoning" stream from the Brain.
- [ ] The dashboard successfully toggles between different HOPE loop visualizations.
- [ ] The Brain autonomously uses the Calculator to solve a provided math problem within a "Lesson."
- [ ] The Brain autonomously retrieves and summarizes information from Wikipedia within a "Lesson."
- [ ] All tool usage is logged and visible in the Brain Studio UI.

## Out of Scope
- Training the core SSM models (Slow Loop) in this track.
- Real-hardware servo control or motion rehearsal (reserved for later tracks).
