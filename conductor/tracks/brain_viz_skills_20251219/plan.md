# Plan: Brain Visualization Dashboard & Skill-Teaching Roadmap

## Phase 1: Visualization Infrastructure (Brain Studio)
- [x] **Task: Core Data Feed API**
    - [ ] Implement a WebSocket or SSE endpoint in `studio_server.py` for real-time cognitive streaming.
    - [ ] Create a `StateAggregator` in `continuonbrain/studio_server.py` to collect data from the Brain runtime.
- [x] **Task: Cognitive Feed UI Component**
    - [ ] Add a "Thought Stream" panel to the Brain Studio frontend.
    - [ ] Implement real-time rendering of LLM reasoning steps with 200ms latency target.
- [x] **Task: HOPE Loop Visualizer**
    - [ ] Add a UI indicator (Fast/Mid/Slow) reflecting the current active SSM loop.
    - [ ] Connect the visualizer to the `hope_impl` state changes.
- [x] **Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)**

## Phase 2: Metrics & State Tracking Integration
- [x] **Task: Tool Invocation Logging**
    - [ ] Create a centralized hook in the Brain's tool-calling logic to log inputs/outputs.
    - [ ] Add a "Tool Audit" panel to Brain Studio to display these logs.
- [x] **Task: Performance & System Metrics**
    - [ ] Integrate `resource_monitor.py` data into the Studio dashboard feed.
    - [ ] Display Inference Latency and CPU/Memory usage trends.
- [x] **Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)**

## Phase 3: Skill-Teaching Framework & Initial Tools
- [x] **Task: Brain Tool Registry**
    - [ ] Define a standard `BaseBrainTool` interface in `continuonbrain/tools/`.
    - [ ] Implement a `ToolRegistry` for dynamic loading of capabilities.
- [x] **Task: Phase A - Calculator Tool Implementation**
    - [ ] Implement a safe, deterministic `CalculatorTool`.
    - [ ] Add unit tests verifying math accuracy and safety boundaries.
- [x] **Task: Phase B - FileSystem Awareness Tool**
    - [ ] Implement a read-only `FileSystemTool` with strict directory whitelisting.
    - [ ] Verify the Brain can list and read permitted project files.
- [x] **Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)**

## Phase 4: Knowledge Retrieval & Autonomous Curriculum
- [x] **Task: Phase C - Wikipedia Retrieval Tool**
    - [ ] Implement a `WikipediaTool` for factual grounding (requires internet).
    - [ ] Add an "Offline/Online" safety check to the tool invocation logic.
- [x] **Task: Curriculum Manager Logic**
    - [ ] Create a `CurriculumManager` to trigger autonomous "Lessons" (test scenarios).
    - [ ] Define JSON-based "Lesson Plans" for Math and Knowledge retrieval.
- [x] **Task: Dashboard Curriculum Controls**
    - [ ] Add a "Skill Roadmap" tab to Brain Studio to view and start autonomous lessons.
- [x] **Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)**

## Phase 5: Verification & Final Polish
- [x] **Task: System-Wide Integration Test**
    - [ ] Run an end-to-end "Lesson" where the Brain uses both Calculator and Wikipedia.
    - [ ] Verify all logs and metrics are captured in Brain Studio during the lesson.
- [x] **Task: Code Quality & Coverage**
    - [ ] Achieve >80% coverage for the new tools and curriculum logic.
    - [ ] Run `ruff` / `pytest` across the new components.
- [x] **Task: Conductor - User Manual Verification 'Phase 5' (Protocol in workflow.md)**
