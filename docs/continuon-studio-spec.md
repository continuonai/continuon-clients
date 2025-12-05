# Continuon Studio: On-device Robot Editor Design

This document outlines the on-device Robot Editor that ships as part of Continuon Brain and runs in any shell (web browser first), bridging the Robot API into portable panels. It aligns with the broader architecture and HOPE/CMS learning loops described in [`docs/architecture.md`](./architecture.md) and [`docs/hope-cms-vla.md`](./hope-cms-vla.md). Where XR app contracts in [`docs/xr-app-spec.md`](./xr-app-spec.md) apply (e.g., RLDS logging, control adapters), the editor reuses them but is not XR-native; panels are expected to render in standard web canvases or lightweight native shells.

## Goals
- Provide a shell-agnostic (browser-first) Robot Editor that lets operators inspect capabilities, edit routines, and visualize HOPE/CMS signals without needing XR hardware.
- Reuse the stable Robot API exposed by ContinuonBrain/OS while keeping latency-sensitive rendering on-device, consistent with the edge-first split in the overall architecture.
- Surface safety posture (estop, rate limits, envelopes) inline with teleop and playback tools so incidents are caught before commands are applied.
- Support mock and offline modes so the editor remains usable when hardware is unavailable, while keeping schema-compatible outputs for validation.

## Scope and Non-goals
- **In scope:** UI panels for capability discovery, live telemetry, HOPE/CMS visualizations, lightweight routine editing, and safety controls consistent with trainer/observer/workstation modes.
- **Out of scope:** Cloud-only IDE features (CI/CD, dataset curation), OTA packaging, and training job orchestration. Those remain covered in the broader system architecture and HOPE/CMS lifecycle docs.

## Architecture Alignment
- **Bridge orientation:** The editor is a panel set that sits alongside teleop and workstation experiences; it consumes the same gRPC/WebRTC Robot API and RLDS logging path to avoid divergent schemas. Panels should degrade cleanly between browser canvases and native shells.
- **HOPE/CMS linkage:** Visualizations read RLDS-compatible streams and labels so Fast/Mid/Slow loop diagnostics shown in panels match the learning flows defined in the HOPE/CMS note. Mid/Slow visualizations rely on episodic summaries cached on-device, while Fast visual overlays run against live telemetry.
- **Data plane:** Control and telemetry flows reuse the `ContinuonBrainBridgeService` contract. WebRTC is preferred for low latency; gRPC is the fallback. Both transports must tag `client_id` and propagate diagnostics for RLDS logging.

## Required Robot Editor Endpoints
The on-device editor reuses Robot API surfaces but constrains them for editor use. All endpoints should be available over gRPC and mirrored over WebRTC data channels when negotiated.

### Capability Manifest
- **Endpoint:** `GetCapabilityManifest(GetCapabilityManifestRequest) -> CapabilityManifest`.
- **Purpose:** Returns declarative metadata for skills, sensors, and safety features so panels can render toggles, documentation, and availability states.
- **Fields:**
  - `robot_model`, `software_versions`, `safety.envelopes_supported`, `safety.estop_supported`.
  - `skills[]` with `id`, `name`, `parameters`, `required_modalities` (pose, glove, gaze), `safety_tags`.
  - `sensors[]` with `id`, `sample_rate_hz`, `latency_ms`, `frame_id_domain`, `calibration_status`.
- **UI mapping:** Populates the Capabilities panel and seeds editor templates with parameter schemas; drives validation on save.

### Telemetry Stream
- **Endpoint:** `StreamRobotEditorTelemetry(StreamRobotEditorTelemetryRequest) -> StreamRobotEditorTelemetryResponse`.
- **Purpose:** Low-latency stream that merges `robot_state`, `diagnostics`, and selected HOPE/CMS head outputs for overlay rendering.
- **Fields:**
  - `robot_state` (aligned with RLDS robot block, includes `frame_id`, timestamps).
  - `diagnostics` (latency, packet loss, BLE RSSI, mock-mode flag).
  - `safety_state` (estop, rate-limit status, envelope violations, predicted collision horizon from SafetyHead).
  - `hope_cms_signals` (Fast loop hazard scores, Mid loop intent confidence, Slow loop policy version metadata).
- **UI mapping:** Drives live HUD overlays, chart widgets, and logging bars inside the editor. Data should be loggable to RLDS steps when the editor is used during Trainer/Observer modes.

### Routine Apply / Preview
- **Endpoints:**
  - `PreviewRoutine(PreviewRoutineRequest) -> PreviewRoutineResponse` for deterministic simulation on-device (mock-mode and live robot should both be supported).
  - `ApplyRoutine(ApplyRoutineRequest) -> ApplyRoutineResponse` gated by safety checks.
- **Safety gates:** Validate against capability manifest, enforce rate limits, and require Safety workflow acknowledgements before execution.

### Event Hooks
- **Hardware discovery:** Subscribe to device discovery events so panels can show connection states for robots, Continuon Glove, and external sensors. Accepts plug-in providers for future hardware without app updates.
- **Mock-mode fallback:** Exposes `EnableMockMode(EnableMockModeRequest)` to switch to software simulators; telemetry and capability data must remain schema-compatible and marked `mock=true` in diagnostics.
- **UI actions:** Emit `UiAction` events to the RLDS logger when operators run scripts, adjust envelopes, or acknowledge safety prompts, preserving workstation contracts.

## HOPE/CMS Visualizations
- **Fast loop overlays:**
  - Render collision cones, gaze-weighted hazard grids, and SafetyHead overrides from `hope_cms_signals.fast`. Data comes from the telemetry stream and aligns with the RLDS `frame_id` domains.
  - Show per-frame latency and drop counters to trace Fast loop reflex reliability.
- **Mid loop panels:**
  - Session-level intent timelines (from LanguagePlanner) and skill sequence confidence charts tied to `ui_context` and `action.command` history.
  - Adaptive UI suggestions surfaced as inline panel hints; users can pin/accept suggestions, which writes `ui_action` entries to RLDS.
- **Slow loop snapshots:**
  - Show current deployed policy bundle IDs, Memory Plane merge status, and drift alerts sourced from Slow loop summaries cached on-device.
  - Allow downloading comparison summaries from Continuon Cloud when online; offline snapshots are read-only.

## Safety Workflows
- **Inline estop and envelope editor:** Persistent estop widget, envelope visualization, and per-skill safety tags displayed before execution. Envelope edits generate `UiAction` logs and require confirmation when thresholds narrow.
- **Preflight checks:** On `ApplyRoutine`, run capability/parameter validation, check SafetyHead readiness, and confirm robot state (estop cleared, brakes, arm homed). Blocks execution when telemetry reports inconsistent timestamps or missing required modalities (e.g., glove absent for tactile skills).
- **Live intervention:** Telemetry stream events trigger inline toasts and visual highlights; operator acknowledgements are logged and propagated to SafetyHead for adaptive margins.
- **Incident capture:** When a violation occurs, auto-start RLDS incident capture with `episode_metadata.tags += ["safety_incident"]` and persist diagnostic snapshots for HOPE/CMS regression.

## On-device Editor ↔ Robot API ↔ UI Panels
- **Binding strategy:** Panels subscribe to the capability manifest and telemetry streams; actions from UI controls are normalized to Robot API commands using the same control adapters defined for teleop. This keeps `action.command` and `ui_action` semantics aligned with the XR app spec.
- **Panel types:**
  - Capabilities/skills list with parameter editors (schema-driven forms).
  - Telemetry dashboard (state charts, diagnostics, HOPE/CMS overlays).
  - Routine workspace (code/graph editor with Preview/Apply controls).
  - Safety console (envelopes, estop, incident log viewer).
- **Logging:** All panel interactions emit RLDS-consumable events via the XR logger; mock-mode marks observations with `diagnostics.mock_mode=true` but preserves schema for playback and validation.

## Mock-mode and Offline Operation
- **Deterministic simulation:** Preview routines against a physics or scripted simulator with seeded randomness; outputs must mirror `StreamRobotEditorTelemetry` fields so UI components do not branch on transport source.
- **Capability synthesis:** When no robot is present, fabricate a capability manifest from cached profiles, marking `source="mock"` and omitting hardware-only fields. Editor should highlight unsupported features but keep editing enabled.
- **Data replay:** Allow loading RLDS episodes to drive the telemetry stream for UX and visualization testing; HOPE/CMS overlays should use the same parsing path as live data.

## Hardware Discovery and Extensibility Hooks
- **Discovery bus:** Listen for robot endpoints (USB/Ethernet/Wi-Fi), Continuon Glove BLE advertisements, and other sensor providers. Emit structured discovery events with device IDs, transport, and firmware versions.
- **Plug-in adapters:** Define a registry for capability providers so new hardware classes can contribute manifest entries and telemetry blocks without app updates, as long as they conform to RLDS-compatible schemas.
- **Diagnostics propagation:** Discovery and connection state changes feed into `diagnostics` for RLDS logging and UI badges; panel-level filters allow users to scope telemetry to selected devices.

## Open Questions / Next Steps
- Formalize protobuf definitions for `CapabilityManifest` and `StreamRobotEditorTelemetry` in `proto/continuonxr/continuonbrain/v1/` with backward-compatible fields.
- Decide how much Slow loop summary data to cache on-device versus fetching on demand from Continuon Cloud, balancing privacy and freshness.
- Validate mock-mode schemas against the existing RLDS validation tooling to ensure episodes produced during editor testing remain ingestible.
