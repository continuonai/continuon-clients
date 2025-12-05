# Robot Editor Design (Continuon Brain Studio)

This design describes the on-device Robot Editor that ships with Continuon Brain Studio. It bridges the Robot API into browser-first panels while aligning with the XR runtime contracts in [`docs/xr-app-spec.md`](./xr-app-spec.md) and the HOPE/CMS learning loops outlined in [`docs/hope-cms-vla.md`](./hope-cms-vla.md) and the edge-first split in [`docs/system-architecture.md`](./system-architecture.md).

## Goals and Alignment
- Deliver a shell-agnostic editor (web-first, native shells allowed) so operators can inspect capabilities, edit routines, and visualize HOPE/CMS signals without XR hardware.
- Reuse the same Robot API transports (gRPC/WebRTC) and RLDS logging path used by the XR app, keeping schemas identical for teleop and workstation parity.
- Keep rendering and safety validation on-device per the edge-first safety boundary from the system architecture while surfacing Mid/Slow summaries cached from Continuon Cloud.

## Required Endpoints
The editor consumes the Robot API with a constrained set of editor-facing endpoints exposed over gRPC and mirrored over WebRTC data channels when available.

### Capability Manifest
- **Endpoint:** `GetCapabilityManifest(GetCapabilityManifestRequest) -> CapabilityManifest`.
- **Purpose:** Declarative inventory of skills, sensors, and safety features used to seed panels and validate edits.
- **Fields:** `robot_model`, `software_versions`, `safety.envelopes_supported`, `safety.estop_supported`, `skills[]` (id, name, parameter schema, required modalities, safety tags), `sensors[]` (id, sample_rate_hz, latency_ms, frame_id_domain, calibration_status), `ui_panels_supported`.
- **Behavior:** Responses must be cacheable offline and tag derived entries with `source` (`live` vs `mock`) so mock-mode can prefill UI without hardware.

### Telemetry Stream
- **Endpoint:** `StreamRobotEditorTelemetry(StreamRobotEditorTelemetryRequest) -> StreamRobotEditorTelemetryResponse`.
- **Purpose:** Low-latency stream that merges `robot_state`, `diagnostics`, and HOPE/CMS head outputs for overlays; should mirror the XR `StreamRobotState` payload and tagging.
- **Fields:** `robot_state` (frame_id + timestamps), `diagnostics` (latency, packet loss, BLE RSSI, mock_mode flag), `safety_state` (estop, rate-limit status, envelope violations, predicted collision horizon), `hope_cms_signals` (Fast hazard scores, Mid intent confidence, Slow policy bundle ids), `ui_context` for workstation parity.
- **Logging:** All frames are RLDS-loggable so Trainer/Observer modes stay consistent with the XR app contract; diagnostics must include transport type and frame drops.

### Routine Preview and Apply
- **Endpoints:** `PreviewRoutine` (deterministic simulation or mock hardware) and `ApplyRoutine` (gated by safety and capability validation).
- **Safety gates:** Validate against capability manifest, enforce rate limits, and require acknowledgement of Safety workflow prompts before execution; reject when required modalities (e.g., glove presence) are missing.

### Event Hooks
- **Hardware discovery:** Subscription API emitting structured discovery events (device id, transport, firmware) for robots, Continuon Glove, and external sensors; plug-in providers can extend without app updates.
- **Mock-mode switch:** `EnableMockMode` toggles simulator feeds; telemetry and manifests must mark `mock=true` in diagnostics while keeping schemas identical for UI reuse.
- **UI actions:** Emit `UiAction` events for panel interactions, envelope edits, and safety prompts so RLDS `ui_action` is populated in workstation flows.

## HOPE/CMS Visualizations
- **Fast loop overlays:** Collision cones, gaze-weighted hazard grids, and SafetyHead overrides from `hope_cms_signals.fast`, consistent with the Fast loop responsibilities described in the HOPE/CMS note.
- **Mid loop panels:** Session intent timelines (LanguagePlanner), skill sequence confidence, and adaptive UI hints tied to `ui_context` and `action.command` history; updates rely on Mid-loop cadence from the system architecture.
- **Slow loop snapshots:** Display deployed policy bundle ids, Memory Plane merge state, and drift alerts sourced from Slow-loop summaries cached on-device, with optional comparisons pulled from Continuon Cloud when online.

## Safety Workflows
- **Persistent estop + envelope editor:** Inline estop control plus envelope visualization; edits require confirmation and emit `UiAction` audit logs.
- **Preflight checks:** On `ApplyRoutine`, verify manifest compatibility, SafetyHead readiness, estop/brake state, and timestamp coherence; block commands when diagnostics show latency spikes or missing modalities.
- **Live intervention and incidents:** Telemetry-triggered toasts highlight violations; acknowledgements are logged and forwarded to SafetyHead for adaptive margins. Violations auto-start RLDS incident capture with `episode_metadata.tags` including `safety_incident`.

## Robot API ↔ Editor Panels Binding
- **Data flow:** Panels subscribe to capability manifests and telemetry; actions from controls are normalized into Robot API commands using the same control adapters defined for XR teleop, keeping `action.command`/`ui_action` semantics aligned.
- **Panel set:** Capabilities list with schema-driven forms, telemetry dashboard with HOPE/CMS overlays, routine workspace (code/graph), and safety console (envelopes, estop, incident viewer).
- **Transport policy:** Prefer WebRTC data channels for low latency with gRPC fallback; both must propagate `client_id` and diagnostics fields into RLDS logging as described in the XR app spec.

## Mock-mode and Offline Operation
- **Deterministic playback:** Preview routines against seeded simulators or RLDS episode replay so UI components remain transport-agnostic.
- **Capability synthesis:** When offline, fabricate manifests from cached profiles with `source="mock"`; panels stay editable while highlighting unsupported features.
- **Schema fidelity:** Mock telemetry must preserve `frame_id` domains and timestamp alignment so HOPE/CMS overlays and safety gating operate identically to live runs.

## Hardware Discovery and Extensibility
- **Discovery bus:** Listen for robot endpoints (USB/Ethernet/Wi-Fi), Continuon Glove BLE advertisements, and external sensor providers; emit structured events that feed diagnostics badges and panel filters.
- **Plug-in registry:** Allow capability providers to register manifest and telemetry enrichers that conform to RLDS-compatible schemas, enabling new hardware classes without editor updates.
- **Diagnostics propagation:** Discovery/connectivity transitions update `diagnostics` and surface UI badges; panel filters let operators scope telemetry to selected devices.
