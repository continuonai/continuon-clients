# XR App Specification (Draft)

This spec maps PRD requirements to an implementable XR app. It focuses on Phase 0 contracts and Phase 1 MVP boundaries.

## Modes
- **Trainer (Mode A):** Teleop UI, live robot video/pose, action logging to RLDS with `source = "human_teleop_xr"`.
- **Workstation (Mode B):** Floating IDE/terminal/log panels; captures UI actions and focus context for RLDS.
- **Observer (Mode C):** Safety overlays, trajectory previews, annotation tools writing into `steps[*].action`.

## Functional requirements (MVP)
- XR shell with minimal panels (teleop, console/log, status).
- gRPC/WebRTC link to ContinuonBrain/OS (mock allowed in Phase 1).
- Local RLDS episode writer implementing `docs/rlds-schema.md`.
- BLE glove ingestion at ~100 Hz with MTU >= 64 bytes; parse frames into `glove.*` fields.
- Gaze ray capture from headset (origin, direction, confidence, target id) for interaction and logging.
- Audio capture (waveform or URI with frame_id) synchronized to steps.
- UI context logging for workstation mode (`observation.ui_context`) and per-step annotations.
- Manual upload/export of completed episodes to Continuon Cloud.

## Non-functional requirements
- Hard cap on end-to-end input-to-action latency for teleop (<50 ms target).
- Resilient logging: steps buffered locally; partial episodes recoverable after crashes.
- Configurable modes and environments via flags/config file (no rebuild required).

## UI primitives
- **Panels/windows:** Resizable/anchored within XR scene; focus events recorded.
- **Teleop widgets:** EE pose gizmo, joystick/gesture controls, emergency stop, mode switcher.
- **Observer overlays:** Safety zones, predicted trajectories, annotation tools (polygon/mask/flag).
- **Workstation panels:** Code/edit pane, terminal pane, rollout/log viewer.

## Services and adapters
- **ContinuonBrainClient:** gRPC/WebRTC for robot state + command streaming.
- **RldsEpisodeWriter:** Schema enforcement, local storage, export hook.
- **GloveBleClient:** BLE connect/MTU negotiation, frame parser, diagnostics.
- **FeatureFlags/Config:** Mode selection, mock/live endpoints, logging level, upload toggles.

## Testing approach (Phase 1)
- Simulated ContinuonBrain/OS service to replay robot state and accept commands.
- BLE simulator for glove frames and drop-rate/latency measurements.
- Schema validation tests for RLDS output.
