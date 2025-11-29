# XR App Contract (ContinuonXR)

This contract defines what the Android XR client must support to generate training-ready RLDS episodes and interoperate with the ContinuonBrain/OS runtime. It complements the schema defined in [`docs/rlds-schema.md`](./rlds-schema.md) and the bridge proto in [`proto/continuonxr/continuonbrain/v1/continuonbrain_link.proto`](../proto/continuonxr/continuonbrain/v1/continuonbrain_link.proto).

## Supported Modes & Required Tags

XR sessions must declare their operating mode in `episode_metadata.continuon.xr_mode` and use the corresponding action sources so downstream pipelines can segment training data:

- **Trainer (`trainer`)**
  - **Purpose:** Direct teleop and demonstration capture.
  - **RLDS tags:** `episode_metadata.continuon.control_role = "human_teleop"`; `action.source = "human_teleop_xr"`.
  - **Observations:** Headset + left/right hand poses (validity flags), egocentric RGB (+ depth when available), robot state with aligned `frame_id`, glove block populated for dexterous runs, gaze/audio as available.
  - **Actions:** Normalized control vector in `action.command` or per-arm `action.arm_commands`; optional `bimanual_intent` for coordinated tasks.

- **Workstation (`workstation`)**
  - **Purpose:** Spatial PC replacement logging workflow context.
  - **RLDS tags:** `episode_metadata.continuon.control_role = "human_dev_xr"`; `action.ui_action` entries for IDE/panel interactions.
  - **Observations:** UI context in `observation.ui_context`, gaze with `target_id` for focus tracking, optional audio/voice commands, glove block marked `valid=false` when absent.
  - **Actions:** `action.ui_action` describing panel focus, command runs, labeling, deployments; `action.command` omitted/empty unless teleop overlays are active.

- **Observer (`observer`)**
  - **Purpose:** Passive supervision and labeling of robot behavior.
  - **RLDS tags:** `episode_metadata.continuon.control_role = "human_supervisor"`; `action.annotation` populated with polygons/masks/flags.
  - **Observations:** Robot state, egocentric video/depth for annotation context; gaze optional.
  - **Actions:** Annotation payloads plus optional `ui_action` when supervision occurs inside workstation panels.

All modes must keep robot, video, and glove timestamps aligned within **±5 ms** and include `episode_metadata.software` (XR app, ContinuonBrain/OS, glove firmware versions).

## Runtime Interfaces (gRPC + WebRTC)

The XR app must expose both gRPC and WebRTC transports to ContinuonBrain/OS to allow low-latency control while keeping a typed API surface:

- **gRPC control plane** (see `ContinuonBrainBridgeService`):
  - `StreamRobotState(StreamRobotStateRequest)`: XR subscribes to structured robot telemetry (`continuonxr.rlds.v1.RobotState`) for rendering and logging; must include monotonic + wall-clock timestamps for RLDS alignment.
  - `SendCommand(SendCommandRequest)`: XR sends normalized commands derived from teleop input. Required fields: `client_id`, `control_mode` (EE velocity, joint delta, or gripper), `target_frequency_hz`, `safety` block, and oneof command payload. Responses must surface estop/rate-limit status.
- **WebRTC data channel**:
  - Mirror the above streams over a reliable/ordered data channel when available to cut latency; fall back to gRPC when negotiation fails.
  - Carry `StreamRobotStateResponse` payloads tagged with the same `client_id` to correlate with RLDS steps and log transport drops into `observation.diagnostics`.
- **Session metadata exchange**:
  - On session start, publish `episode_metadata` seeds (modes, environment id, software versions, MTU negotiated for glove) so server-side ingestion can validate consistency.
  - Heartbeat/ping messages should emit observed round-trip latency and BLE RSSI into diagnostics for logging.

## RLDS Logging Requirements

XR must emit RLDS episodes that satisfy [`docs/rlds-schema.md`](./rlds-schema.md) and the generated proto types:

- **Episode metadata**
  - `continuon.xr_mode` restricted to `trainer`, `workstation`, or `observer`.
  - `continuon.control_role` set per mode (see above) plus `environment_id`, `tags`, and `software` versions.
- **Observations (per step)**
  - Headset + hand poses with `valid` flags.
  - `egocentric_video` frame references and optional `egocentric_depth` sharing the same `frame_id` when present.
  - `robot_state` including joint positions/velocities, end-effector pose, force/torque (if available), `frame_id`, monotonic + wall-clock timestamps.
  - `glove` block with `flex`, `fsr`, `orientation_quat`, `accel`, `valid`, and inferred sample rate captured in `diagnostics.glove_sample_rate_hz`.
  - `gaze` origin/direction (normalized), `confidence`, `target_id` when used.
  - `audio` metadata (`uri`, `sample_rate_hz`, `num_channels`, `format`, `frame_id`) when voice is captured.
  - `ui_context` describing active panel/layout/focus for workstation mode.
  - `diagnostics` with latency_ms, BLE RSSI, and glove drop counters.
- **Actions (per step)**
  - `command` vector or `arm_commands` aligned to the control mode selected.
  - `source` value tied to mode (`human_teleop_xr`, `human_dev_xr`, `human_supervisor`).
  - `annotation` payloads in observer mode; `ui_action` events in workstation mode; optional `bimanual_intent` tags for coordinated manipulation.
- **File layout**
  - Persist episodes as `metadata.json` + `steps/*.jsonl` with blob references for video/depth/audio consistent with `docs/rlds-schema.md`.

## BLE Glove Expectations (Continuon Glove v0)

- **Link setup:** Connect via BLE with negotiated MTU **≥64 bytes**; capture the final MTU value in diagnostics for QA.
- **Sampling:** Target **100 Hz** streaming; log measured `glove_sample_rate_hz` and increment `glove_drops` when frames are missed beyond 5% tolerance.
- **Frame parsing:** Parse the raw frame (e.g., 45-byte payload) into normalized values: `flex[5]`, `fsr[8]`, `orientation_quat[4]`, `accel[3]`, plus a `valid` flag.
- **Fusion:** Align glove timestamps with robot/video frames (≤5 ms skew). Set `glove.valid=false` when disconnected and propagate BLE RSSI to `diagnostics.ble_rssi`.
- **Safety:** Ensure command streams honor estop/rate-limit responses before applying glove-derived commands to robots.

## Schema Alignment Checklist

Use this checklist whenever updating the app or schema to keep contracts synchronized:

- [ ] Enumerations and field names in this contract match [`docs/rlds-schema.md`](./rlds-schema.md) (`xr_mode`, `control_role`, pose, glove, diagnostics blocks).
- [ ] Required gRPC methods (`StreamRobotState`, `SendCommand`) and `ControlMode`/`ReferenceFrame` values align with [`continuonbrain_link.proto`](../proto/continuonxr/continuonbrain/v1/continuonbrain_link.proto).
- [ ] File layout and timestamp tolerance (≤5 ms skew) reflect the validation rules in [`docs/rlds-schema.md`](./rlds-schema.md).
- [ ] BLE expectations (MTU, sample rate, drop logging) are captured in both `diagnostics` and `glove` sections of the schema.
- [ ] Mode-specific `action.source` and `action.ui_action`/`annotation` usage are covered by automated RLDS validation when added.
