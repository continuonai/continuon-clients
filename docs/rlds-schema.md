# RLDS Schema Contract (Draft, v1.1)

This draft defines the RLDS-style schema ContinuonXR must emit. All episodes must validate against these rules before upload.

**Versioning**
- Episodes SHOULD include `metadata.schema_version` (semantic-ish string, e.g. `"1.0"` / `"1.1"`). When absent, consumers MUST treat as `"1.0"`.
- All schema additions in v1.1 are **optional** and MUST be ignored by older consumers.

## Pi-side RLDS schema (Continuon Brain runtime)
The Pi-side recorder (Continuon Brain runtime) MUST mirror the headset schema so OTA and cloud handoff stay lossless. Minimum f
ields:

- **Episode metadata**
  - `metadata.xr_mode` and `metadata.control_role` copied from the headset and persisted in the edge bundle.
  - `metadata.robot_id`, `metadata.robot_model`, and `metadata.frame_convention` (e.g., `base_link`, `camera_link`).
  - `metadata.start_time_unix_ms` and `metadata.duration_ms` for lifecycle auditing.
- **Egocentric sensors**
  - `observation.egocentric_video`: RGB frame in the Pi camera frame; include `frame_id`, `timestamp_ns`, and `camera_intrinsic
    s` when available.
  - `observation.egocentric_depth`: optional depth frame aligned to the same `frame_id` and `timestamp_ns`.
- **Proprioceptive state**
  - `observation.robot_state` MUST include per-arm `joint_positions`, `joint_velocities`, and `end_effector_pose` in the Pi ca
    mera frame (or explicitly specify `frame_id`).
  - `observation.robot_state.*.gripper_state` is recommended for grasp reproducibility.
- **Glove placeholders**
  - When gloves are not paired, emit `glove.valid=false` and keep the blocks present so downstream consumers stay schema-stable
    .
- **Normalized actions**
  - `action.command` MUST be normalized (e.g., EE velocity in m/s or unitless deltas 0..1 with explicit `command_type`).
  - For mirrored/bimanual rigs, include `action.arm_commands` with the same `frame_id` convention as observations.
- **Episode packaging**
  - Episodes are written as the same `metadata.json` + `steps/*.jsonl` layout defined below; blobs reside in `blobs/` with fram
    e-aligned filenames.

## Episode-level metadata
- `metadata.xr_mode`: string enum — `trainer`, `workstation`, `observer` (also mirrored into `metadata.tags` as `continuon.xr_mode:<value>` for stratification).
- `metadata.control_role`: string enum — `human_teleop`, `human_supervisor`, `human_dev_xr`.
- `metadata.environment_id`: string — deployment target or mock instance id (e.g., `lab-mock`, `pbos-dev01`).
- `metadata.software`: object — XR app version, Continuon Brain runtime version, glove firmware version.
- `metadata.tags`: list<string> — freeform labels such as task name, scene, robot type, and the natural language task instruction.

### v1.1 optional metadata blocks (autonomy + continuous learning)
- `metadata.schema_version`: string (see Versioning above).
- `metadata.episode_id`: string stable identifier (`ep_<ts>` or UUID). If absent, derive from filename.
- `metadata.robot_id`: string stable device/robot identity (may be hardware serial hashed).
- `metadata.robot_model`: string (e.g., `pi5-donkeycar`, `so-arm101`).
- `metadata.frame_convention`: string (e.g., `base_link`, `camera_link`) to disambiguate pose frames.
- `metadata.start_time_unix_ms`: int64 wall-clock for lifecycle auditing.
- `metadata.duration_ms`: int64.
- `metadata.capabilities`: object describing the brain+body capabilities for this run (used for conditioning and safety).
  - `compute`: `{ "device": "pi5|jetson|laptop|tpu", "ram_gb": 8, "gpu": "RTX2050", "dtype": "bf16|fp16|fp32" }`
  - `sensors`: `{ "rgb": true, "depth": true, "audio": false, "imu": false, "glove": false }`
  - `actuators`: `{ "drive": true, "arm": false, "gripper": false }`
  - `limits`: `{ "allow_motion": false, "max_speed_mps": 0.0 }`
- `metadata.owner`: **optional, local-only** identity and preference block used for supervised personalization.
  - `owner_id`: string (prefer stable hash; avoid raw serials)
  - `display_name`: string (PII)
  - `preferred_name`: string (PII)
  - `roles`: list<string> (e.g., `["creator", "owner"]`)
  - `preferences`: map<string,string> (e.g., `{"tone":"curious","ask_clarifying":"true"}`)
- `metadata.safety`: required for public/share flows; recommended always.
  - `content_rating`: `{ "audience": "general|teen|adult", "violence": "none|mild|graphic", "language": "clean|some|strong" }`
  - `pii_attestation`: `{ "pii_present": true|false, "faces_present": true|false, "name_present": true|false, "consent": true|false }`
  - `pii_cleared`: bool (default false when faces/plates/audio present)
  - `pii_redacted`: bool
  - `pending_review`: bool
- `metadata.share`: governs whether an episode is publishable/listable (Continuon Cloud).
  - `public`: bool
  - `slug`: string
  - `title`: string
  - `license`: string (SPDX-ish, e.g., `CC-BY-4.0`, `CC-BY-NC-4.0`, `Proprietary`)
  - `tags`: list<string>
- `metadata.provenance`: object
  - `origin`: string (e.g., `origin:pi5:oakd`, `origin:studio:windows:webcam`)
  - `source_commit`: string git SHA (optional)
  - `source_host`: string
  - `notes`: string

## Step structure
Each step is timestamped in monotonic time (ns) and wall-clock time (ms) for reconciliation. Steps must align robot state, video, glove, and pose to within 5 ms.

```
step {
  observation { ... }
  action { ... }
  reward: optional float (future use)
  is_terminal: optional bool (default false)
}
```

### `observation`
- `xr_headset_pose`: position (x, y, z) + orientation quaternion (x, y, z, w).
- `xr_hand_right_pose` and `xr_hand_left_pose`: position + orientation quaternion; include validity flags.
- `gaze`: optional block with `origin` (x, y, z), `direction` (unit vector), `confidence` (0..1), and `target_id` (UI element/object id) for gaze-based interactions.
- `audio`: optional block with `uri` or inline buffer reference plus `sample_rate_hz`, `num_channels`, `format`, and `frame_id`.
- `egocentric_video`: URI or handle to synced frame buffer; include frame_id.
- `egocentric_depth`: optional; same frame_id as video when present.
- `robot_state`: structured per-arm state blocks plus coordination tags.
  - `left_arm` / `right_arm` (optional objects; default null when hardware absent):
    - `joint_positions`: float[] ordered by robot model.
    - `joint_velocities`: float[] aligned to `joint_positions` order.
    - `end_effector_pose`: position (x, y, z) + orientation quaternion (x, y, z, w) in `frame_id`.
    - `force_torque`: optional 6D vector `[Fx, Fy, Fz, Tx, Ty, Tz]` at the wrist in `frame_id`.
    - `gripper_state`: optional object with `position` (scalar/opening), `velocity`, and `is_closed` boolean.
    - `frame_id`: string indicating the robot-centric frame for pose/FT data; must align with vision frames (see validation).
  - `coordination`: optional object to capture bimanual coupling and intents.
    - `mode`: enum (`independent`, `mirrored`, `coordinated_task`, `handoff`) describing synchronization intent.
    - `shared_task_tag`: optional string label for the current coordinated maneuver.
    - `phase`: optional enum (`approach`, `engage`, `manipulate`, `release`) for downstream alignment.
- `glove.flex`: float[5] (normalized 0..1).
- `glove.fsr`: float[8] (normalized 0..1).
- `glove.orientation_quat`: float[4].
- `glove.accel`: float[3] (m/s^2).
- `glove.valid`: boolean indicating whether glove data is present for this step.
- `ui_context`: optional block for workstation mode (active panel id, layout state, focus context).
- `language_instruction`: first step should echo the task instruction string; later steps may override if sub-tasks change.
- `step_metadata`: per-step string map for quick flags/ids without schema changes. Use to surface `ball_reached=true` terminal markers and `safety_violations` lists when clamps/firewalls trip.
- `diagnostics`: drop counters, latency measurements, BLE RSSI.

#### v1.1 optional observation blocks (multimodal + dialog + tools)
- `media`: optional object with per-step references to recorded blobs; all URIs are local paths or signed URLs.
  - `rgb`: `{ "uri": "...", "frame_id": "...", "timestamp_ns": 0, "width": 0, "height": 0, "format": "jpeg|png|raw" }`
  - `depth`: `{ "uri": "...", "frame_id": "...", "timestamp_ns": 0, "width": 0, "height": 0, "format": "png16|raw16", "units": "mm" }`
  - `audio`: `{ "uri": "...", "frame_id": "...", "timestamp_ns": 0, "sample_rate_hz": 0, "num_channels": 0, "format": "pcm16le|wav" }`
- `dialog`: optional block for HOPE-style training conversations.
  - `speaker`: `user|assistant|system`
  - `text`: string
  - `turn_id`: string (stable within episode)
  - `conversation_id`: string (stable across episodes, optional)
- `world_model`: optional block for self-learning signals.
  - `latent_tokens`: list<int> (VQ/VAE codes)
  - `surprise`: float (prediction error proxy)
  - `belief_state_id`: string (planner state pointer)
- `segmentation`: optional block for per-frame segmentation artifacts (e.g., SAM3), generated offline after capture.
  - `model`: string (e.g., `facebook/sam3`)
  - `prompt`: string (text prompt used to produce masks; store for reproducibility/distillation)
  - `masks`: list of `{ "uri": "...", "frame_id": "...", "format": "png", "instance_id": int32, "score": float }`
  - `boxes_xyxy`: list of `{ "x1": float, "y1": float, "x2": float, "y2": float, "instance_id": int32, "score": float }`
  - `notes`: optional string

### `action`
- `command`: normalized control vector (e.g., EE velocity or joint delta).
- `arm_commands`: optional object (default null) for per-arm actuation.
  - `left_arm` / `right_arm`: optional blocks containing
    - `joint_torques`: optional float[] (Nm) matching `robot_state.*.joint_positions` order.
    - `joint_pwm`: optional float[] (0..1) for low-level driver passthrough when torques unavailable.
    - `gripper_command`: optional scalar (0..1) or struct with `position` and `force` for the specified gripper.
  - `sync_intent`: optional enum (`independent`, `synchronous_execute`, `mirror_left_to_right`, `mirror_right_to_left`, `bimanual_task`) to annotate bimanual coordination.
- `bimanual_intent`: optional string tag describing the shared task (e.g., `two_hand_lift`, `handoff`).
- `source`: string — must be `human_teleop_xr` for Mode A.
- `annotation`: optional; polygons/masks/flags for Mode C supervision.
- `ui_action`: optional; workstation/IDE context events for Mode B (`open_panel`, `run_command`, `label_run` etc.).

#### v1.1 optional action blocks (planner + tools + conversation)
- `dialog`: optional response block mirroring observation dialog (assistant turn).
  - `speaker`: `assistant`
  - `text`: string
  - `turn_id`: string
- `planner`: optional planner outputs for autonomy-eligible logs.
  - `intent`: string
  - `plan_steps`: list<string>
  - `selected_skill`: string
  - `confidence`: float (0..1)
- `tool_calls`: optional list of tool call records (for later distillation/replay).
  - each item: `{ "tool": "http|mcp|filesystem|...", "name": "...", "args_json": "...", "result_json": "...", "ok": true|false }`

### `step_metadata`
- Freeform string map for per-step tags (e.g., quality flags, scene ids). Use for lightweight contextual tags without changing schema.

## File layout for episodes
- `metadata.json`: episode-level metadata.
- `steps/000000.jsonl`: ordered steps with references to binary blobs (video/depth).
- `blobs/`: raw media or compression artifacts (to be defined during implementation).

## Edge bundle manifest (for OTA + cloud handoff)
Each exported bundle must include `edge_manifest.json` at the root to support Pi-side OTA swaps and cloud verification.

- `version`: semantic version of the bundle schema (e.g., `1.0.0`).
- `model_name`: human-readable skill/policy name to assist operator audits.
- `tflite_path`: relative path to the packaged TFLite (or model archive) inside the bundle.
- `dependencies`: map of runtime dependencies with versions and URIs (e.g., `{ "pose_decoder": { "version": "0.2.1", "uri": ""
  } }`).
- `signature`: cryptographic signature or checksum block covering `metadata.json`, `steps/`, and the model artifact for tamper e
  vidence.
- `created_at_unix_ms`: creation timestamp for lifecycle plan alignment.
- `source`: string identifying the producer (`continuon.brain_runtime` vs `continuon.xr_app`).

## Validation rules (MVP)
- Reject episodes missing required fields above.
- Reject if any step has mismatched frame_ids across video/depth/robot state.
- Reject if `robot_state.left_arm.frame_id` or `robot_state.right_arm.frame_id` timestamps deviate from corresponding vision `frame_id` by more than ±5 ms when present.
- Warn (but keep) if glove frames drop below 95% of expected count per episode.
- Ensure MTU and sample rate are logged for glove BLE for QA.
- Gaze vectors (when present) must be normalized and include origin/direction with 3 floats each.
- Audio (when present) must specify sample_rate_hz and num_channels.

## Extensibility principles
- Prefer extending `observation` with new nested blocks (e.g., `gaze`, `audio`) rather than external sidecar files so schema remains self-contained.
- Keep new fields optional with defaults so episodes remain readable when sensors are absent.

## PII + public safety (local-first defaults)
- Episodes that include faces, names, license plates, or raw audio SHOULD default to:
  - `metadata.safety.pii_attestation.pii_present=true`
  - `metadata.safety.pii_cleared=false`
  - `metadata.safety.pending_review=true`
  - `metadata.share.public=false`
- Public listing MUST require `pii_cleared=true` and `pending_review=false` (and prefer redacted assets when `pii_redacted=true`).
