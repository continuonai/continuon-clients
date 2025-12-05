# RLDS Schema Contract (Draft)

This draft defines the RLDS-style schema ContinuonXR must emit. All episodes must validate against these rules before upload.

## Episode-level metadata
- `metadata.xr_mode`: string enum — `trainer`, `workstation`, `observer` (also mirrored into `metadata.tags` as `continuon.xr_mode:<value>` for stratification).
- `metadata.control_role`: string enum — `human_teleop`, `human_supervisor`, `human_dev_xr`.
- `metadata.environment_id`: string — deployment target or mock instance id (e.g., `lab-mock`, `pbos-dev01`).
- `metadata.software`: object — XR app version, Continuon Brain runtime version, glove firmware version.
- `metadata.tags`: list<string> — freeform labels such as task name, scene, robot type, and the natural language task instruction.

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

### `step_metadata`
- Freeform string map for per-step tags (e.g., quality flags, scene ids). Use for lightweight contextual tags without changing schema.

## File layout for episodes
- `metadata.json`: episode-level metadata.
- `steps/000000.jsonl`: ordered steps with references to binary blobs (video/depth).
- `blobs/`: raw media or compression artifacts (to be defined during implementation).

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
