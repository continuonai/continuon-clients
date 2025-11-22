# RLDS Schema Contract (Draft)

This draft defines the RLDS-style schema ContinuonXR must emit. All episodes must validate against these rules before upload.

## Episode-level metadata
- `episode_metadata.continuon.xr_mode`: string enum — `trainer`, `workstation`, `observer`.
- `episode_metadata.continuon.control_role`: string enum — `human_teleop`, `human_supervisor`, `human_dev_xr`.
- `episode_metadata.environment_id`: string — deployment target or mock instance id (e.g., `lab-mock`, `pbos-dev01`).
- `episode_metadata.software`: object — XR app version, PixelBrain/OS version, glove firmware version.
- `episode_metadata.tags`: list<string> — freeform labels such as task name, scene, robot type.

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
- `egocentric_video`: URI or handle to synced frame buffer; include frame_id.
- `egocentric_depth`: optional; same frame_id as video when present.
- `robot_state`: joints, end-effector pose, gripper state, velocities as exposed by PixelBrain/OS.
- `glove.flex`: float[5] (normalized 0..1).
- `glove.fsr`: float[8] (normalized 0..1).
- `glove.orientation_quat`: float[4].
- `glove.accel`: float[3] (m/s^2).
- `diagnostics`: drop counters, latency measurements, BLE RSSI.

### `action`
- `command`: normalized control vector (e.g., EE velocity or joint delta).
- `source`: string — must be `human_teleop_xr` for Mode A.
- `annotation`: optional; polygons/masks/flags for Mode C supervision.
- `ui_action`: optional; workstation/IDE context events for Mode B (`open_panel`, `run_command`, `label_run` etc.).

## File layout for episodes
- `metadata.json`: episode-level metadata.
- `steps/000000.jsonl`: ordered steps with references to binary blobs (video/depth).
- `blobs/`: raw media or compression artifacts (to be defined during implementation).

## Validation rules (MVP)
- Reject episodes missing required fields above.
- Reject if any step has mismatched frame_ids across video/depth/robot state.
- Warn (but keep) if glove frames drop below 95% of expected count per episode.
- Ensure MTU and sample rate are logged for glove BLE for QA.

