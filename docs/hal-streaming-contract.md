# Hardware Abstraction Layer (HAL) Streaming Contract

This specification models sensors and actuators as typed streams to align with the Continuon Brain runtime safety kernel. Streams follow the familiar `stdin` / `stdout` / `stderr` pattern but are typed to encode safety and timing guarantees.

## Stream model
- **Input streams (`stdin`-like):** typed sensor channels (e.g., `rgb:video/egocentric`, `depth:video/depth`, `imu:motion`, `gaze:vector`, `glove:fingers`, `joint_state:robot`, `ui_context:text/json`). Producers must declare rate, units, and validity flags; consumers negotiate downsampling in the Mid/Slow loops.
- **Output streams (`stdout`-like):** actuator intents emitted by planners/policies (`trajectory`, `gripper`, `torque_profile`, `beep/haptics`). Each message carries `target_frame`, `constraints` (limits, velocity/accel caps), and `valid_until` timestamps so the safety kernel can expire stale commands.
- **Diagnostics (`stderr`-like):** safety kernel and driver warnings (limit hits, thermal throttling, calibration drift, watchdog preemption). Diagnostics are treated as first-class streams so they can be logged into RLDS episodes for replay.

Streams are **typed** via `{channel}:{mime}` pairs and must include:
- `schema_ref` to the RLDS block or device descriptor.
- `ts_unix_ms` and `seq` for ordering.
- `valid` flags per field so backfill/inference never silently overwrites raw signals.

## Syscall surface (enforced by the safety kernel)
The safety kernel exposes a minimal syscall surface to shell drivers and planners. Calls are validated against per-shell capabilities, joint limits, and safety policies before scheduling onto the fast loop.

| Syscall | Description | Required fields | Validation path |
| --- | --- | --- | --- |
| `move_to` | Cartesian or joint-space move with timing | `pose` or `joint_positions`, `frame`, `max_vel`, `max_accel`, `deadline_ms` | Limits clamped; self-collision + workspace checked; fails closed if constraints conflict |
| `set_torque_profile` | Adjust torque/impedance per joint | `joint_id`, `torque_nm`, `stiffness`, `damping`, `duration_ms` | Compared to per-joint envelope; thermal/voltage budget enforced; mid-loop smoothing |
| `apply_twist` | Short horizon twist for base/arm | `twist_linear`, `twist_angular`, `duration_ms` | Saturated by fast loop; inertial/IMU sanity; drops if IMU/odom stale |
| `open_gripper` / `close_gripper` | Gripper actuation with force limits | `grip_force_n`, `width_mm`, `timeout_ms` | Force/width clipped to hardware table; retries gated; stalls reported on `stderr` |
| `set_mode` | Mode handoff (autonomy/manual/estop) | `mode`, `reason`, `requested_by`, `valid_until` | Authority + pairing token verified; pending moves canceled on estop |
| `set_torque_guard` | Safety envelope for subsequent calls | `joint_limits`, `temp_max_c`, `current_max_a` | Guard rails cached; all subsequent calls filtered through guard |
| `sync_stream` | Declare rate/latency budgets for streams | `channel`, `rate_hz`, `max_latency_ms`, `valid` | Used to backpressure capture; denial triggers degraded recording profile |

### Validation and error paths
- **Static validation (mid/slow loop):** schema completeness, auth/ownership, capability checks (per-shell joints, DoF), and budget envelopes (thermal/current). Denied calls return `EINVAL` with a structured reason; nothing reaches the fast loop.
- **Dynamic validation (fast loop):** real-time limit enforcement, watchdog timers, and collision/IMU health. Violations trigger `EFAULT` + `stderr` diagnostics and may inject `set_mode(estop)` if persistent.
- **Fallbacks:**
  - If `move_to` is denied, the safety kernel emits a `stderr` record with `reason` and the driver may re-submit with tighter constraints.
  - If diagnostics exceed rate limits, summaries are batched into RLDS `diagnostics` blocks to avoid flooding the bus.

## Mapping to loop structure
- **Fast loop (1–5 ms):** Executes admitted syscalls as bounded fragments (segment-level interpolation, torque clamp) and streams diagnostics. Only uses already-validated parameters.
- **Mid loop (20–50 ms):** Performs intent shaping (e.g., motion planning, jerk limiting) and safety pre-checks. It owns `sync_stream` budgeting and coalesces intents from multiple heads.
- **Slow loop (100–500 ms):** Mode/state management, capability negotiation, and long-horizon planners. It publishes policy outputs to the Mid loop via typed `stdout` streams and listens for `stderr` to adjust planning (e.g., degrade to lower-speed profile).

## Shell driver adaptation examples
- **Humanoid shell:** The driver maps `move_to` Cartesian intents to joint targets using the humanoid’s kinematics table and applies `set_torque_profile` to ankle/knee joints only when the safety kernel reports a stable IMU. If the torso thermal budget is low, the driver pre-scales `torque_nm` before submitting to avoid kernel rejection.
- **Arm-on-base shell:** For a 6-DoF arm on a mobile base, the driver splits intents: base twist via `apply_twist` (with odom guard) and arm trajectory via `move_to`. When `stderr` signals proximity-stop from the base, the driver inserts a `set_mode(estop)` until the base reports clear, preventing arm drift.
- **XR teleop shell:** Teleop packets from the Continuon XR app are treated as `stdin` streams (`ui_action`, `hand_pose`). The driver converts them into Mid-loop intents with capped `max_vel` to ensure haptic feedback latency stays under the declared `sync_stream` budget. Any dropped packets are reflected as `valid=false` to keep the fast loop from extrapolating unsafe motions.

## RLDS and provenance
All streams (including denied syscalls and diagnostics) should be logged into RLDS `diagnostics` or `action` blocks to preserve provenance for replay and cloud-side safety retraining. See `docs/rlds-schema.md` for field mappings and `docs/bundle_manifest.md` for how admitted syscalls are represented in edge bundles.
