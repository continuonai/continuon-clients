# Human-Centric Data Flow (One Brain, Many Shells)

This note captures how ContinuonXR feeds unified RLDS episodes across all human-centric modes and aligns with the continuonos principle **“One Brain, Many Shells.”** The XR stack is a reusable shell that any continuonos deployment (headset, robot-mounted rig, or offline video ingest) can wear to produce consistent data.

## Modes -> RLDS tags
- Trainer (Mode A): `episode_metadata.continuon.xr_mode = "trainer"`, `control_role = "human_teleop"`, `action.source = "human_teleop_xr"`.
- Workstation (Mode B): `xr_mode = "workstation"`, `action.source = "human_dev_xr"`, `observation.ui_context` populated, `ui_action` events.
- Observer (Mode C): `xr_mode = "observer"`, annotations live in `action.annotation`.
- YouTube/TV (Mode D): `xr_mode = "youtube_tv"`, inferred actions and 3D cues injected offline.

## Observation blocks (synchronized)
- Poses: `xr_headset_pose`, `xr_hand_*_pose` with validity flags.
- Gaze: `gaze.origin`, `gaze.direction` (unit), `confidence`, `target_id`.
- Vision: `egocentric_video` + `egocentric_depth` (frame_id aligned).
- Audio: `audio.uri` (or buffer), `sample_rate_hz`, `num_channels`, `format`, `frame_id`.
- Glove: `glove.flex`, `glove.fsr`, `glove.orientation_quat`, `glove.accel`, `glove.valid`.
- Robot state: joints/EE pose/gripper from ContinuonBrain/OS.
- UI context: active panel, layout, focus for Mode B.
- Diagnostics: latency, BLE RSSI, drop counts, glove sample rate.

## Action and step metadata
- `action.command`: normalized control vector (teleop) or UI commands (workstation) or annotations (observer).
- `action.source`: distinguishes teleop vs dev vs inferred.
- `step_metadata`: freeform string map for per-step tags (e.g., quality flags, scene ids, safety state).

## Missing data and synthesis
- Each block is optional with validity flags (e.g., `glove.valid = false`); absent sensors keep schema intact.
- Cloud Factory can backfill or hallucinate missing tactile/audio cues while preserving original metadata for provenance.

## Robot-worn XR (ego-centric shell)
- A robot-mounted XR rig can emit the same observation blocks, giving ego-centric video/pose/gaze for self-supervised episodes.
- Spatial HUD/overlays can be driven from world-model predictions to visualize safety envelopes before actuation.
- Workflow/introspection logs (Mode B) can be captured from on-robot debugging sessions to train the Continuon assistant.

## Implementation hints (repo touchpoints)
- Schema: `docs/rlds-schema.md`, `proto/rlds_episode.proto`.
- App wiring: `apps/continuonxr/src/main/java/com/continuonxr/app/` (teleop, logging, glove, connectivity).
- Validation and file sink: `RldsEpisodeWriter` + `RldsValidator`.
- Next additions: Jetpack XR/SceneCore input for poses/gaze, audio capture pipeline, and ContinuonBrain/OS gRPC client/mock.
