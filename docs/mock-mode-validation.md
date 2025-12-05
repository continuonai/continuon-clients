# Mock-Mode RLDS Validation Checklist

Use this checklist when testing Studio mock-mode exports to keep the RLDS schema
aligned with `docs/rlds-schema.md` and `proto/continuonxr/rlds/v1/rlds_episode.proto`.

## Steps
1. Generate or export a mock-mode episode from the Studio editor (JSON format).
2. Run the lightweight validator:
   ```bash
   python -m continuonbrain.rlds.validators --episode <path>/episode.json
   ```
3. Confirm the validator reports success. If failures appear, fix the generator
   before committing new mock-mode assets.

## Required fields enforced by the validator
- `episode_metadata.continuon.xr_mode` and `episode_metadata.continuon.control_role`.
- Per-step `observation` blocks with:
  - Synchronized `video_frame_id` and `depth_frame_id` values.
  - `robot_state` timestamps/frame IDs plus `wall_time_millis` for alignment.
  - `diagnostics` including `mock_mode`, latency, glove drop counters, and sample rate.
- `action.command` populated and `action.source` set (e.g., `human_teleop_xr`).

## Current findings
- Early mock-mode exports were missing the `diagnostics.mock_mode` flag and
  `episode_metadata.continuon` block. The generator now emits both, and the
  validator surfaces explicit errors if they regress.
- Sample fixture: `continuonbrain/rlds/episodes/studio_mock_editor.json` passes
  the validator and can be used as a baseline when adjusting Studio mock flows.
