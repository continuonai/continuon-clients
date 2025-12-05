# Mock-Mode RLDS Validation Checklist

Use this checklist when testing Studio mock-mode exports to keep the RLDS schema
aligned with `docs/rlds-schema.md` and `proto/continuonxr/rlds/v1/rlds_episode.proto`.

## Steps
1. Generate or export a mock-mode episode from the Studio editor (JSON format).
2. Run the lightweight validator (root of this repo):
   ```bash
   python -m continuonbrain.rlds.validators --episode <path>/episode.json
   ```
3. Confirm the validator reports success. If failures appear, fix the generator
   before committing new mock-mode assets.

## Required fields enforced by the validator
- `metadata` block containing `xr_mode`, `control_role`, `environment_id`, proto-aligned
  `software` versions, and `tags`.
- Per-step `observation` blocks with:
  - Synchronized `video_frame_id` and `depth_frame_id` values.
  - `robot_state` timestamps/frame IDs plus `wall_time_millis` for alignment.
  - `diagnostics` covering latency, RSSI, glove drop counters, and sample rate.
- `action.command` populated and `action.source` set (e.g., `human_teleop_xr`).

## Current findings
- The validator enforces proto field names (e.g., `metadata.*`) and reports
  unexpected extra fields to catch divergence from the RLDS proto.
- Sample fixture: `continuonbrain/rlds/episodes/studio_mock_editor.json` passes
  the validator and can be used as a baseline when adjusting Studio mock flows.

## CI coverage
- `.github/workflows/rlds-validation.yml` runs the validator against the Studio
  fixture and executes the RLDS-focused pytest module. Run those same commands
  locally to reproduce CI.
