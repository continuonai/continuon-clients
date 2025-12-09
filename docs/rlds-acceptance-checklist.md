# RLDS Episode Acceptance Checklist (Pi 5 + AI HAT seed runs)

Use this checklist before promoting episodes for cloud upload or TPU training. It aligns with `docs/rlds-schema.md` and `continuonbrain.rlds.validators`.

## Required structure
- `metadata.json` present; `steps/*.jsonl` present; optional `blobs/` for media.
- Episode-level fields: `metadata.xr_mode`, `metadata.control_role`, `metadata.environment_id`, `metadata.tags`, `metadata.software.*`, `metadata.start_time_unix_ms`.

## Step alignment & timing
- Frame IDs align across `observation.egocentric_video`, `observation.egocentric_depth` (when present), and `robot_state.*.frame_id`.
- Timestamps between vision frames and robot_state are within ±5 ms.
- Monotonic time never decreases across steps.

## Observation completeness
- `observation.robot_state` present; per-arm blocks include `joint_positions` and `end_effector_pose`; `frame_id` set.
- Gloves: if absent, emit placeholders with `glove.valid=false` (keep fields so schema stays stable).
- `language_instruction` present on first step; `step_metadata` map allowed for flags.

## Actions
- `action.command` present and normalized (document the convention: e.g., -1..1 or m/s).
- If bimanual, `action.arm_commands.left_arm/right_arm` frame_id aligns with observation.

## Health & safety markers
- Safety violations, dropouts, or notable events recorded in `step_metadata` (e.g., `safety_violation=true`, `frame_drop=true`).

## Counts and quality
- No empty episodes; at least N steps for the task (pick a floor, e.g., ≥50 for short clips, higher for driving runs).
- Video/depth/glove drop rate acceptable (warn if glove coverage <95% when expected).
- File sizes sane (no zero-byte blobs).

## Anonymization (before upload)
- Run `prepare_cloud_export` to hash PII-like tags and rewrite audio URIs.
- Ensure `manifest.json` in the export bundle records salt/hash config.

## Quick validation commands

Schema validation (local episodes):
```bash
PYTHONPATH=$PWD python - <<'PY'
from pathlib import Path
from continuonbrain.rlds.validators import validate_episode

episodes = list(Path("/opt/continuonos/brain/rlds/episodes").glob("*.json"))
for ep in episodes:
    report = validate_episode(ep)
    print(ep.name, "OK" if report.valid else "FAIL", report.errors)
PY
```

Full export + validate + anonymize:
```bash
PYTHONPATH=$PWD python - <<'PY'
from pathlib import Path
from continuonbrain.rlds.export_pipeline import prepare_cloud_export

bundle = prepare_cloud_export(
    episodes=list(Path("/opt/continuonos/brain/rlds/episodes").glob("*.json")),
    output_dir=Path("./export-bundle"),
)
print(bundle.to_json())
PY
```

## Promote/Reject gate
- Promote if: schema valid, timing alignment OK, min-step floor met, no safety violations flagged as hard failures, anonymization applied for upload.
- Reject or quarantine if: frame_id/timestamp mismatch >5 ms, missing required fields, empty/short episodes, or safety violations not annotated.

