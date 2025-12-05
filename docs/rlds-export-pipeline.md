# RLDS Export Pipeline for Continuon Cloud

The RLDS export helpers in `continuonbrain/rlds/export_pipeline.py` keep
mock-mode and recorder artifacts aligned with the serialized proto while
removing personally identifying information before the bundle is prepared
for Continuon Cloud.

## Anonymization
- `anonymize_episode` deep-copies an episode payload and hashes free-form
  text maps (`step_metadata`, `action.annotation.fields`, `ui_context`)
  using salted SHA-256 digests. Configurable `drop_keys` remove
  high-risk entries such as `email`, `user_id`, and `session_id`.
- Tags that resemble user identifiers (e.g., `user:alice` or emails) are
  hashed and a summary tag `anonymization:hash:sha256:salted` is appended so
  downstream consumers know the episode has been scrubbed.
- Audio URIs are rewritten to `anonymized://<hash>` strings to avoid leaking
  workstation paths while keeping frame IDs and other RLDS-required fields
  intact.

## Validation guardrail
- `prepare_cloud_export(...)` runs `continuonbrain.rlds.validators` on every
  anonymized episode. Validation reports are written next to the bundle
  (`reports/<episode>.validation.json`) and the helper raises on schema
  violations so the upload can fail fast.
- The validator enforces the required RLDS fields listed in
  `docs/mock-mode-validation.md` to prevent anonymization from stripping
  mandatory metadata or step content.

## Cloud upload bundle layout
`prepare_cloud_export` produces a self-contained directory that callers can
hand off to the Continuon Cloud uploader or artifact store:

```
<bundle>/
  manifest.json              # environment + anonymization manifest
  episodes/<episode>.json    # anonymized RLDS payloads
  reports/<episode>.validation.json
```

The manifest records the Python/platform version, anonymization config
(salt, hashed/dropped keys), and the paths for each anonymized episode so
upload tooling has provenance for audit logs.

## Usage sketch
```
python - <<'PY'
from pathlib import Path
from continuonbrain.rlds.export_pipeline import prepare_cloud_export

bundle = prepare_cloud_export(
    episodes=[Path("./episode.json")],
    output_dir=Path("./export-bundle"),
)
print(bundle.to_json())
PY
```

After the helper finishes, run your Continuon Cloud upload routine against
`export-bundle/` to ship the sanitized dataset plus validation evidence.
