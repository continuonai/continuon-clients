# Seed Model Timeline and Playbook (Pi 5 + AI HAT + Cloud TPU)

This doc captures a short, execution-ready plan for getting the Pi 5 + AI HAT edge seed model trained and packaged, then running the first TPU-backed seed training and Hailo export.

## 4-Week Timeline (starting now)

- **Week 0 — Prep + Validation**
  - Validate hardware loop on Pi 5 + AI HAT: run `pi5_hardware_validation`, `system_health`, `tests/integration_test.py --real-hardware`; confirm RLDS step alignment within ±5 ms and HAT enumerates.
  - Freeze the exact RLDS field list for this push (use `docs/rlds-schema.md` + `continuonbrain.rlds.validators`); add a short acceptance checklist (min frames, frame_id alignment, glove placeholders).
- **Week 1 — Data Capture + Smoke Training**
  - Capture 10–20 episodes on Pi (mock if needed) via `arm_episode_recorder`/integration test; store at `/opt/continuonos/brain/rlds/episodes`.
  - Convert to TFRecord and run `jax_models.train.local_sanity_check` on Pi CPU; optionally run `trainer/local_lora_trainer --use-stub-hooks` to exercise budgets.
  - Produce an anonymized export bundle with `rlds.export_pipeline.prepare_cloud_export` to prove ingest artifact shape.
- **Week 2 — First TPU Seed Run**
  - Upload TFRecords to GCS and run the small `CoreModelConfig.development()` on TPU v5e (5–10k steps) to get a seed checkpoint.
  - Add minimal eval (loss curves, action L2 vs logged actions) and store checkpoints in `gs://…/checkpoints/core_model_v0`.
- **Week 3 — Edge Variant + HAT Packaging**
  - Export two artifacts from the TPU checkpoint: (a) JAX/CPU inference bundle for Pi sanity; (b) ONNX for Hailo. Generate a placeholder `.hef` now; if Hailo compiler is available, build the real HEF and record compiler version.
  - Drop the exported edge bundle into `/opt/continuonos/brain/model/base_model` (and adapter slots if used); verify `inference_router` selects Hailo when present and falls back cleanly when absent.
- **Week 4 — OTA + Closed Loop**
  - Shadow-test the new edge bundle on Pi with RLDS logging; compare actions vs logged actions for safety deltas.
  - Package the bundle with `edge_manifest.json` + validation reports; stage an OTA swap test via the existing manifest examples.
  - If stable, set as the “seed” for continued data capture; start the next TPU run with newly collected episodes for distillation.

## Pre-filled paths and buckets

- GCS bucket: `gs://continuon-rlds`
- RLDS prefix: `gs://continuon-rlds/rlds/episodes`
- Checkpoints prefix: `gs://continuon-rlds/checkpoints/core_model_v0`
- Local Pi paths: `/opt/continuonos/brain/rlds/episodes`, `/opt/continuonos/brain/rlds/tfrecord`, `/opt/continuonos/brain/model/base_model`, `/opt/continuonos/brain/model/base_model/hailo`
- Env default: `CONTINUON_PREFER_JAX=1`

## Seed Run Playbook (commands + expected artifacts)

> Set `PYTHONPATH=$PWD` at repo root for all Python invocations.

### 1) Pi hardware + RLDS sanity

```bash
PYTHONPATH=$PWD python -m continuonbrain.pi5_hardware_validation --log-json /tmp/pi5_check.json
PYTHONPATH=$PWD python -m continuonbrain/system_health.py --quick
PYTHONPATH=$PWD python -m continuonbrain.tests.integration_test --real-hardware
```

Artifacts:

- `/tmp/pi5_check.json` (hardware readiness)
- RLDS episodes under `continuonbrain/rlds/episodes/` (mock or real)

### 2) Episode capture (real or mock)

```bash
PYTHONPATH=$PWD python -m continuonbrain.recording.arm_episode_recorder \
  --output-dir /opt/continuonos/brain/rlds/episodes
```

Acceptance checklist: frame_id alignment within ±5 ms; glove placeholders present; required schema fields present (see `docs/rlds-schema.md`).

### 3) Convert to TFRecord (Pi)

```bash
PYTHONPATH=$PWD python -m continuonbrain.jax_models.data.tfrecord_converter \
  --input-dir /opt/continuonos/brain/rlds/episodes \
  --output-dir /opt/continuonos/brain/rlds/tfrecord \
  --compress
```

Artifacts:

- `/opt/continuonos/brain/rlds/tfrecord/*.tfrecord`

### 4) Local sanity training (Pi CPU)

```bash
PYTHONPATH=$PWD python -m continuonbrain.jax_models.train.local_sanity_check \
  --rlds-dir /opt/continuonos/brain/rlds/tfrecord \
  --obs-dim 128 --action-dim 32 --max-steps 10 --batch-size 4
```

Optional LoRA budget smoke (Torch stub):

```bash
python -m continuonbrain.trainer.local_lora_trainer \
  --config continuonbrain/configs/pi5-donkey.json --use-stub-hooks
```

### 5) Anonymize + bundle for cloud ingest

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

Artifacts:

- `export-bundle/manifest.json`
- `export-bundle/episodes/*.json`
- `export-bundle/reports/*.validation.json`

Validate episodes locally (optional, mirrored in CI):

```bash
python scripts/validate_rlds_episodes.py continuonbrain/rlds/episodes/*.json
```

### 6) Upload to GCS (TPU input)

```bash
PYTHONPATH=$PWD python -m continuonbrain.jax_models.utils.upload_to_gcs \
  --local-dir /opt/continuonos/brain/rlds/tfrecord \
  --gcs-bucket continuon-rlds \
  --gcs-prefix rlds/episodes
```

Artifacts:

- `gs://<bucket>/rlds/episodes/*.tfrecord`

### 7) TPU seed training (v5e, small config)

```bash
PYTHONPATH=$PWD python -m continuonbrain.jax_models.train.cloud.tpu_train \
  --data-path gs://continuon-rlds/rlds/episodes \
  --output-dir gs://continuon-rlds/checkpoints/core_model_v0 \
  --batch-size 256 --num-steps 10000 --learning-rate 1e-4
```

Artifacts:

- `gs://<bucket>/checkpoints/core_model_v0/*` (Orbax checkpoints)

### 8) Export inference bundles

JAX/CPU:

```bash
PYTHONPATH=$PWD python -m continuonbrain.jax_models.export.export_jax \
  --checkpoint-path gs://continuon-rlds/checkpoints/core_model_v0 \
  --output-path ./models/core_model_inference \
  --quantization fp16
```

Hailo/ONNX (generates placeholder HEF if compiler absent):

```bash
PYTHONPATH=$PWD python -m continuonbrain.jax_models.export.export_hailo \
  --checkpoint-path ./models/core_model_inference \
  --output-dir ./models/core_model_hailo
```

Artifacts:

- `./models/core_model_inference/` (CPU bundle)
- `./models/core_model_hailo/` (`.onnx`, `.hef` or placeholder, compiler logs)

### 9) Deploy to Pi and verify routing

Place artifacts:

- `/opt/continuonos/brain/model/base_model/` ← CPU bundle (and adapters if used)
- `/opt/continuonos/brain/model/base_model/hailo/` ← ONNX/HEF

Run inference router smoke:

```bash
PYTHONPATH=$PWD CONTINUON_PREFER_JAX=1 python -m continuonbrain.jax_models.export.inference_router \
  --model-dir /opt/continuonos/brain/model/base_model \
  --obs-dim 128 --action-dim 32
```

Expectation: selects Hailo when `.hef` + SDK present; otherwise clean CPU fallback.

### 10) Shadow test + OTA staging

```bash
PYTHONPATH=$PWD python -m continuonbrain.tests.integration_test --real-hardware
```

Check:

- RLDS actions vs logged actions (safety deltas)
- Package `edge_manifest.json` + validation reports with the bundle for OTA swap test.

### OTA checklist (subscription-gated)

- Use the signed bundle manifest contract in `docs/bundle_manifest.md` (checksums + signature).
- Only offer OTA to robots that are registered + paid in the ContinuonAI app.
- Keep a last-known-good bundle on device; apply only after download+verify passes.
- Include safety manifest alongside the model and preserve immutable base rules on device.
- See `docs/ota-checklist.md` for the single-stop OTA reference.
- Remove temporary artifacts (e.g., `startup-script.sh`) from buckets after runs to avoid stale scripts.

## Flutter companion (ownership/OTA) implementation backlog (active)
- Add real LAN detection and gate claim/seed on local presence.
- Persist per-robot ownership/subscription/seed status (e.g., SharedPreferences/Firestore); preload on app start.
- Align client endpoints/keys with backend (`/api/claim`, `/api/ota/install_seed`, `/api/ownership/status`); surface error messages in UI.
- Per-card busy + error/retry UI for claim/seed/status; show state badges (Unclaimed, Seed pending, Ready).
- Secure token storage (e.g., flutter_secure_storage) and clear-token action.
- Add backend-aligned error messages and retries for status/claim/seed.
- Improve LAN detection (mdns/ping gateway) and reflect in UI.
- Add reachability ping (/api/ping) consumption in the Flutter client to show device id/uptime and better error surfacing.
- Update `/api/ownership/status` on the brain to return real ownership/subscription/seed state from settings.
- Persist ownership/subscription/seed in brain settings to survive restarts (`BrainService` now loads placeholders).
- Add account hierarchy metadata to ownership (account_id/type/owner_id) and enforce in app/runtime.
