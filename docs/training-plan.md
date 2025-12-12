# End-to-End Training Plan (Pi5 → Cloud → Pi OTA)

Scope: Pi 5 (8 GB) running Continuon Brain runtime + HOPE seed, logging RLDS, promoting adapters to Continuon Cloud for TPU training, and returning a signed Hope Model v1 edge bundle that replaces the seed. See `docs/seed-model-plan.md` for the detailed Fast/Mid/Slow seed playbook (Pi capture + GCP TPU export) that this plan references. The on-device path is now JAX-first (CoreModel + HOPE states); the PyTorch LoRA trainer remains an optional fallback.

## Phases & Gates

1) **Hardware init & health**
   - Run: `PYTHONPATH=$PWD python3 continuonbrain/system_health.py --quick`
   - Run: `python -m continuonbrain.pi5_hardware_validation --log-json /tmp/pi5_check.json`
   - Gate: camera + PCA9685 detected; timestamp skew ≤5 ms; battery/thermal OK.

2) **Seed model boot (headless)**
   - Start via systemd `continuonbrain-startup.service` (headless, background trainer off by default).
   - Gate: Robot API up, LAN discovery reachable, AUTONOMOUS mode armed but idle.

3) **Data capture (seed → RLDS)**
   - Collect ≥16 episodes (pi5-donkey config) using XR/companion/manual teleop.
   - Target: 20–50 Hz action/state, depth RGB-D at stable 640x480@30; sync within ≤5 ms.
   - Storage: `/opt/continuonos/brain/rlds/episodes/…` (JSON/JSONL or TFRecord accepted).

4) **On-device JAX sanity check (Pi)**
   - Config: JAX `arch_preset=pi5` (fast/mid HOPE state sizes). Optional `--sparsity-lambda` to mirror cloud regularization.
   - Run: `python -m continuonbrain.jax_models.train.local_sanity_check --rlds-dir /opt/continuonos/brain/rlds/episodes --arch-preset pi5 --max-steps 8 --batch-size 4 --metrics-path /tmp/jax_sanity.csv --checkpoint-dir /tmp/jax_ckpts`
   - Notes: Automatically falls back to JSON episode loading when TensorFlow is unavailable; writes lightweight pickle checkpoints if requested.
   - Gate: loss finite and decreasing; state shapes match config; no CRITICAL resource alerts.

5) **Proof-of-learning (artifact)**
   - Run: `python prove_learning_capability.py` (background learner with mocked resource headroom).
   - Output: `proof_of_learning.json` + console verdict showing parameter deltas and novelty.
   - Gate: non-zero learning updates and parameter change >1e-9; stash artifact for bundle audit.

6) **RLDS prep for cloud (TFRecord)**
   - Convert/validate: `python -m continuonbrain.jax_models.data.tfrecord_converter --input-dir /opt/continuonos/brain/rlds/episodes --output-dir /opt/continuonos/brain/rlds/tfrecord --compress`
   - Optional: `python -m continuonbrain.rlds.export_pipeline --episodes-dir ... --output-dir ...` to produce anonymized manifests.
   - Env tag: `CONTINUON_EXPORT_ORIGIN=pi5` (default) to stamp manifest.
   - Gate: validation reports OK; manifest written under export dir.

7) **Cloud training (Continuon Cloud / TPU)**
   - Upload RLDS TFRecords + any candidate adapters to cloud bucket (per cloud ingest docs).
   - Run JAX/TPU trainer: `python -m continuonbrain.run_trainer --trainer jax --mode tpu --data-path gs://... --output-dir gs://... --config-preset tpu --num-steps 10000` (matches the seed playbook’s TPU step).
   - Output: Hope Model v1 artifacts (TFLite/ONNX/Hailo-HEF placeholders) + safety manifest.
   - Gate: eval metrics pass (set project-specific thresholds), bundle passes integrity/signature.

8) **Edge bundle assembly**
   - Follow `docs/bundle_manifest.md`; include:
     - Base/adapter weights (TFLite or ONNX), optional `.hef` if compiled.
     - `edge_manifest.json` with version, compat, signatures, preferred backends, safety manifest.
     - Training evidence: `proof_of_learning.json`, JAX sanity metrics, cloud eval summaries.
   - Sign bundle; store last-known-good for rollback.

9) **OTA back to Pi**
   - Deliver via companion app OTA flow (ownership + subscription gated).
   - On Pi: download → verify signature/checksum → stage A/B → hot-swap.
   - Gate: startup manager passes health with new model; control loop latency within budget; fallback preserved.

10) **Post-deploy validation**
   - Run: `PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --real-hardware`
   - Smoke inference: `python -m continuonbrain.jax_models.export.infer_cpu --model-path ./models/core_model_inference --obs-dim 128 --action-dim 32`
   - Spot-check RLDS logging still aligned; run short teleop and autonomous demos.
   - If regressions: rollback to prior bundle, keep failing bundle + logs for cloud triage.

## Operational parameters (defaults)

- Headless boot: `CONTINUON_HEADLESS=1`
- Background trainer: `CONTINUON_ENABLE_BACKGROUND_TRAINER=0` (enable if memory headroom allows)
- JAX preference: `CONTINUON_PREFER_JAX=1`
- Export origin tag: `CONTINUON_EXPORT_ORIGIN=pi5`
- RAG/world-model assets: wiki/episodic shards are referenced via `/opt/continuonos/brain/memory/wiki/manifest.json` (see `docs/wiki-rag-plan.md` and `continuonbrain/configs/pi5-rag.json`); latent tokens from the VQ-VAE on the HAT are logged in RLDS and used by the predictor for surprise-based grounding.

## Ownership of steps

- Edge (Pi5): Health, capture, on-device adapter training, export prep, OTA apply.
- Cloud: RLDS ingest/clean, TPU training, bundle/sign, OTA serve.

## HOPE/CMS boundaries for this plan

- **Fast/Mid on edge:** Fast loop stays on-device for reflexive safety + control; Mid loop runs bounded adapter training (LoRA) against the captured RLDS. These two loops are the only ones that ever train on the Pi 5 in this plan.
- **Slow in cloud:** Slow loop retrains spectral/SSM cores and merges Memory Plane state into the OTA bundle; the Pi only applies the signed output.
- **AINA hook:** Manipulation runs may use the AINA policy head (see `continuonbrain/aina_impl/`) inside the Fast loop, with Mid-loop LoRA updates on the same head. Cloud Slow-loop refreshes the underlying spectral cores and AINA head before shipping the bundle.

## RLDS capture and eval logging (Pi → Cloud)

- **Episode capture:** `continuonbrain/recording/arm_episode_recorder.py` records synchronized RGB/depth frames, servo state, and teleop or policy commands as RLDS steps with per-frame timestamps, action provenance, optional audio buffers, and episode-level tags so slow-loop trainers can stratify by xr_mode/control_role before promotion.【F:continuonbrain/recording/arm_episode_recorder.py†L22-L115】
- **Lightweight logging for JAX:** For quick Pi-side sanity checks, `continuonbrain/jax_models/data/episode_logger.py` writes JSON/JSONL episodes (observation/action/reward/done) with per-step timestamps and metadata that can be converted to TFRecord ahead of TPU training and OTA bundle assembly.【F:continuonbrain/jax_models/data/episode_logger.py†L1-L93】【F:continuonbrain/jax_models/data/episode_logger.py†L98-L164】
- **Eval traces:** HOPE eval runners log graded Q&A directly as RLDS episodes. `continuonbrain/eval/hope_eval_runner.py` captures each prompt/answer, whether a fallback model was used, and the fallback order for replay; `continuonbrain/eval/multi_model_compare_eval.py` runs HOPE plus Gemma/Gemini hints and writes a winner decision/rationale per step so cloud-side voters can revisit or override the heuristic during slow-loop merging.【F:continuonbrain/eval/hope_eval_runner.py†L15-L88】【F:continuonbrain/eval/multi_model_compare_eval.py†L1-L92】

## Success criteria

- Data: ≥16 clean episodes, timestamp skew ≤5 ms, schema validation passes.
- Pi training: completes without CRITICAL resource alerts; produces candidate adapters.
- Cloud training: meets target metrics; bundle signed; size within Pi limits.
- Deployment: OTA apply succeeds; control loop ≤100 ms mid-loop; rollback path intact.

## Training manager helper (TFRecord-ready)

- Dry-run + episode inventory + TFRecord check:

  ```bash
  python -m continuonbrain.services.training_manager \
    --episodes-dir /opt/continuonos/brain/rlds/episodes \
    --tfrecord-dir /opt/continuonos/brain/rlds/tfrecord
  ```

- Full edge-side flow (health + convert TFRecord + local train + export):

  ```bash
  python -m continuonbrain.services.training_manager \
    --episodes-dir /opt/continuonos/brain/rlds/episodes \
    --tfrecord-dir /opt/continuonos/brain/rlds/tfrecord \
    --health \
    --convert-tfrecord \
    --train-local \
    --trainer-data-path /opt/continuonos/brain/rlds/tfrecord \
    --export
  ```

  Notes:
  - `--convert-tfrecord` runs the JAX TFRecord converter before training.
  - `--trainer-data-path` points the trainer at the TFRecord directory (JAX/TF path).
  - Flags are opt-in; without them the manager reports status and suggested commands.
