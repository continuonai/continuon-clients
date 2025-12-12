# End-to-End Training Plan (Pi5 → Cloud → Pi OTA)

Scope: Pi 5 (8 GB) running Continuon Brain runtime + HOPE seed, logging RLDS, promoting adapters to Continuon Cloud for TPU training, and returning a signed Hope Model v1 edge bundle that replaces the seed. See `docs/seed-model-plan.md` for the detailed Fast/Mid/Slow seed playbook (Pi capture + GCP TPU export) that this plan references.

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
   - Storage: `/opt/continuonos/brain/rlds/episodes/…`

4) **On-device adapter training (Pi)**
   - Config: `continuonbrain/configs/pi5-donkey.json` (batch 8, max_steps 300, shuffle 2).
   - Run: `python -m continuonbrain.trainer.local_lora_trainer --config continuonbrain/configs/pi5-donkey.json --use-stub-hooks` (replace hooks when real model ready).
   - Output: `/opt/continuonos/brain/model/adapters/candidate/lora_adapters.pt`
   - Gate: loss converges, resource monitor not CRITICAL/EMERGENCY, checkpoints saved.

5) **RLDS prep for cloud**
   - Anonymize/validate: `python -m continuonbrain.rlds.export_pipeline --help` (or call `prepare_cloud_export` in code).
   - Env tag: `CONTINUON_EXPORT_ORIGIN=pi5` (default) to stamp manifest.
   - Gate: validation reports OK; manifest written under export dir.

6) **Cloud training (Continuon Cloud / TPU)**
   - Upload RLDS + candidate adapters to cloud bucket (per cloud ingest docs).
   - Run JAX/TPU trainer: `python -m continuonbrain.run_trainer --trainer jax --mode tpu --data-path gs://... --output-dir gs://... --config-preset tpu --num-steps 10000` (matches the seed playbook’s TPU step).
   - Output: Hope Model v1 artifacts (TFLite/ONNX/Hailo-HEF placeholders) + safety manifest.
   - Gate: eval metrics pass (set project-specific thresholds), bundle passes integrity/signature.

7) **Edge bundle assembly**
   - Follow `docs/bundle_manifest.md`; include:
     - Base/adapter weights (TFLite or ONNX), optional `.hef` if compiled.
     - `edge_manifest.json` with version, compat, signatures, preferred backends, safety manifest.
   - Sign bundle; store last-known-good for rollback.

8) **OTA back to Pi**
   - Deliver via companion app OTA flow (ownership + subscription gated).
   - On Pi: download → verify signature/checksum → stage A/B → hot-swap.
   - Gate: startup manager passes health with new model; control loop latency within budget; fallback preserved.

9) **Post-deploy validation**
   - Run: `PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --real-hardware`
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

## Training manager helper

- Dry-run the plan and optionally execute steps with the orchestrator:

  ```bash
  python -m continuonbrain.services.training_manager \
    --health \
    --train-local \
    --export \
    --post-validate
  ```

  Flags are opt-in; without them the manager only reports status and suggested commands.
