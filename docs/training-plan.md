# End-to-End Training Plan (Pi5 → Cloud → Pi OTA)

Scope: Pi 5 (8 GB) running Continuon Brain runtime + HOPE seed, logging RLDS, promoting adapters to Continuon Cloud for TPU training, and returning a signed Hope Model v1 edge bundle that replaces the seed.

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
   - Run JAX/TPU trainer: `python -m continuonbrain.run_trainer --trainer jax --mode tpu --data-path gs://... --output-dir gs://... --config-preset tpu --num-steps 10000`
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

## Ownership of steps

- Edge (Pi5): Health, capture, on-device adapter training, export prep, OTA apply.
- Cloud: RLDS ingest/clean, TPU training, bundle/sign, OTA serve.

## Success criteria

- Data: ≥16 clean episodes, timestamp skew ≤5 ms, schema validation passes.
- Pi training: completes without CRITICAL resource alerts; produces candidate adapters.
- Cloud training: meets target metrics; bundle signed; size within Pi limits.
- Deployment: OTA apply succeeds; control loop ≤100 ms mid-loop; rollback path intact. 

