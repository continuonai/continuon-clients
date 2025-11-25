# Continuon Lifecycle Initiator Plan (Donkey Car Pi 5 + Jetson Nano)

This expands the initial plan to make it actionable. Goal: get a Pi 5 Donkey Car (with optional Jetson Nano) running ContinuonBrain/OS fully offline, logging RLDS locally (opt-in uploads), and enabling XR teleop for gold data, then closing the loop with cloud retraining when you choose.

## Hard constraints
- Robot runs ContinuonOS fully offline; no dependency on cloud schedulers.
- Fast/Mid/Slow learning loops all live on-robot with strict budgets and safety/rollback.
- RLDS logging is local; uploads/sharing are manual/opt-in only.
- HOPE architecture stays on-device: Fast/Medium/Slow loops execute locally on Pi 5 without internet.

## Hardware layout
- **Pi 5**: primary host running ContinuonBrain/OS (continuonos), vehicle control, RLDS logger.
- **Jetson Nano (optional)**: accelerator or camera encoder; TFLite GPU/CUDA delegate if powered; otherwise idle.
- **Sensors/actuators**: Donkey cam (CSI/USB), IMU (if present), PWM steering/throttle (Pi GPIO), optional depth cam.
- **Network**: Wi‑Fi/hotspot for optional uploads; local gRPC/WebRTC for XR/teleop.

## Software stack
- **ContinuonBrain/OS (continuonos repo)**:
  - `src/core/`: control loop (50–100 ms), scheduler, safety head.
  - `platform/linux_sbc`: Pi 5 HAL (PWM, camera, IMU), optional Jetson delegate.
  - `backends`: TFLite XNNPACK; optional CUDA delegate if Nano used.
  - `rlds_logger/`: robot-side RLDS writer (cam/IMU/state/actions).
  - `configs/pi5-donkey.json`: tick rate, model paths, IO mappings, backend.
- **ContinuonXR (this repo)**:
  - XR teleop sends normalized steering/throttle; receives state over gRPC/WebRTC (mock off in prod).
  - RLDS episodes tagged `continuon.xr_mode=trainer` for gold data.
- **Continuon-Cloud** (optional):
  - Minimal HTTP ingest for RLDS zips; later full pipeline for training/packaging Edge Bundles.

## Data flow (offline-first)
1. Pi loop: capture camera/IMU, run control policy (PID/joystick), log `RobotState` + actions via `rlds_logger` locally.
2. XR teleop (optional): ContinuonXR connects to ContinuonBrain on Pi via gRPC/WebRTC; sends commands; logs RLDS on XR side.
3. Episodes stored locally per run (Pi and XR) in per-episode dirs.
4. Upload is manual/opt-in: batch/cron or XR uploader posts zipped episodes to Cloud ingest (`environment_id=pi5-donkey`, model version, hw profile). Default is local only.

## Minimal steps to initiate
1. **Brain profile**: add `configs/pi5-donkey.json` in continuonos (PWM pins, camera source, backend=xnnpack, tick ~50–100 ms, safety limits).
2. **HAL adapter**: implement `platform/linux_sbc` for Pi 5 (PWM steering/throttle, camera reader, optional IMU, heartbeat/estop).
3. **Proto alignment**: ensure `proto/continuonbrain_link.proto` carries steering/throttle/state; regenerate stubs in ContinuonBrain and XR.
4. **XR hookup**: point `ContinuonBrainClient` to Pi IP, disable mock, enable TLS/auth if available; enable SceneCore deps when available to feed pose/gaze/audio.
5. **RLDS logging**: Pi-side `rlds_logger` records cam + control + diagnostics locally (opt-in upload). XR keeps per-episode dirs.
6. **Upload path (optional)**: stand up minimal HTTP ingest in Continuon-Cloud; configure uploader endpoint/token on Pi/XR. Keep uploads manual/opt-in.
7. **Safety/control**: start with joystick/PID; add estop/heartbeat; log safety flags in `step_metadata`.

## Suggested sequencing
- Week 1: configs + HAL stubs + proto regen; XR pointed to Pi mock.
- Week 2: vehicle loop logging RLDS locally; simple ingest endpoint; first opt-in uploads.
- Week 3: XR teleop to Pi over gRPC/WebRTC; start collecting gold data.
- Week 4: Cloud retrains and ships first Edge Bundle back to Pi; close the loop (optional).

## Success markers
- Pi generates RLDS episodes with camera/IMU/commands; local logs intact; uploads succeed only when explicitly triggered.
- XR teleop commands drive the car; RLDS episodes recorded on XR with `trainer` tags.
- Cloud retrains and ships a bundle consumed by Pi; validation run shows improved driving (optional path).

## Offline Nested Learning (Fast/Mid/Slow on-robot)
- **Fast loop (control-rate tweaks)**: tiny stateful updates (biases, norms, small memory vectors) at 20–50 ms; no cloud.
- **Mid loop (bounded local training)**: when idle + enough episodes, run limited gradient steps on adapters/heads; validate and commit if safe.
- **Slow loop (docked/idle consolidation)**: short, budgeted jobs to merge memory and lightly update world/skills; strict time/thermals; fully local.
- **Safety/rollback**: checkpoint rotation (factory/last-good/current), shadow testing before commit, guardrails on safety params.

## Folder/process sketch (Pi)
- `/opt/continuon/brain/runtime/` — binaries, configs (`pi5-donkey.json`), HAL.
- `/opt/continuon/brain/models/` — tflite/onnx heads, adapters, world model.
- `/opt/continuon/brain/checkpoints/` — `brain_v0` (factory), `brain_v1` (last-good), `brain_v2` (current).
- `/opt/continuon/brain/logs/rlds/` — per-episode dirs (metadata + steps + blobs).
- `/opt/continuon/brain/train/` - local trainer jobs, temp adapters, validation results.
- `/opt/continuon/brain/upload/` - staging zips for manual/opt-in cloud upload.

## Optional next additions
- Pseudocode for the local trainer (LoRA/heads with fixed budgets) and a sample `pi5-donkey.json` config.
- Minimal HTTP ingest service (Continuon-Cloud) to accept RLDS zips and queue them for training. 

## Local LoRA trainer (Pi 5 Donkeycar, offline-first)
Goal: bounded, on-robot adapter updates with zero network use. Base model is frozen; only LoRA adapters train. RLDS never auto-uploads.

**Folders (all local)**
```
/opt/continuonos/brain/
  model/
    base_model/                  # frozen
    adapters/
      current/                   # in-use adapters
      candidate/                 # latest trained, pending promotion
      history/                   # rotated archives
  rlds/episodes/                 # local RLDS only (no auto-upload)
  trainer/
    config.yaml
    logs/
```

**Job config (example)**
```python
@dataclass
class LocalTrainerJobConfig:
    rlds_dir: str = "/opt/continuonos/brain/rlds/episodes"
    min_episodes: int = 16
    max_episodes: int = 256
    max_steps: int = 500          # gradient steps
    max_wall_time_s: int = 300    # hard stop
    batch_size: int = 32
    lora_layers: List[str] = None # layer names to adapt
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adapters_out_dir: str = "/opt/continuonos/brain/model/adapters/candidate"
    log_dir: str = "/opt/continuonos/brain/trainer/logs"
```

**RLDS loader (local only)**
- Enumerate `*.tfrecord` (or JSON) from `rlds_dir`, require `min_episodes`, cap at `max_episodes`.
- Streaming reader yields `(obs, action)`; small RAM shuffle buffer creates batches.
- No network calls; optional filters for unsafe steps before batching.

**Model setup**
- Load base model from `/opt/continuonos/brain/model/base_model`, freeze all params.
- Attach LoRA to `lora_layers` (`rank=8` or similar); only LoRA params `requires_grad=True`.
- Optimizer: `Adam(lora_params, lr, weight_decay)`, gradient clipping (`max_norm=1.0`).

**Training loop with budgets**
```python
start = time.time()
step = 0
for batch in make_batch_iterator(files, cfg.batch_size):
    if time.time() - start > cfg.max_wall_time_s: break
    if step >= cfg.max_steps: break
    loss = behavior_cloning_loss(model(batch.obs), batch.action)
    loss.backward()
    clip_grad_norm_(lora_params, 1.0)
    optimizer.step()
    step += 1
```
- Log avg loss every N steps; stop on budgets only.
- Save adapters to `adapters_out_dir/lora_adapters.pt` (candidate slot), write trainer log locally.

**Safety gate + promotion**
- Shadow-evaluate candidate adapters on a handful of recent local episodes:
  - Count safety violations (`violates_safety`), measure action deltas vs logged actions.
  - Reject if any violation or delta exceeds threshold (e.g., `avg_action_delta > 0.25`).
- On accept: move `adapters/current` to `adapters/history/timestamp`, move candidate to `adapters/current`. On reject: delete candidate.

**Scheduler hook (ContinuonOS)**
- Run trainer when robot idle, battery >= 40%, thermals OK, no active teleop.
- Example call: `lora_layers=["policy.transformer.blocks.10", "policy.transformer.blocks.11"]`.
- After successful run, invoke promotion gate. Runtime brain always loads from `adapters/current`.

**Implementation TODOs (progress)**
- [x] Add local trainer scaffold with config parsing and CLI: `continuonbrain/trainer/local_lora_trainer.py` + `continuonbrain/trainer/__init__.py`.
- [x] Sample Pi 5 job config with budgets/paths: `continuonbrain/configs/pi5-donkey.json`.
- [x] Scheduler hook + safety gate + log rotation: `maybe_run_local_training`, `promote_candidate_adapters_if_safe`, `write_trainer_log`.
- [x] Torch hook builder stub for real adapters: `continuonbrain/trainer/hooks_torch.py`.
- [x] RLDS loader dispatcher for JSON/JSONL + TFRecord helper: `continuonbrain/trainer/rlds_loader.py`.
- [x] Basic safety heuristics for shadow eval: `continuonbrain/trainer/safety.py`.
- [x] Integration scaffold for Pi 5 wiring: `continuonbrain/trainer/examples/pi5_integration.py`.
- [x] Config now carries `base_model_path` (e.g., `/opt/continuonos/brain/model/base_policy.pt`) for Pi 5.
- [x] Gemma Pi 5 plan with `flutter_gemma` runtime + sidecar trainer: `continuonbrain/trainer/GEMMA_PI5.md`.
- [x] Sample runtime manifest for Pi: `continuonbrain/model/manifest.pi5.example.json` (base + adapter paths for `flutter_gemma`).
- [x] Gemma LoRA hooks and gating helpers: `continuonbrain/trainer/gemma_hooks.py`, `continuonbrain/trainer/gating_continuonos.py`; config lora layers set to Gemma proj targets.
- [x] Safety placeholder + manifest example with safety head: `continuonbrain/trainer/safety_head_stub.py`, `continuonbrain/model/manifest.pi5.safety.example.json`.
- [x] Integration checklist for continuonos: `continuonbrain/trainer/INTEGRATION_CHECKLIST.md`.
- [ ] Wire real model hooks (Torch/TFLite) in continuonos runtime and replace scaffold.
- [ ] Swap stub safety heuristics with device-specific checks; gate scheduler with real battery/thermal/teleop signals.
- [ ] Collect and stage RLDS episodes locally (>= `min_episodes`) before running trainer; ensure base checkpoint present.
- [x] Document cloud export path (zip RLDS episodes, upload, train on TPU/Colab, repackage adapters) in `continuonbrain/trainer/CLOUD_EXPORT.md`.
