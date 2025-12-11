# ContinuonBrain

The Continuon Brain runtime and scaffolding now live together in this monorepo. Use this folder to ship production runtime assets alongside the existing **scaffolding and contracts** used by ContinuonXR:
- `proto/continuonbrain_link.proto` (shared contract; mirror downstream).
- `trainer/` offline Pi/Jetson adapter-training scaffold (bounded, RLDS-only, safety-gated) to align with ContinuonBrain/OS goals. Synthetic RLDS samples for dry-runs sit under `continuonbrain/rlds/episodes/`. Sample manifest in `continuonbrain/model/manifest.pi5.example.json` shows how Pi 5 + `flutter_gemma` can load base + LoRA without extra quantization.
- Raspberry Pi 5 bring-up checklist (depth cam + PCA9685) lives in `continuonbrain/PI5_CAR_READINESS.md`.
- Pi 5 edge brain v0 execution steps (health checks, RLDS recording, trainer runbook) live in `continuonbrain/PI5_EDGE_BRAIN_INSTRUCTIONS.md`.
- JAX training pipeline lives under `continuonbrain/jax_models/` with a unified trainer (`run_trainer.py`) that selects JAX vs PyTorch based on hardware. Use `--config-preset` or `--config-json` to tune CoreModel settings.
- Model selection is JAX-first (`CONTINUON_PREFER_JAX=1` by default). Transformers/Gemma remains available as fallback. Gemma 3n JAX/Flax weights in the HF cache are detected when present.

Production runtime code belongs here; keep docs explicit about what is production-ready versus placeholder scaffolding so downstream consumers can promote features confidently.

For training-time autonomy (when humans are away), keep the robot focused on safe, creator-aligned work items listed in `SELF_IMPROVEMENT_BACKLOG.md`. Tasks emphasize offline-first checks, system health validation, and strict adherence to the safety protocol.

### Pi 5 HAT vision seed (reference names)
- Base model placeholder for HAT vision runs: `/opt/continuonos/brain/model/base_model/hat_vision_seed.pt`
- Current adapter target: `/opt/continuonos/brain/model/adapters/current/lora_hat_vision.pt`
- Candidate adapter target: `/opt/continuonos/brain/model/adapters/candidate/lora_hat_vision.pt`
- RLDS episodes directory: `/opt/continuonos/brain/rlds/episodes/` (camera-only acceptable when PCA is down; OAK-D Lite provides RGB+depth frames)
- If on-device LLM/tools (e.g., Gemma 3n 2B with MCP/http) are used, log tool traces into `step_metadata` for later cloud/JAX ingestion.
- Hailo: prefer Hailo inference when available; placeholder HEF path `/opt/continuonos/brain/model/base_model/model.hef` with CPU fallback if SDK/hef are absent.
- To route inference via Hailo with fallback, use `InferenceRunner`:
  ```python
  from pathlib import Path
  from continuonbrain.inference_runner import InferenceRunner
  runner = InferenceRunner(
      use_hailo=True,
      hef_path=Path("/opt/continuonos/brain/model/base_model/model.hef"),
      cpu_fn=your_cpu_forward_fn,
  )
  out = runner(inputs)
  ```

The Robot API server launched by `startup_manager.py` now uses `python -m continuonbrain.robot_api_server` and runs in **real hardware mode by default** (it fails fast if controllers are absent). This keeps the production path aligned with the previous mock features (UI/JSON bridge) while enforcing hardware readiness.

## Pi5 Robot Arm Integration (Design Validation)

Built SO-ARM101 + OAK-D Lite integration for design validation without physical hardware.

### Components Built

**Sensors** (`sensors/`):
- `oak_depth.py` - OAK-D Lite depth camera capture (RGB + depth @ 30fps)
- `hardware_detector.py` - Auto-detection for cameras, HATs, servos, IMUs

**Actuators** (`actuators/`):
- `pca9685_arm.py` - PCA9685 6-DOF servo control with safety bounds

**Recording** (`recording/`):
- `arm_episode_recorder.py` - RLDS episode recorder (depth + arm state + actions)
- `episode_upload.py` - Episode packaging and upload pipeline

**System Management**:
- `system_health.py` - Comprehensive hardware/software health checks, including MCP/Gemini discovery and a $5/day API budget guard to stay offline-first
- `startup_manager.py` - Startup orchestration with automatic wake checks and boot
  enforcement for system instructions + safety protocol

**Testing** (`tests/`):
- `integration_test.py` - Full stack test (mock + real hardware modes)

### Quick Start

```bash
# Check system health (always run on wake from sleep)
cd /home/craigm26/ContinuonXR
PYTHONPATH=$PWD python3 continuonbrain/system_health.py --quick

# Auto-detect connected hardware
PYTHONPATH=$PWD python3 continuonbrain/sensors/hardware_detector.py

# Test full stack in mock mode (with auto-detection)
PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py

# Only detect hardware (don't run full test)
PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --detect-only

# Test with real hardware (when available, auto-detects devices)
PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --real-hardware

# Startup manager (automatic health check on wake from sleep)
PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py
```

See [Hardware Detection Guide](../docs/hardware-detection.md) for supported devices and [System Health](../docs/system-health.md) for health checking.

### JAX Training & Inference (CoreModel + Gemma 3n JAX)
- Local sanity check: `python -m continuonbrain.run_trainer --trainer jax --mode local --config-preset pi5 --max-steps 5`
- TPU training: `python -m continuonbrain.run_trainer --trainer jax --mode tpu --data-path gs://... --output-dir gs://... --config-preset tpu --num-steps 10000`
- CPU inference smoke: `python -m continuonbrain.jax_models.export.infer_cpu --model-path ./models/core_model_inference --obs-dim 128 --action-dim 32`
- Gemma 3n JAX/Flax: ensure weights are present in HF cache; model detector will list them as `jax-gemma`.

### Hailo (AI HAT+) status
- Export pipeline JAX→TF→ONNX is available; `.hef` creation is a placeholder without the Hailo SDK. Runtime inference will skip Hailo if `.hef` is missing; full acceleration requires integrating Hailo compiler/runtime tools.
- When `.hef` is missing or placeholder, the inference router falls back to CPU and logs a warning.
- To use a provided HEF (e.g., model zoo), place it at `/opt/continuonos/brain/model/base_model/model.hef` (or pass `--hef-source` and `--install-hef-path` to `continuonbrain/jax_models/export/export_hailo.py`). The export step copies the HEF to the runtime path and records input/output vstream metadata when `hailo_platform` is installed.
- Enable depth/Hailo offload flag with `CONTINUON_HAILO_OFFLOAD=1`; OAK-D detection will mark devices as offload-capable and the hardware detector will tag the Hailo HAT accordingly.
- Quick benchmark on-device: `hailortcli benchmark /opt/continuonos/brain/model/base_model/model.hef --time-to-run 5`

### OTA packaging
- Follow the signed bundle contract in `docs/bundle_manifest.md` when preparing edge bundles (CPU/Hailo artifacts + safety manifest).
- OTA apply is gated in the ContinuonAI app by robot ownership + paid subscription; device verifies checksums/signature before swap.

Conversation log: Pi5 startup/training optimization (2025-12-10) summarized at `../docs/conversation-log.md` (headless Pi5 boot defaults, optional background trainer, tuned Pi5 training config, RLDS origin tagging).

## Autostart on boot (systemd template)
- A systemd unit template lives at `continuonbrain/systemd/continuonbrain-startup.service`.
- Edit the `Environment=` paths to match your install (e.g., `PYTHONPATH=/home/pi/ContinuonXR`, `CONFIG_DIR=/opt/continuonos/brain`, `WorkingDirectory=/home/pi/ContinuonXR`).
- Install and enable:
  ```bash
  sudo cp continuonbrain/systemd/continuonbrain-startup.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable continuonbrain-startup.service
  sudo systemctl start continuonbrain-startup.service
  ```
- The unit runs `python -m continuonbrain.startup_manager` on boot, which performs health checks and launches the Robot API in real-hardware mode. Check logs with `sudo journalctl -u continuonbrain-startup.service -f`.

### Boot safety + system instructions
- Add non-negotiable instructions in `system_instructions.json` under your `CONFIG_DIR`.
  Defaults always include loading the safety protocol before enabling motion and rejecting
  unsafe commands; user-provided entries append to those defaults.
- Safety rules live under `CONFIG_DIR/safety/protocol.json` and are merged with the
  immutable base rules (e.g., "Do not harm humans or other organisms" and respect for
  property/laws). Setting `override_defaults` has no effect because the base rules cannot
  be removed.
- The humanity-first **Continuon AI mission statement** is baked into boot: the firmware loads
  `continuonbrain/MISSION_STATEMENT.md` and anchors all autonomy to the mission pillars
  (collective intelligence in service of humans, distributed/on-device by default).

### Hardware Compatibility

✅ **Confirmed Compatible:**
- PCA9685 (I2C 0x40) + AI HAT+ (PCIe + GPIO passthrough)
- OAK-D Lite (USB3) + AI HAT+ (independent buses)
- SO-ARM101 servos via PCA9685 (external 5-7V power required)

- Quick validation on-device: `python -m continuonbrain.pi5_hardware_validation --log-json /tmp/pi5_check.json` (creates the expected `/opt/continuonos/brain` layout, checks UVC/I2C, and exercises depth + servo timestamps).

See `PI5_CAR_READINESS.md` for full hardware setup and validation steps.
