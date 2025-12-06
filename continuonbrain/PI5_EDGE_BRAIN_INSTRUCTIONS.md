# Pi 5 Edge Brain v0 Execution Guide

This guide stitches together the core components needed for a “version zero” Pi 5 edge brain: depth camera capture, PCA9685 drivetrain control, RLDS recording, health/startup orchestration, and integration tests. It assumes you are running the combined runtime + scaffolding that now lives in this repo, with clear markers for any remaining placeholders.

## 1) Prep the Pi 5 environment
- OS: 64-bit Raspberry Pi OS Lite or Ubuntu 24.04 with I2C enabled (`raspi-config` → Interfacing → I2C).
- Packages: `sudo apt install -y python3-venv python3-pip v4l-utils libatlas-base-dev`.
- Create the expected runtime/training layout (also done automatically by the validation script):
  ```bash
  sudo mkdir -p /opt/continuonos/brain/{model/base_model,model/adapters/{current,candidate},rlds/episodes,trainer/logs}
  ```
- Recommended quick check: `python -m continuonbrain.pi5_hardware_validation --log-json /tmp/pi5_check.json` (detects UVC/I2C, timestamps depth + servo pulses, and writes a readiness report).

## 2) Verify hardware paths
- Depth camera: `v4l2-ctl --list-devices` should show a UVC device (e.g., OAK-D Lite). Stick to `640x480@30` or `1280x720@30` profiles until bandwidth is proven stable.
- PCA9685: `i2cdetect -y 1` should list `0x40`. Power servos separately and share ground with the Pi.
- If either device is missing, rerun cabling and the validation script before proceeding.

## 3) Run system health and startup manager
- Health check on wake/boot (fails fast if controllers are absent):
  ```bash
  PYTHONPATH=$PWD python3 continuonbrain/system_health.py --quick
  ```
- Startup orchestration with boot-time instructions and safety protocol:
  ```bash
  PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py
  ```
  This loads `system_instructions.json`, enforces safety rules, and launches the Robot API server in **real hardware** mode.

## 4) Exercise sensors and actuators
- Auto-detect hardware only:
  ```bash
  PYTHONPATH=$PWD python3 continuonbrain/sensors/hardware_detector.py
  ```
- Quick depth capture sanity (headless):
  ```bash
  ffmpeg -f v4l2 -i /dev/video0 -frames:v 10 /tmp/depth_test.mkv
  ```
- Simple PCA9685 pulse test (uses the default steering/throttle mapping):
  ```bash
  PYTHONPATH=$PWD python -m continuonbrain.pi5_hardware_validation --detect-only
  ```
  Servos should respond with bounded pulses; timestamps target ≤5 ms skew between depth frames and actions.

## 5) Record an RLDS episode in mock vs. real modes
- Full integration test (mock hardware, auto-detect):
  ```bash
  PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py
  ```
- Detect-only (no motion):
  ```bash
  PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --detect-only
  ```
- Real hardware path (requires depth cam + PCA9685 connected):
  ```bash
  PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --real-hardware
  ```
  Episodes are written under `continuonbrain/rlds/episodes/` by default; sync depth + robot state within the ≤5 ms budget.

## 6) Train a LoRA adapter with safety guards
- Use the Pi-ready config and stub hooks for IO validation:
  ```bash
  python -m continuonbrain.trainer.local_lora_trainer \
    --config continuonbrain/configs/pi5-donkey.json \
    --use-stub-hooks
  ```
- When ready for Gemma-based runs, wire real loaders per `trainer/GEMMA_PI5.md` and point manifests to `/opt/continuonos/brain/model/adapters/current/`.

## 7) Promote adapters and run with the Flutter runtime
- Manifests: `continuonbrain/model/manifest.pi5.example.json` (and `.safety`) show how to load the base model + LoRA via `flutter_gemma` on Pi.
- Keep adapter promotions in `/opt/continuonos/brain/model/adapters/current/`; the Flutter runtime should hot-reload on manifest change.
- Teleop + preview should travel over the Robot API (`proto/continuonbrain_link.proto`); ensure depth + drivetrain feedback align with RLDS fields.

## 8) Pre-drive checklist
- Depth stream stable at chosen resolution; PCA9685 responds to bounded commands.
- `min_episodes` satisfied for training config and camera/robot timestamps align.
- Safety bounds and gating hooks enabled; violations logged in RLDS `step_metadata` before allowing autonomous motion.
