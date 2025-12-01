# Raspberry Pi 5 Donkey Car Bring-Up (depth cam + PCA9685)

Goal: a repeatable Pi 5 setup that can record RLDS, train LoRA adapters locally, and drive a PCA9685-controlled RC car while previewing a USB3 depth camera. Production runtime still lives in `continuonos`; this doc keeps the `continuonbrain/` scaffold Pi-ready.

## Hardware + OS prep
- Use 64-bit Raspberry Pi OS Lite or Ubuntu 24.04, enable I2C (`raspi-config` → Interfacing → I2C) and reboot.
- Confirm devices: `v4l2-ctl --list-devices` (depth cam) and `i2cdetect -y 1` (PCA9685 should show `0x40`).
- Minimal packages: `sudo apt install -y python3-venv python3-pip v4l-utils libatlas-base-dev` (skip heavier ML deps until needed).
- Directory layout expected by configs:  
  `sudo mkdir -p /opt/continuonos/brain/{model/base_model,model/adapters/{current,candidate},rlds/episodes,trainer/logs}` (matches `configs/pi5-donkey.json` and manifests).

## Depth camera notes (USB3)
- Stick to UVC profiles that the Pi can sustain (e.g., 640x480@30 fps with depth). Check with `v4l2-ctl --list-formats-ext`.
- Keep timestamps aligned to robot state within **≤5 ms**; log them into RLDS `observation` alongside `observation.robot_state`.
- Quick sanity capture (headless): `ffmpeg -f v4l2 -i /dev/video0 -frames:v 10 /tmp/depth_test.mkv` to verify bandwidth.
- When emitting RLDS: store per-step depth frames and include camera intrinsics in `episode_metadata`; follow `docs/rlds-schema.md` to keep schema compatible.

## PCA9685 control (steering/throttle)
- Power servos separately from the Pi; only share ground. Keep PCA9685 on the default `0x40` unless you have multiple boards.
- Suggested Python driver (optional, guard imports on-device): `adafruit-circuitpython-pca9685` + `adafruit-circuitpython-servokit`.
- Minimal mapping sketch:
  ```python
  # Map normalized [-1, 1] steering/throttle into servo pulses
  from adafruit_servokit import ServoKit
  kit = ServoKit(channels=16)
  def send_action(steering, throttle):
      kit.servo[0].angle = 90 + steering * 45  # steering servo
      kit.continuous_servo[1].throttle = throttle  # ESC / motor
  ```
- Wire this to the trainer safety hooks (`build_simple_action_guards`) and to runtime control paths in `continuonos`, keeping bounds logged in RLDS `step_metadata`.

## First local training smoke test
- Populate `rlds_dir` with a few episodes (synthetic samples live under `continuonbrain/rlds/episodes/`).
- Run the scaffold with stub hooks to validate IO and gating on Pi 5:
  ```bash
  python -m continuonbrain.trainer.local_lora_trainer \
    --config continuonbrain/configs/pi5-donkey.json \
    --use-stub-hooks
  ```
- For Gemma-based runs, wire real loaders/injectors per `continuonbrain/trainer/GEMMA_PI5.md` and point manifests to adapters.

## Runtime + manifest handshake
- Use `continuonbrain/model/manifest.pi5.example.json` (or `.safety`) to load base + LoRA adapters via `flutter_gemma` on the Pi.
- Keep adapter promotions in `/opt/continuonos/brain/model/adapters/current/` and have the Flutter runtime hot-reload on manifest change.
- Teleop/preview data to the Flutter companion should flow over the Robot API (`proto/continuonbrain_link.proto`) with RLDS-compatible fields for depth + drivetrain feedback.

## Ops checklist before driving
- Depth stream stable at chosen resolution; PCA9685 responds to bounded commands.
- `min_episodes` met (see `configs/pi5-donkey.json`) and timestamps align camera + robot state.
- Safety bounds enabled; violations logged. Idle/battery/thermal gates hooked via `gating_continuonos.py` before enabling training on-device.
