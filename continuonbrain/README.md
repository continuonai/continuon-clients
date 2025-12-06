# ContinuonBrain

The Continuon Brain runtime and scaffolding now live together in this monorepo. Use this folder to ship production runtime assets alongside the existing **scaffolding and contracts** used by ContinuonXR:
- `proto/continuonbrain_link.proto` (shared contract; mirror downstream).
- `trainer/` offline Pi/Jetson adapter-training scaffold (bounded, RLDS-only, safety-gated) to align with ContinuonBrain/OS goals. Synthetic RLDS samples for dry-runs sit under `continuonbrain/rlds/episodes/`. Sample manifest in `continuonbrain/model/manifest.pi5.example.json` shows how Pi 5 + `flutter_gemma` can load base + LoRA without extra quantization.
- Raspberry Pi 5 bring-up checklist (depth cam + PCA9685) lives in `continuonbrain/PI5_CAR_READINESS.md`.
- Pi 5 edge brain v0 execution steps (health checks, RLDS recording, trainer runbook) live in `continuonbrain/PI5_EDGE_BRAIN_INSTRUCTIONS.md`.

Production runtime code belongs here; keep docs explicit about what is production-ready versus placeholder scaffolding so downstream consumers can promote features confidently.

For training-time autonomy (when humans are away), keep the robot focused on safe, creator-aligned work items listed in `SELF_IMPROVEMENT_BACKLOG.md`. Tasks emphasize offline-first checks, system health validation, and strict adherence to the safety protocol.

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
