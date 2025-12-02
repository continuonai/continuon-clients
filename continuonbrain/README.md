# ContinuonBrain (scaffolding only)

ContinuonBrain/OS runtime lives in the separate `continuonos` repo (platform-agnostic core, HAL adapters, backends, configs). This folder only carries **scaffolding and contracts** used by ContinuonXR:
- `proto/continuonbrain_link.proto` (shared contract; mirror downstream).
- `trainer/` offline Pi/Jetson adapter-training scaffold (bounded, RLDS-only, safety-gated) to align with ContinuonBrain/OS goals. Synthetic RLDS samples for dry-runs sit under `continuonbrain/rlds/episodes/`. Sample manifest in `continuonbrain/model/manifest.pi5.example.json` shows how Pi 5 + `flutter_gemma` can load base + LoRA without extra quantization.
- Raspberry Pi 5 bring-up checklist (depth cam + PCA9685) lives in `continuonbrain/PI5_CAR_READINESS.md`.

No production runtime should live here; wire real implementations inside `continuonos` while keeping the interface consistent.

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
- `system_health.py` - Comprehensive hardware/software health checks
- `startup_manager.py` - Startup orchestration with automatic wake checks

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

### Hardware Compatibility

✅ **Confirmed Compatible:**
- PCA9685 (I2C 0x40) + AI HAT+ (PCIe + GPIO passthrough)
- OAK-D Lite (USB3) + AI HAT+ (independent buses)
- SO-ARM101 servos via PCA9685 (external 5-7V power required)

See `PI5_CAR_READINESS.md` for full hardware setup and validation steps.

