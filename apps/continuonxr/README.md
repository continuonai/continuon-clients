# Android Robot Trainer

Android app serving as "robot eyes" and trainer interface with on-device AI via NexaSDK on Qualcomm Hexagon NPU.

**Target:** [Nexa × Qualcomm On-Device Bounty](https://on-device-bounty-mobile.devpost.com/) (Deadline: Feb 15, 2026)

## Status (2026-01)

- **NexaSDK integration** — VLM and ASR pipelines for vision and voice
- **Trainer UI** — Camera preview, voice commands, joystick, arm sliders
- **RLDS recording** — Extended with video frames, transcripts, detections
- **BLE parser + RLDS writer** — Shipped and working

## Modules

### Core
- `MainActivity.kt` — Entry point, routes to mode-specific shells
- `config/` — Configuration loading and flags

### NexaSDK Integration (NEW)
- `nexa/NexaManager.kt` — SDK lifecycle, model loading (VLM + ASR)
- `nexa/VisionPipeline.kt` — Camera → VLM → scene descriptions + detections
- `nexa/VoicePipeline.kt` — Audio → ASR → transcripts

### Trainer UI (NEW)
- `trainer/TrainerScreen.kt` — Main UI with camera, controls, voice
- `trainer/TrainerController.kt` — Coordinates all trainer components
- `trainer/DriveControls.kt` — Virtual joystick for base movement
- `trainer/ArmControls.kt` — 6-axis sliders + gripper
- `trainer/VoicePanel.kt` — Mic button and transcript display
- `trainer/TrainerRldsExtensions.kt` — RLDS with video/voice data

### Camera (NEW)
- `camera/CameraPreview.kt` — CameraX composable
- `camera/DetectionOverlay.kt` — Bounding box rendering

### Voice (NEW)
- `voice/CommandParser.kt` — Transcript → robot commands

### Existing Infrastructure
- `connectivity/` — ContinuonBrain gRPC/WebRTC bridge
- `glove/` — BLE Continuon Glove v0
- `audio/` — Audio capture for RLDS
- `teleop/` — Input fusion and command mapping
- `logging/` — RLDS schema enforcement and validation
- `ui/` — UI context tracker

## Voice Commands

| Command | Action |
|---------|--------|
| `forward` / `back` | Drive robot |
| `left` / `right` | Turn or strafe |
| `stop` | Emergency stop |
| `arm up/down` | Move arm vertically |
| `open` / `close gripper` | Gripper control |
| `teach [name]` | Start teaching mode |
| `done` | Save taught behavior |

## Build/Test

```bash
# Build debug APK
./gradlew :apps:continuonxr:assembleDebug

# Run unit tests
./gradlew :apps:continuonxr:testDebugUnitTest

# Generate proto stubs
./gradlew :apps:continuonxr:generateDebugProto
```

**Requirements:**
- Android Studio Koala+ (AGP 8.5)
- JDK 17
- Snapdragon 8 Gen 3+ device for NPU acceleration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Android Device                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    NexaSDK Layer                       │  │
│  │   VLM (OmniNeural-4B)  │  ASR (Whisper-small)         │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  VisionPipeline  │  VoicePipeline  │  TeleopController│  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               TrainerScreen (Compose)                  │  │
│  │  CameraPreview │ VoicePanel │ DriveControls │ Arm     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │ gRPC
                              ▼
                    Robot (Pi 5 + ContinuonBrain)
```

## Bounty Checklist

| Requirement | Status |
|-------------|--------|
| Runs on Qualcomm Hexagon NPU | ✅ NexaSDK with NPU plugin |
| Uses NexaSDK | ✅ VLM + ASR integration |
| Privacy-focused, local-first | ✅ All AI on-device |
| Working Android demo | ⬜ Build APK |
| Clear documentation | ✅ README + PRD |
| Demo video | ⬜ Record session |

## Next Steps

1. **Test on device** — Install on Snapdragon 8 Gen 3+ device, verify NPU acceleration
2. **Connect to robot** — Test gRPC connection to Pi 5 with ContinuonBrain
3. **Record demo** — 2-3 minute video showing voice commands + robot control
4. **Performance metrics** — Measure inference latency, FPS, power consumption
5. **Submit to bounty** — Package APK with documentation

## Related Files

- `PRD_ANDROID_ROBOT_TRAINER.md` — Full product requirements
- `CLAUDE.md` — Claude Code guidance for this app
- `../../docs/rlds-schema.md` — RLDS episode format
- `../../proto/continuonxr/` — Proto definitions