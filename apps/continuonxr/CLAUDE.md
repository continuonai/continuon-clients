# CLAUDE.md — Android Robot Trainer App

This file provides guidance to Claude Code when working with the Android XR / Robot Trainer app.

## Quick Start

```bash
# Build the app
./gradlew :apps:continuonxr:assembleDebug

# Run unit tests
./gradlew :apps:continuonxr:testDebugUnitTest

# Generate proto stubs
./gradlew :apps:continuonxr:generateDebugProto
```

## Project Purpose

This Android app serves as:
1. **Robot Eyes** — Camera feed with on-device AI vision (object detection, scene understanding)
2. **Robot Trainer** — Teach behaviors via voice commands and manual controls
3. **RLDS Recorder** — Capture training episodes for ContinuonBrain

Target: **Nexa × Qualcomm On-Device Bounty Program** (Deadline: Feb 15, 2026)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Android Robot Trainer                   │
├─────────────────────────────────────────────────────────┤
│  NexaSDK (Qualcomm Hexagon NPU)                         │
│  ├── VLM: Scene understanding, object detection         │
│  ├── ASR: Voice commands → Brain B actions              │
│  └── LLM: Natural language teaching interface           │
├─────────────────────────────────────────────────────────┤
│  Existing Infrastructure                                │
│  ├── ContinuonBrainClient (gRPC/WebRTC → Pi 5)         │
│  ├── RldsRecorder (training data collection)           │
│  ├── TeleopController (drive/arm commands)             │
│  └── GloveBleClient (BLE glove input)                  │
├─────────────────────────────────────────────────────────┤
│  UI (Jetpack Compose)                                   │
│  ├── Camera preview (robot eyes view)                   │
│  ├── Drive controls (virtual joystick)                  │
│  ├── Arm controls (6-axis sliders)                     │
│  ├── Voice indicator + transcript                       │
│  └── Teaching mode UI                                   │
└─────────────────────────────────────────────────────────┘
```

## Module Structure

```
apps/continuonxr/src/main/java/com/continuonxr/app/
├── MainActivity.kt          # Entry point, mode routing
├── config/                   # Configuration (mode, connectivity, logging)
├── connectivity/             # ContinuonBrain gRPC/WebRTC bridge
├── glove/                    # BLE Continuon Glove v0 integration
├── audio/                    # Audio capture for RLDS
├── teleop/                   # Input fusion, command mapping
├── logging/                  # RLDS schema enforcement, validation
├── ui/                       # UI context tracking
├── input/                    # Sensor fusion engine
├── navigation/               # Mode routing (Trainer/Workstation/Observer)
├── xr/                       # SceneCore/Compose XR rendering
│
│  # NEW for Bounty (to be implemented)
├── nexa/                     # NexaSDK integration
│   ├── NexaManager.kt        # SDK init, model lifecycle
│   ├── VisionPipeline.kt     # Camera → VLM → detections
│   ├── VoicePipeline.kt      # Mic → ASR → commands
│   └── models/               # Model configs and wrappers
├── trainer/                  # Robot trainer UI
│   ├── TrainerScreen.kt      # Main trainer composable
│   ├── DriveControls.kt      # Virtual joystick
│   ├── ArmControls.kt        # 6-axis arm sliders
│   ├── VoicePanel.kt         # Voice command UI
│   └── TeachingMode.kt       # Behavior teaching flow
└── camera/                   # Camera preview composables
    ├── CameraPreview.kt      # Live camera feed
    └── DetectionOverlay.kt   # AI detection visualization
```

## Key Dependencies

### Current
- **Jetpack Compose** — UI framework
- **Jetpack XR/SceneCore** — XR rendering (optional, for XR mode)
- **gRPC + Protobuf** — ContinuonBrain communication
- **WebRTC** — Low-latency video/state streaming
- **Kotlin Coroutines** — Async/Flow patterns

### To Add (for Bounty)
- **NexaSDK** — On-device AI inference on Hexagon NPU
  ```kotlin
  implementation("ai.nexa:core:0.0.19")
  ```
- **CameraX** — Modern camera API
  ```kotlin
  implementation("androidx.camera:camera-camera2:1.3.4")
  implementation("androidx.camera:camera-lifecycle:1.3.4")
  implementation("androidx.camera:camera-view:1.3.4")
  ```

## Development Guidelines

### Code Style
- Use Kotlin coroutines/Flows for async work
- Avoid blocking XR/SceneCore threads
- Prefer extension functions over utility singletons
- Small, previewable composables with state hoisting
- No business logic in UI nodes

### RLDS Logging
- Preserve schema tags: `xr_mode`, `action.source`, provenance metadata
- Align with `docs/rlds-schema.md`
- Use `RldsRecorder` for all episode recording
- Maintain ≤5 ms timestamp alignment

### Proto Contracts
- Keep aligned with `proto/continuonxr/` schemas
- Run `./gradlew :apps:continuonxr:generateDebugProto` after proto changes
- Note temporary stubs in code comments

### NexaSDK Integration
- Initialize in Application class or MainActivity
- Use coroutines for model loading (can take seconds)
- Prefer streaming APIs for real-time inference
- Release models when activity pauses to save memory

## Testing

```bash
# Unit tests
./gradlew :apps:continuonxr:testDebugUnitTest

# Connected tests (requires device/emulator)
./gradlew :apps:continuonxr:connectedDebugAndroidTest

# Build APK for testing
./gradlew :apps:continuonxr:assembleDebug
# Output: apps/continuonxr/build/outputs/apk/debug/
```

## Bounty Requirements Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Runs on Qualcomm Hexagon NPU | ⬜ | Requires Snapdragon 8 Gen 4 device |
| Uses NexaSDK | ⬜ | Add `ai.nexa:core` dependency |
| Privacy-focused, local-first | ⬜ | All AI inference on-device |
| Working Android demo | ⬜ | APK or Play Store |
| Clear README | ⬜ | This file + PRD |
| Demo video | ⬜ | Record robot control session |
| Performance metrics | ⬜ | FPS, latency, power consumption |

## Related Documents

- `PRD_ANDROID_ROBOT_TRAINER.md` — Full product requirements
- `README.md` — Build instructions and module overview
- `AGENTS.md` — Agent-specific instructions
- `../../docs/rlds-schema.md` — RLDS episode format
- `../../proto/continuonxr/` — Proto definitions

## Modes

The app supports three operational modes:

| Mode | Purpose | Primary Features |
|------|---------|------------------|
| **Trainer** | Teach robot behaviors | Voice commands, drive/arm controls, RLDS recording |
| **Workstation** | Monitor and debug | Telemetry display, state inspection |
| **Observer** | Passive viewing | Camera feed, no control |

## ContinuonBrain Connection

The app connects to a robot running ContinuonBrain (typically on Pi 5):

```kotlin
// gRPC connection
val client = ContinuonBrainClient(
    host = "robot.local",  // mDNS discovery
    port = 8080,
    transport = Transport.GRPC  // or WEBRTC for low-latency
)

// Subscribe to robot state
client.streamRobotState().collect { state ->
    // Update UI with joint positions, camera frame, etc.
}

// Send commands
client.sendCommand(ControlCommand(
    mode = CommandMode.EE_VELOCITY,
    vector = Vector3(x = 0.1f, y = 0.0f, z = 0.0f)
))
```

## NexaSDK Usage Pattern

```kotlin
// Initialize SDK (in Application.onCreate or MainActivity)
NexaSdk.getInstance().init(applicationContext)

// Load VLM for vision
val vlm = VlmWrapper.builder()
    .vlmCreateInput(VlmCreateInput(
        model_name = "omni-neural",
        model_path = modelPath,
        plugin_id = "npu",  // Use Hexagon NPU
        config = ModelConfig()
    ))
    .build()
    .getOrThrow()

// Run inference on camera frame
val description = vlm.generate(
    prompt = "Describe what you see. List any objects the robot could interact with.",
    image = cameraFrame
)

// For real-time, use streaming
vlm.generateStreamFlow(prompt, config).collect { token ->
    updateUI(token)
}
```

## Voice Command Flow

```
User speaks → Mic capture → NexaSDK ASR → Text
    ↓
Parse command ("move forward", "teach patrol", "pick up cup")
    ↓
Route to handler:
  - Drive command → TeleopController → ContinuonBrain
  - Teach command → TeachingMode state machine
  - Object command → VLM localization → arm planning
    ↓
RLDS recording captures action + observation
```

## Contact

For bounty-specific questions, see:
- Devpost: https://on-device-bounty-mobile.devpost.com/
- NexaSDK Docs: https://docs.nexa.ai/
- GitHub: https://github.com/NexaAI/nexa-sdk
