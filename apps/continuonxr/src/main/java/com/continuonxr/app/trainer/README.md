# Trainer Module

Robot trainer UI with NexaSDK integration for the Nexa × Qualcomm On-Device Bounty.

## Components

### Core Classes

| File | Purpose |
|------|---------|
| `TrainerScreen.kt` | Main UI composable with camera, controls, and voice |
| `TrainerController.kt` | Coordinates voice, vision, and robot control |
| `DriveControls.kt` | Virtual joystick for base movement |
| `ArmControls.kt` | 6-axis joint sliders and gripper |
| `VoicePanel.kt` | Mic button and transcript display |
| `TrainerRldsExtensions.kt` | RLDS recording with video/voice data |

### NexaSDK Integration (`../nexa/`)

| File | Purpose |
|------|---------|
| `NexaManager.kt` | SDK lifecycle, model loading |
| `NexaSdkWrapper.kt` | SDK abstraction for testing |
| `VisionPipeline.kt` | Camera frames → VLM → detections |
| `VoicePipeline.kt` | Audio → ASR → transcripts |

### Camera (`../camera/`)

| File | Purpose |
|------|---------|
| `CameraPreview.kt` | CameraX composable |
| `DetectionOverlay.kt` | Bounding box rendering |

### Voice (`../voice/`)

| File | Purpose |
|------|---------|
| `CommandParser.kt` | Transcript → robot commands |

## Voice Commands

| Command | Action |
|---------|--------|
| `forward` / `back` | Drive robot |
| `left` / `right` | Turn or strafe |
| `stop` | Emergency stop |
| `arm up/down/forward/back` | Move arm |
| `open gripper` / `close gripper` | Gripper control |
| `arm home` / `arm ready` | Arm presets |
| `teach [name]` | Start teaching mode |
| `done` | Save taught behavior |
| `cancel` | Cancel teaching |
| `[behavior name]` | Run learned behavior |

## Usage

```kotlin
// In MainActivity or ViewModel
val nexaManager = NexaManager(context)
val brainClient = ContinuonBrainClient(config)
val trainerController = TrainerController(nexaManager, brainClient, viewModelScope)

// Initialize (loads NexaSDK models)
lifecycleScope.launch {
    trainerController.initialize()
}

// In Compose
TrainerScreen(
    nexaManager = trainerController.nexaManager,
    visionPipeline = trainerController.visionPipeline,
    voicePipeline = trainerController.voicePipeline,
    onCommand = { command -> trainerController.sendCommand(command) },
    onStartRecording = { /* start RLDS */ },
    onStopRecording = { /* stop RLDS */ }
)
```

## RLDS Recording

The trainer extends RLDS observations with:

- `videoFrameId` - Reference to saved camera frame
- `voice_transcript` - Latest voice command
- `scene_description` - VLM scene analysis
- `detections_count` / `detections_labels` - Object detection results
- `teaching_state` - Current teaching mode state
- `input_source` - How command was issued (voice/joystick/slider)

## Architecture

```
TrainerScreen (UI)
    │
    ├── CameraPreview → VisionPipeline → NexaManager (VLM)
    │                       │
    │                       └── Detections → DetectionOverlay
    │
    ├── VoicePanel → VoicePipeline → NexaManager (ASR)
    │                    │
    │                    └── Transcript → CommandParser
    │
    ├── DriveControls ─┐
    │                  │
    └── ArmControls ───┴──→ TrainerController → ContinuonBrainClient
                                   │
                                   └── TrainerRldsRecorder
```

## Testing

```bash
# Unit tests
./gradlew :apps:continuonxr:testDebugUnitTest

# Specific test
./gradlew :apps:continuonxr:testDebugUnitTest --tests "*.CommandParserTest"
```

## Requirements

- Android 10+ (API 29)
- Snapdragon 8 Gen 3+ for NPU acceleration
- Camera and microphone permissions
- NexaSDK 0.0.19
