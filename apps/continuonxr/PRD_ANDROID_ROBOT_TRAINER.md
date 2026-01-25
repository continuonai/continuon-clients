# PRD: Android Robot Trainer App

**Product Requirements Document**
**Version:** 1.0
**Last Updated:** 2026-01-25
**Target:** Nexa × Qualcomm On-Device Bounty Program

---

## Executive Summary

Build an Android app that serves as "robot eyes" and a trainer interface for home robots. The app runs on-device AI (via NexaSDK on Qualcomm Hexagon NPU) for vision understanding, voice commands, and natural language teaching. It connects to ContinuonBrain-powered robots and records RLDS training episodes.

**Bounty Alignment:**
- Deadline: February 15, 2026 (Phase 1), March 24, 2026 (Phase 2)
- Prize: $500 + Snapdragon device (Phase 1), $5,000 grand prize (Phase 2)
- Requirement: Android app with on-device AI via NexaSDK on Hexagon NPU

---

## Problem Statement

### User Pain Points

1. **Training robots is hard** — Users can't easily teach robots new behaviors without coding
2. **Robot vision is cloud-dependent** — Privacy concerns, latency, connectivity requirements
3. **No mobile control** — Most robot interfaces are desktop web apps
4. **Training data collection is fragmented** — No unified recording across sessions

### Market Opportunity

- Home robotics market growing rapidly (cleaning, companion, assistance robots)
- On-device AI eliminates cloud costs and latency
- Mobile-first generation expects phone-based control
- RLDS standard enables cross-platform training data

---

## Target Users

### Primary: Robot Enthusiasts / Early Adopters

- Own or building home robots (Raspberry Pi, Arduino, commercial platforms)
- Technical enough to set up robot, want easier training
- Privacy-conscious, prefer local processing

### Secondary: Robotics Developers

- Building robot products, need training data collection
- Want to test on-device AI capabilities
- Evaluating NexaSDK for their products

### Tertiary: Accessibility Users

- Need voice-controlled robot assistance
- Benefit from natural language teaching interface

---

## Product Goals

### Phase 1 (Bounty Submission — Feb 15, 2026)

1. **Working demo** on Qualcomm Hexagon NPU via NexaSDK
2. **Robot eyes** — Camera with on-device vision understanding
3. **Voice control** — On-device ASR for hands-free operation
4. **Basic training** — Manual drive/arm controls with RLDS recording
5. **Compelling demo video** showing real robot control

### Phase 2 (Grand Prize — Mar 24, 2026)

1. **Play Store release** or public APK
2. **Teaching mode** — Natural language behavior teaching ("teach patrol")
3. **Object interaction** — "Pick up the red cup" with VLM grounding
4. **Polish** — Production UI, error handling, onboarding

---

## Feature Specifications

### F1: Camera Preview (Robot Eyes)

**Description:** Live camera feed with AI-powered scene understanding.

**User Stories:**
- As a user, I want to see what the robot's camera sees on my phone
- As a user, I want the app to describe objects in view
- As a user, I want to tap an object to get information about it

**Technical Requirements:**
- CameraX for camera capture (30 FPS target)
- NexaSDK VLM for scene understanding
- Overlay composable for detection visualization
- Optional: Receive camera feed from robot via WebRTC

**Acceptance Criteria:**
- [ ] Camera preview displays with <100ms latency
- [ ] VLM can describe scene on demand
- [ ] Object detections overlay on camera feed
- [ ] Works in portrait and landscape

**Implementation Plan:**
1. Add CameraX dependencies to build.gradle.kts
2. Create `CameraPreview.kt` composable with CameraX integration
3. Create `VisionPipeline.kt` to connect camera frames to NexaSDK VLM
4. Create `DetectionOverlay.kt` for rendering bounding boxes/labels
5. Add "Describe Scene" button that triggers VLM inference
6. Wire to TrainerScreen

---

### F2: Voice Commands

**Description:** Hands-free robot control via on-device speech recognition.

**User Stories:**
- As a user, I want to control the robot with voice commands
- As a user, I want to see my transcribed speech
- As a user, I want the robot to confirm my command

**Supported Commands:**
| Command Pattern | Action |
|-----------------|--------|
| "forward" / "move forward" | Drive forward |
| "back" / "reverse" | Drive backward |
| "left" / "turn left" | Turn left |
| "right" / "turn right" | Turn right |
| "stop" | Stop all motion |
| "arm up" / "arm down" | Move arm vertically |
| "open gripper" / "close gripper" | Gripper control |
| "teach [name]" | Enter teaching mode |
| "done" / "finished" | Exit teaching mode |
| "run [name]" / "do [name]" | Execute learned behavior |

**Technical Requirements:**
- NexaSDK ASR model for speech-to-text
- Command parser with fuzzy matching
- Audio capture with noise filtering
- TTS for robot responses (optional, can use system TTS)

**Acceptance Criteria:**
- [ ] Voice activation works reliably in normal environment
- [ ] Commands recognized with >90% accuracy
- [ ] Transcription displays in <500ms
- [ ] Visual feedback during listening

**Implementation Plan:**
1. Add RECORD_AUDIO permission to manifest
2. Create `VoicePipeline.kt` with audio capture + NexaSDK ASR
3. Create `CommandParser.kt` for text → action mapping
4. Create `VoicePanel.kt` composable (mic button, transcript, status)
5. Wire commands to TeleopController
6. Add haptic/visual feedback for command recognition

---

### F3: Drive Controls

**Description:** Manual robot driving via touch controls.

**User Stories:**
- As a user, I want to drive the robot with a virtual joystick
- As a user, I want precise speed control
- As a user, I want emergency stop always accessible

**Technical Requirements:**
- Virtual joystick composable (2-axis: forward/back, turn)
- Speed multiplier slider (0.25x to 1.0x)
- Emergency stop button (always visible, red)
- Haptic feedback on input

**Acceptance Criteria:**
- [ ] Joystick responds to touch with no perceptible lag
- [ ] Commands sent to robot at 20Hz minimum
- [ ] Emergency stop halts robot within 100ms
- [ ] Controls work in all orientations

**Implementation Plan:**
1. Create `VirtualJoystick.kt` composable with touch handling
2. Create `DriveControls.kt` wrapper with stop button and speed slider
3. Map joystick position to drive commands via TeleopController
4. Add haptic feedback using VibrationEffect
5. Wire to ContinuonBrainClient for robot communication

---

### F4: Arm Controls

**Description:** 6-axis robotic arm control via sliders.

**User Stories:**
- As a user, I want to control each arm joint independently
- As a user, I want to control the gripper
- As a user, I want to see current joint positions

**Technical Requirements:**
- 6 joint sliders (J1-J6) with angle display
- Gripper slider (0-100% open)
- Current position display (from robot state)
- Preset positions (home, ready, pick)

**Acceptance Criteria:**
- [ ] All 6 joints controllable independently
- [ ] Gripper opens/closes smoothly
- [ ] Current positions update in real-time
- [ ] Presets move arm to defined positions

**Implementation Plan:**
1. Create `JointSlider.kt` composable with label and value display
2. Create `ArmControls.kt` with 6 sliders + gripper
3. Add preset buttons with predefined joint angles
4. Wire to TeleopController and ContinuonBrainClient
5. Subscribe to robot state for current position display

---

### F5: RLDS Recording

**Description:** Record training episodes for ContinuonBrain training.

**User Stories:**
- As a user, I want to record my control sessions
- As a user, I want recordings to include camera, audio, and actions
- As a user, I want to upload recordings for training

**Technical Requirements:**
- Leverage existing `RldsRecorder` infrastructure
- Capture: camera frames, audio, joint states, commands, timestamps
- Schema compliance with `proto/continuonxr/rlds/v1/rlds_episode.proto`
- Local storage with upload queue

**Acceptance Criteria:**
- [ ] Recording starts/stops with single button
- [ ] Episodes include all multimodal data
- [ ] Schema validates against proto definition
- [ ] Episodes uploadable to ContinuonBrain

**Implementation Plan:**
1. Add record button to TrainerScreen
2. Wire camera frames to RldsRecorder observations
3. Wire TeleopController commands to RldsRecorder actions
4. Add audio capture to observations
5. Test schema validation with existing validator
6. Add upload UI with progress indicator

---

### F6: Teaching Mode (Phase 2)

**Description:** Natural language behavior teaching.

**User Stories:**
- As a user, I want to say "teach patrol" and demonstrate a behavior
- As a user, I want the robot to learn from my demonstration
- As a user, I want to run learned behaviors by name

**Teaching Flow:**
```
User: "teach patrol"
App: "Ready to learn 'patrol'. Show me what to do."
User: [drives robot in a pattern using controls]
User: "done"
App: "Learned 'patrol': forward → left → forward → left"

User: "patrol"
App: "Running 'patrol'..." [robot executes learned sequence]
```

**Technical Requirements:**
- Behavior state machine (idle → recording → learned)
- Action sequence storage (JSON or proto)
- Integration with Brain B for behavior execution
- LLM for natural language parsing (NexaSDK)

**Acceptance Criteria:**
- [ ] "teach X" enters recording mode
- [ ] Actions recorded during teaching
- [ ] "done" saves behavior
- [ ] "X" or "run X" executes behavior

**Implementation Plan:**
1. Create `TeachingMode.kt` state machine
2. Create `BehaviorStore.kt` for persisting learned behaviors
3. Add teaching commands to CommandParser
4. Record action sequences during teaching
5. Implement behavior playback via TeleopController
6. Wire to Brain B for execution on robot

---

### F7: Object Interaction (Phase 2)

**Description:** Natural language object manipulation.

**User Stories:**
- As a user, I want to say "pick up the red cup"
- As a user, I want the robot to identify and grasp the object

**Technical Requirements:**
- VLM object grounding (find object in frame → bounding box)
- Coordinate transformation (image → robot workspace)
- Grasp planning (simplified: move to position, close gripper)
- NexaSDK VLM with object detection capabilities

**Acceptance Criteria:**
- [ ] VLM can locate named objects in camera view
- [ ] Robot moves to object location
- [ ] Gripper closes on object
- [ ] Success/failure feedback to user

**Implementation Plan:**
1. Add object grounding prompt to VisionPipeline
2. Create `ObjectLocator.kt` for image → 3D position
3. Create `GraspPlanner.kt` for simple pick actions
4. Wire to TeleopController for execution
5. Add confirmation prompts and error handling

---

## Technical Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Android Device                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                     NexaSDK Layer                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │  │
│  │  │ VLM Model   │  │ ASR Model   │  │ LLM Model   │       │  │
│  │  │ (Vision)    │  │ (Voice)     │  │ (Teaching)  │       │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │  │
│  └─────────┼────────────────┼────────────────┼───────────────┘  │
│            │                │                │                   │
│  ┌─────────▼────────────────▼────────────────▼───────────────┐  │
│  │                   Application Layer                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │  │
│  │  │ Vision     │  │ Voice      │  │ Teleop             │   │  │
│  │  │ Pipeline   │  │ Pipeline   │  │ Controller         │   │  │
│  │  └─────┬──────┘  └─────┬──────┘  └─────────┬──────────┘   │  │
│  │        │               │                   │               │  │
│  │  ┌─────▼───────────────▼───────────────────▼──────────┐   │  │
│  │  │               RLDS Recorder                         │   │  │
│  │  │  (observations + actions → episodes)                │   │  │
│  │  └─────────────────────┬───────────────────────────────┘   │  │
│  └─────────────────────────┼─────────────────────────────────┘  │
│                            │                                     │
│  ┌─────────────────────────▼─────────────────────────────────┐  │
│  │                    UI Layer (Compose)                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │ Camera   │  │ Voice    │  │ Drive    │  │ Arm      │   │  │
│  │  │ Preview  │  │ Panel    │  │ Controls │  │ Controls │   │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ gRPC / WebRTC
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Robot (Pi 5)                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    ContinuonBrain                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │  │
│  │  │ Brain B  │  │ Motor    │  │ Camera   │                 │  │
│  │  │ (Simple) │  │ Control  │  │ Stream   │                 │  │
│  │  └──────────┘  └──────────┘  └──────────┘                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Hardware: Motors, Arms, Grippers, Sensors                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Module Dependencies

```
MainActivity
    ├── config/AppConfig
    ├── nexa/NexaManager
    │     ├── VisionPipeline
    │     └── VoicePipeline
    ├── trainer/TrainerScreen
    │     ├── camera/CameraPreview
    │     ├── camera/DetectionOverlay
    │     ├── trainer/DriveControls
    │     ├── trainer/ArmControls
    │     ├── trainer/VoicePanel
    │     └── trainer/TeachingMode
    ├── teleop/TeleopController
    │     └── input/InputFusionEngine
    ├── connectivity/ContinuonBrainClient
    └── logging/RldsRecorder
```

### Data Flow

```
Camera Frame                Voice Audio
     │                           │
     ▼                           ▼
VisionPipeline              VoicePipeline
     │                           │
     ├──→ Scene Description      ├──→ Transcription
     │    (on demand)            │    (real-time)
     │                           │
     └──→ Object Detection       └──→ Command
          (continuous)                │
                                      ▼
                              CommandParser
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
              Drive Command      Arm Command      Teach Command
                    │                 │                 │
                    └────────┬────────┘                 │
                             │                          │
                             ▼                          ▼
                      TeleopController           TeachingMode
                             │                          │
                             └──────────┬───────────────┘
                                        │
                                        ▼
                                  RldsRecorder
                                        │
                                        ▼
                              ContinuonBrainClient
                                        │
                                        ▼
                                  Robot (Pi 5)
```

---

## Detailed API Specifications

### gRPC Service Contract

The Android app communicates with ContinuonBrain via the `ContinuonBrainBridgeService` defined in `proto/continuonxr/continuonbrain/v1/continuonbrain_link.proto`.

#### Service Methods

| Method | Type | Purpose |
|--------|------|---------|
| `StreamRobotState` | Server streaming | Real-time robot kinematics (30Hz) |
| `SendCommand` | Unary | Send control commands to robot |
| `GetCapabilityManifest` | Unary | Query robot skills, sensors, safety features |
| `StreamRobotEditorTelemetry` | Server streaming | Diagnostics + safety signals |
| `ListTasks` | Unary | Query available autonomous tasks |
| `SelectTask` | Unary | Select task for execution |

#### Control Command Types

```kotlin
sealed class ControlCommand {
    // End-effector Cartesian twist (meters/sec, radians/sec)
    data class EndEffectorVelocity(
        val linearMps: Vector3,
        val angularRadS: Vector3,
        val referenceFrame: ReferenceFrame
    ) : ControlCommand()

    // Joint-space delta commands (radians)
    data class JointDelta(
        val deltaRadians: List<Float>  // Matches robot DoF ordering
    ) : ControlCommand()

    // Gripper control
    data class Gripper(
        val mode: GripperMode,
        val positionM: Float?,    // For POSITION mode
        val velocityMps: Float?   // For VELOCITY mode
    ) : ControlCommand()
}

enum class ReferenceFrame { BASE, TOOL }
enum class GripperMode { POSITION, VELOCITY }
```

#### Robot State Observation

```kotlin
data class RobotState(
    val timestampNanos: Long,
    val jointPositions: List<Float>,      // Current joint angles (radians)
    val jointVelocities: List<Float>,     // Joint velocities (rad/s)
    val jointEfforts: List<Float>,        // Joint torques (Nm)
    val endEffectorPose: Pose,            // EE position + orientation
    val endEffectorTwist: List<Float>,    // EE velocity (6-DOF)
    val gripperOpen: Boolean,
    val frameId: String                   // Coordinate frame ("base_link")
)

data class Pose(
    val position: List<Float>,            // [x, y, z] meters
    val orientationQuat: List<Float>      // [qx, qy, qz, qw]
)
```

#### Safety Status Protocol

```kotlin
data class SafetyStatus(
    val estopReleasedAck: Boolean,  // Operator acknowledges e-stop released
    val safetyToken: String?        // Optional OEM safety interlock token
)

data class SafetyState(
    val estopEngaged: Boolean,
    val rateLimited: Boolean,
    val envelopeViolated: Boolean,
    val predictedCollisionHorizonS: Double,
    val safetyHeadState: String,
    val activeEnvelopes: List<String>
)
```

### REST API Fallback

For scenarios where gRPC is unavailable, the app supports REST API communication:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/discovery/info` | GET | Robot discovery on LAN |
| `/api/status` | GET | Robot status snapshot |
| `/api/joints` | POST | Joint control command |
| `/api/drive` | POST | Drive control command |
| `/api/camera/stream` | GET | MJPEG camera stream |
| `/api/chat` | POST | Send message to HOPE brain |

#### Discovery Response Schema

```json
{
  "status": "ok",
  "product": "continuon_brain_runtime",
  "device_id": "pi5-abc123def456",
  "robot_name": "ContinuonBot",
  "version": "0.1.0",
  "capabilities": ["arm_control", "depth_vision", "training_mode"],
  "base_url": "http://192.168.1.100:8081",
  "endpoints": {
    "status": "/api/status",
    "pair_start": "/api/ownership/pair/start"
  }
}
```

---

## State Management Architecture

### ViewModel Hierarchy

```
TrainerViewModel (root)
├── CameraState
│   ├── previewActive: Boolean
│   ├── currentFrame: ImageBitmap?
│   ├── detections: List<Detection>
│   └── sceneDescription: String?
├── VoiceState
│   ├── listening: Boolean
│   ├── transcript: String
│   ├── lastCommand: ParsedCommand?
│   └── confidence: Float
├── TeleopState
│   ├── connected: Boolean
│   ├── robotState: RobotState?
│   ├── pendingCommand: ControlCommand?
│   └── commandLatencyMs: Long
├── RecordingState
│   ├── recording: Boolean
│   ├── episodeId: String?
│   ├── stepCount: Int
│   └── durationMs: Long
└── TeachingState
    ├── mode: TeachingMode
    ├── behaviorName: String?
    ├── recordedActions: List<Action>
    └── learnedBehaviors: Map<String, Behavior>
```

### StateFlow Pattern

```kotlin
class TrainerViewModel(
    private val brainClient: ContinuonBrainClient,
    private val nexaManager: NexaManager,
    private val rldsRecorder: RldsRecorder
) : ViewModel() {

    // Immutable state exposed to UI
    private val _uiState = MutableStateFlow(TrainerUiState())
    val uiState: StateFlow<TrainerUiState> = _uiState.asStateFlow()

    // Robot state stream (30Hz from gRPC)
    private val robotStateFlow: Flow<RobotState> = brainClient
        .robotStateFlow()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), null)

    // Derived state for UI
    val isConnected: StateFlow<Boolean> = robotStateFlow
        .map { it != null }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(), false)

    // Command rate limiting (20Hz max)
    private val commandThrottler = CommandThrottler(
        maxRateHz = 20.0,
        scope = viewModelScope
    )

    fun sendDriveCommand(direction: DriveDirection, speed: Float) {
        commandThrottler.throttle {
            val command = ControlCommand.EndEffectorVelocity(
                linearMps = direction.toLinearVelocity(speed),
                angularRadS = direction.toAngularVelocity(speed),
                referenceFrame = ReferenceFrame.BASE
            )
            brainClient.sendCommand(command)
            recordAction(command)
        }
    }
}
```

### Side Effect Handling

```kotlin
sealed class TrainerSideEffect {
    data class ShowToast(val message: String) : TrainerSideEffect()
    data class PlayHaptic(val pattern: HapticPattern) : TrainerSideEffect()
    data class NavigateTo(val destination: String) : TrainerSideEffect()
    object RequestPermissions : TrainerSideEffect()
}

// In ViewModel
private val _sideEffects = Channel<TrainerSideEffect>(Channel.BUFFERED)
val sideEffects: Flow<TrainerSideEffect> = _sideEffects.receiveAsFlow()

// In Composable
LaunchedEffect(Unit) {
    viewModel.sideEffects.collect { effect ->
        when (effect) {
            is TrainerSideEffect.ShowToast ->
                Toast.makeText(context, effect.message, Toast.LENGTH_SHORT).show()
            is TrainerSideEffect.PlayHaptic ->
                hapticFeedback.performHapticFeedback(effect.pattern.toHapticType())
            // ...
        }
    }
}
```

---

## Error Handling & Resilience

### Error Categories

| Category | Recovery Strategy | User Feedback |
|----------|-------------------|---------------|
| **Network** | Exponential backoff retry | "Reconnecting..." toast |
| **gRPC Stream** | Auto-reconnect with state sync | Status indicator change |
| **NexaSDK** | Fallback to CPU, reduce quality | "Running on CPU" badge |
| **Camera** | Request permission, retry | Permission dialog |
| **Audio** | Request permission, retry | Permission dialog |
| **Recording** | Save partial, create new episode | "Recording saved" |
| **Safety** | Halt motion, require ack | Emergency stop dialog |

### Retry Policy

```kotlin
object RetryPolicy {
    const val INITIAL_DELAY_MS = 500L
    const val MAX_DELAY_MS = 5_000L
    const val MAX_ATTEMPTS = 5
    const val BACKOFF_MULTIPLIER = 2.0

    fun calculateDelay(attempt: Int): Long {
        val delay = INITIAL_DELAY_MS * BACKOFF_MULTIPLIER.pow(attempt).toLong()
        return minOf(delay, MAX_DELAY_MS)
    }
}

// Usage in Flow
brainClient.robotStateFlow()
    .retryWhen { cause, attempt ->
        if (attempt >= RetryPolicy.MAX_ATTEMPTS) {
            emit(ConnectionState.Failed(cause))
            return@retryWhen false
        }
        emit(ConnectionState.Reconnecting(attempt))
        delay(RetryPolicy.calculateDelay(attempt.toInt()))
        true
    }
```

### Safety Error Handling

```kotlin
sealed class SafetyError {
    object EstopEngaged : SafetyError()
    object EnvelopeViolated : SafetyError()
    object RateLimited : SafetyError()
    data class CollisionPredicted(val horizonMs: Long) : SafetyError()
}

fun handleSafetyError(error: SafetyError) {
    when (error) {
        SafetyError.EstopEngaged -> {
            // Halt all commands, show acknowledgment dialog
            _uiState.update { it.copy(
                safetyDialog = SafetyDialog.EstopAck,
                teleopEnabled = false
            )}
        }
        SafetyError.EnvelopeViolated -> {
            // Show workspace boundary warning
            playHaptic(HapticPattern.WARNING)
            showToast("Workspace limit reached")
        }
        is SafetyError.CollisionPredicted -> {
            // Slow down, show warning
            commandThrottler.setMaxRate(5.0) // Reduce to 5Hz
            showOverlay(CollisionWarning(error.horizonMs))
        }
    }
}
```

### Graceful Degradation

```kotlin
class NexaFallbackManager(private val nexaManager: NexaManager) {

    sealed class InferenceMode {
        object NPU : InferenceMode()      // Full speed on Hexagon
        object GPU : InferenceMode()      // Fallback to Adreno
        object CPU : InferenceMode()      // Last resort
        object Disabled : InferenceMode() // Model unavailable
    }

    private val _inferenceMode = MutableStateFlow<InferenceMode>(InferenceMode.NPU)
    val inferenceMode: StateFlow<InferenceMode> = _inferenceMode.asStateFlow()

    suspend fun runInference(input: VisionInput): VisionOutput {
        return try {
            nexaManager.runVlm(input, plugin = "npu")
        } catch (e: NexaNpuUnavailableException) {
            _inferenceMode.value = InferenceMode.GPU
            try {
                nexaManager.runVlm(input, plugin = "gpu")
            } catch (e: NexaGpuException) {
                _inferenceMode.value = InferenceMode.CPU
                nexaManager.runVlm(input, plugin = "cpu")
            }
        }
    }
}
```

---

## Security & Privacy

### Data Classification

| Data Type | Classification | Storage | Transmission |
|-----------|----------------|---------|--------------|
| Camera frames | Sensitive | Memory only (no disk) | Local network only |
| Voice audio | Sensitive | Memory only | On-device processing |
| RLDS episodes | User data | Encrypted local storage | User-initiated upload |
| Robot state | Operational | Memory only | Local network |
| Learned behaviors | User data | Encrypted local storage | Never uploaded |

### Permission Management

```kotlin
object RequiredPermissions {
    val CAMERA = arrayOf(
        Manifest.permission.CAMERA
    )
    val AUDIO = arrayOf(
        Manifest.permission.RECORD_AUDIO
    )
    val NETWORK = arrayOf(
        Manifest.permission.INTERNET,
        Manifest.permission.ACCESS_NETWORK_STATE,
        Manifest.permission.ACCESS_WIFI_STATE
    )
    val STORAGE = arrayOf(
        Manifest.permission.WRITE_EXTERNAL_STORAGE // Only for RLDS export
    )

    val ALL = CAMERA + AUDIO + NETWORK + STORAGE
}

// Permission request flow
@Composable
fun PermissionGate(
    permissions: Array<String>,
    onGranted: @Composable () -> Unit,
    onDenied: @Composable () -> Unit
) {
    val permissionState = rememberMultiplePermissionsState(permissions.toList())

    LaunchedEffect(Unit) {
        if (!permissionState.allPermissionsGranted) {
            permissionState.launchMultiplePermissionRequest()
        }
    }

    when {
        permissionState.allPermissionsGranted -> onGranted()
        permissionState.shouldShowRationale -> PermissionRationale(permissions)
        else -> onDenied()
    }
}
```

### Network Security

```kotlin
// gRPC channel with TLS
val channel = OkHttpChannelBuilder
    .forAddress(config.host, config.port)
    .useTransportSecurity()
    .sslSocketFactory(createTlsSocketFactory())
    .build()

// Certificate pinning for production
fun createTlsSocketFactory(): SSLSocketFactory {
    val certificatePinner = CertificatePinner.Builder()
        .add("*.continuon.ai", "sha256/AAAA...")
        .build()
    // ... configure SSL context
}

// Local network discovery validation
fun validateLocalRobot(discoveryInfo: DiscoveryInfo): Boolean {
    // Only accept robots on local subnet
    val robotIp = InetAddress.getByName(discoveryInfo.host)
    val localSubnet = NetworkInterface.getNetworkInterfaces()
        .asSequence()
        .flatMap { it.inetAddresses.asSequence() }
        .filterIsInstance<Inet4Address>()
        .firstOrNull()

    return robotIp.address.take(3) == localSubnet?.address?.take(3)
}
```

### On-Device AI Privacy

```kotlin
// All AI inference happens on-device
class NexaPrivacyPolicy {
    // Models run entirely on Hexagon NPU - no cloud calls
    val cloudEnabled = false

    // Camera frames never leave device
    val cameraFramesStored = false

    // Voice audio processed locally, then discarded
    val voiceAudioStored = false

    // RLDS episodes stored locally with user control
    val episodesEncrypted = true
    val episodesUploadRequiresConsent = true
}
```

---

## Performance Optimization

### Memory Management

| Component | Budget | Strategy |
|-----------|--------|----------|
| VLM Model | 4 GB | Load on demand, unload when backgrounded |
| ASR Model | 500 MB | Keep loaded while voice enabled |
| Camera Preview | 50 MB | Ring buffer, 3 frames max |
| RLDS Buffer | 100 MB | Flush to disk every 100 steps |
| gRPC Buffers | 10 MB | Bounded channels, drop oldest |

```kotlin
class MemoryManager(private val context: Context) {
    private val activityManager = context.getSystemService<ActivityManager>()

    fun getAvailableMemoryMb(): Long {
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager?.getMemoryInfo(memoryInfo)
        return memoryInfo.availMem / (1024 * 1024)
    }

    fun shouldUnloadModels(): Boolean {
        return getAvailableMemoryMb() < 500 ||
               activityManager?.isLowRamDevice == true
    }

    // Lifecycle-aware model management
    fun onAppBackgrounded() {
        if (getAvailableMemoryMb() < 1000) {
            nexaManager.unloadVlm()
            System.gc()
        }
    }
}
```

### Battery Optimization

```kotlin
object PowerProfile {
    // Full performance when plugged in
    val PLUGGED = PowerConfig(
        vlmInferenceRate = 5.0,   // 5 FPS
        stateStreamRate = 30.0,   // 30 Hz
        commandRate = 20.0,       // 20 Hz
        screenBrightness = 1.0f
    )

    // Balanced when on battery
    val BATTERY = PowerConfig(
        vlmInferenceRate = 2.0,   // 2 FPS
        stateStreamRate = 15.0,   // 15 Hz
        commandRate = 20.0,       // 20 Hz (safety critical)
        screenBrightness = 0.7f
    )

    // Low power when battery < 20%
    val LOW_BATTERY = PowerConfig(
        vlmInferenceRate = 1.0,   // 1 FPS
        stateStreamRate = 10.0,   // 10 Hz
        commandRate = 20.0,       // 20 Hz (safety critical)
        screenBrightness = 0.5f
    )
}

// Battery state observer
class BatteryMonitor(context: Context) {
    val batteryState: Flow<BatteryState> = callbackFlow {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                val plugged = intent.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0)

                trySend(BatteryState(
                    level = level * 100 / scale,
                    isCharging = plugged != 0
                ))
            }
        }
        context.registerReceiver(receiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        awaitClose { context.unregisterReceiver(receiver) }
    }
}
```

### Frame Processing Pipeline

```kotlin
class VisionPipeline(
    private val nexaManager: NexaManager,
    private val scope: CoroutineScope
) {
    // Backpressure handling - drop frames if inference is slow
    private val frameChannel = Channel<CameraFrame>(
        capacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    // Inference runs on dedicated dispatcher
    private val inferenceDispatcher = Dispatchers.Default.limitedParallelism(1)

    fun processFrames(): Flow<VisionResult> = frameChannel
        .receiveAsFlow()
        .flowOn(inferenceDispatcher)
        .mapLatest { frame ->
            measureTimeMillis {
                nexaManager.runVlm(frame.toBitmap())
            }.also { latencyMs ->
                _metrics.update { it.copy(inferenceLatencyMs = latencyMs) }
            }
        }

    fun submitFrame(frame: CameraFrame) {
        frameChannel.trySend(frame) // Non-blocking, drops if full
    }
}
```

### Latency Targets

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Touch → Command sent | < 16ms | End-to-end in TeleopController |
| Command → Robot motion | < 50ms | gRPC round-trip |
| Voice → Transcription | < 500ms | ASR inference time |
| Camera → Detection | < 200ms | VLM inference time |
| Recording start/stop | < 100ms | RLDS writer latency |

---

## NexaSDK Integration

### Dependencies

```kotlin
// build.gradle.kts
dependencies {
    // NexaSDK core
    implementation("ai.nexa:core:0.0.19")

    // CameraX for camera capture
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")
}
```

### Manifest

```xml
<manifest>
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.RECORD_AUDIO" />

    <application android:extractNativeLibs="true">
        <!-- ... -->
    </application>
</manifest>
```

### Model Selection

| Use Case | Model | Size | NPU Compatible |
|----------|-------|------|----------------|
| Scene Description | OmniNeural-4B | ~4GB | Yes |
| Object Detection | (built-in VLM) | — | Yes |
| Speech Recognition | Whisper-small | ~500MB | Yes |
| Command Parsing | (rule-based first, LLM later) | — | N/A |

### Initialization Pattern

```kotlin
class NexaManager(private val context: Context) {
    private var vlm: VlmWrapper? = null
    private var asr: AsrWrapper? = null

    suspend fun initialize() = withContext(Dispatchers.IO) {
        NexaSdk.getInstance().init(context)

        // Load VLM for vision
        vlm = VlmWrapper.builder()
            .vlmCreateInput(VlmCreateInput(
                model_name = "omni-neural",
                model_path = getModelPath("OmniNeural-4B"),
                plugin_id = "npu",
                config = ModelConfig()
            ))
            .build()
            .getOrNull()

        // Load ASR for voice
        asr = AsrWrapper.builder()
            .asrCreateInput(AsrCreateInput(
                model_name = "whisper-small",
                model_path = getModelPath("whisper-small"),
                plugin_id = "npu",
                config = ModelConfig()
            ))
            .build()
            .getOrNull()
    }

    suspend fun release() {
        vlm?.close()
        asr?.close()
    }
}
```

---

## UI Design

### Screen Layout (Portrait)

```
┌────────────────────────────────────────┐
│  ┌──────────────────────────────────┐  │
│  │                                  │  │
│  │         Camera Preview           │  │
│  │     (with detection overlay)     │  │
│  │                                  │  │
│  │  [Describe]              [Rec●]  │  │
│  └──────────────────────────────────┘  │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │ 🎤 "move forward"                │  │
│  │ ✓ Command recognized             │  │
│  └──────────────────────────────────┘  │
│                                        │
│  ┌────────────────┐  ┌──────────────┐  │
│  │                │  │ J1 ──●────── │  │
│  │   [Joystick]   │  │ J2 ────●──── │  │
│  │                │  │ J3 ──────●── │  │
│  │                │  │ J4 ────●──── │  │
│  └────────────────┘  │ J5 ──●────── │  │
│                      │ J6 ────●──── │  │
│  Speed: [====●===]   │              │  │
│                      │ Grip [===●=] │  │
│  [  🛑 STOP  ]       └──────────────┘  │
│                                        │
│  ─────────────────────────────────────│
│  [Trainer]  [Workstation]  [Observer] │
└────────────────────────────────────────┘
```

### Screen Layout (Landscape)

```
┌──────────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────┐  ┌───────────────────────────┐ │
│  │                              │  │ 🎤 "move forward"         │ │
│  │                              │  │ ✓ Command recognized      │ │
│  │       Camera Preview         │  ├───────────────────────────┤ │
│  │   (with detection overlay)   │  │ J1 ──●──  J4 ────●──      │ │
│  │                              │  │ J2 ────●  J5 ──●────      │ │
│  │  [Describe]          [Rec●]  │  │ J3 ──────● J6 ────●──     │ │
│  │                              │  │ Gripper [========●===]    │ │
│  └──────────────────────────────┘  ├───────────────────────────┤ │
│  ┌─────────────┐                   │                           │ │
│  │  [Joystick] │   Speed [===●==]  │      [  🛑 STOP  ]        │ │
│  └─────────────┘                   │                           │ │
│  ──────────────────────────────────┴───────────────────────────┤ │
│  [Trainer]           [Workstation]           [Observer]         │
└──────────────────────────────────────────────────────────────────┘
```

### Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Primary | Deep Blue | #1565C0 |
| Secondary | Teal | #00897B |
| Recording | Red | #D32F2F |
| Stop Button | Red | #F44336 |
| Success | Green | #4CAF50 |
| Warning | Amber | #FFC107 |
| Background | Dark Gray | #121212 |
| Surface | Gray | #1E1E1E |

---

## Implementation Phases

### Phase 1: Bounty MVP (Feb 15, 2026)

**Week 1: Foundation**
- [ ] Add NexaSDK dependency and initialization
- [ ] Add CameraX integration
- [ ] Create VisionPipeline with basic VLM inference
- [ ] Create CameraPreview composable

**Week 2: Voice + Controls**
- [ ] Create VoicePipeline with ASR
- [ ] Implement CommandParser
- [ ] Create DriveControls (joystick)
- [ ] Create ArmControls (sliders)
- [ ] Wire controls to TeleopController

**Week 3: Integration + Polish**
- [ ] Create TrainerScreen layout
- [ ] Integrate all components
- [ ] Test on Snapdragon device
- [ ] Record demo video
- [ ] Write README with setup instructions
- [ ] Submit to Devpost

### Phase 2: Grand Prize (Mar 24, 2026)

**Week 4-5: Teaching Mode**
- [ ] Implement TeachingMode state machine
- [ ] Create BehaviorStore
- [ ] Add teaching commands to parser
- [ ] Test behavior recording and playback

**Week 6: Object Interaction**
- [ ] Add object grounding to VisionPipeline
- [ ] Implement ObjectLocator
- [ ] Create simple GraspPlanner
- [ ] Test pick-and-place commands

**Week 7: Production Polish**
- [ ] Error handling and edge cases
- [ ] Onboarding flow
- [ ] Settings screen
- [ ] Performance optimization
- [ ] Play Store submission or APK release

---

## Testing Strategy

### Test Pyramid

```
                    ┌───────────────┐
                    │   E2E Tests   │  ← 5% (Real robot, real device)
                   ─┴───────────────┴─
                  ┌───────────────────┐
                  │ Integration Tests │  ← 25% (Mock robot, real NexaSDK)
                 ─┴───────────────────┴─
                ┌───────────────────────┐
                │     Unit Tests        │  ← 70% (Pure logic, ViewModels)
               ─┴───────────────────────┴─
```

### Unit Tests

| Component | Test Focus | Coverage Target |
|-----------|------------|-----------------|
| `CommandParser` | Voice command parsing, fuzzy matching | 95% |
| `TeleopController` | State machine transitions, rate limiting | 90% |
| `RldsRecorder` | Schema compliance, step ordering | 95% |
| `BehaviorStore` | Persistence, playback timing | 90% |
| `VisionPipeline` | Frame buffering, backpressure | 85% |
| `TrainerViewModel` | State updates, side effects | 90% |

```kotlin
// Example: CommandParser unit test
class CommandParserTest {

    private val parser = CommandParser()

    @Test
    fun `forward command parses correctly`() {
        val result = parser.parse("move forward")
        assertIs<ParsedCommand.Drive>(result)
        assertEquals(DriveDirection.FORWARD, result.direction)
    }

    @Test
    fun `fuzzy matching handles typos`() {
        val result = parser.parse("forwrd")  // typo
        assertIs<ParsedCommand.Drive>(result)
        assertEquals(DriveDirection.FORWARD, result.direction)
    }

    @Test
    fun `teach command extracts behavior name`() {
        val result = parser.parse("teach patrol")
        assertIs<ParsedCommand.Teach>(result)
        assertEquals("patrol", result.behaviorName)
    }

    @Test
    fun `unknown command returns null`() {
        val result = parser.parse("xyzzy")
        assertNull(result)
    }
}
```

### Integration Tests

```kotlin
// Example: RLDS recording integration test
class RldsRecordingIntegrationTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private lateinit var recorder: RldsRecorder
    private lateinit var mockBrainClient: ContinuonBrainClient

    @Before
    fun setup() {
        mockBrainClient = mockk {
            every { robotStateFlow() } returns flowOf(
                RobotState(
                    timestampNanos = 1000000000L,
                    jointPositions = listOf(0f, 1.57f, -1.57f, 0f, 0f, 0f),
                    gripperOpen = true
                )
            )
        }
        recorder = RldsRecorder(
            outputDir = tempFolder.root,
            brainClient = mockBrainClient
        )
    }

    @Test
    fun `recording captures complete episode`() = runTest {
        // Start recording
        recorder.startRecording("test_episode")

        // Simulate teleop actions
        repeat(10) { step ->
            recorder.recordAction(
                action = Action(
                    command = listOf(0.1f, 0f, 0f, 0f, 0f, 0f),
                    source = "test"
                )
            )
            advanceTimeBy(100) // 100ms between steps
        }

        // Stop and get episode
        val episode = recorder.stopRecording()

        // Validate schema
        val validator = RldsValidator()
        val result = validator.validate(episode)
        assertTrue(result.isValid)
        assertEquals(10, result.stepCount)
    }

    @Test
    fun `episode schema matches proto definition`() = runTest {
        recorder.startRecording("schema_test")
        recorder.recordAction(Action(listOf(0f), "test"))
        val episode = recorder.stopRecording()

        // Validate against proto schema
        val json = episode.toJson()
        val parsed = RldsEpisode.parseFrom(json)

        assertNotNull(parsed.metadata)
        assertNotNull(parsed.metadata.schemaVersion)
        assertEquals("1.1", parsed.metadata.schemaVersion)
    }
}
```

### End-to-End Tests

```kotlin
// Example: Full teleop flow E2E test
@LargeTest
@RunWith(AndroidJUnit4::class)
class TeleopE2ETest {

    @get:Rule
    val composeRule = createAndroidComposeRule<MainActivity>()

    @Test
    fun teleop_joystick_sends_commands() {
        // Wait for connection
        composeRule.waitUntil(10_000) {
            composeRule.onNodeWithTag("connection_indicator")
                .fetchSemanticsNode()
                .config[SemanticsProperties.Text]
                .any { it.text == "Connected" }
        }

        // Drag joystick forward
        composeRule.onNodeWithTag("joystick")
            .performTouchInput {
                down(center)
                moveTo(center.copy(y = center.y - 100))
                up()
            }

        // Verify command was sent (check logs or mock)
        // In real E2E, verify robot actually moved
    }

    @Test
    fun voice_command_controls_robot() {
        // Grant audio permission
        grantAudioPermission()

        // Tap mic button
        composeRule.onNodeWithTag("mic_button").performClick()

        // Wait for listening state
        composeRule.waitUntil {
            composeRule.onNodeWithTag("voice_status")
                .fetchSemanticsNode()
                .config[SemanticsProperties.Text]
                .any { it.text.contains("Listening") }
        }

        // Simulate voice input (requires TTS or audio injection)
        // Verify robot responds
    }
}
```

### Mock Infrastructure

```kotlin
// Mock ContinuonBrain server for testing
class MockContinuonBrainServer {
    private val robotState = MutableStateFlow(RobotState.DEFAULT)
    private val receivedCommands = mutableListOf<ControlCommand>()

    fun startServer(port: Int = 50051) {
        val server = ServerBuilder.forPort(port)
            .addService(MockBridgeService())
            .build()
            .start()
    }

    inner class MockBridgeService : ContinuonBrainBridgeServiceImplBase() {
        override fun streamRobotState(
            request: StreamRobotStateRequest,
            responseObserver: StreamObserver<StreamRobotStateResponse>
        ) {
            robotState
                .onEach { state ->
                    responseObserver.onNext(state.toProto())
                }
                .launchIn(CoroutineScope(Dispatchers.IO))
        }

        override fun sendCommand(
            request: SendCommandRequest,
            responseObserver: StreamObserver<SendCommandResponse>
        ) {
            receivedCommands.add(request.toDomain())
            responseObserver.onNext(
                SendCommandResponse.newBuilder()
                    .setAccepted(true)
                    .build()
            )
            responseObserver.onCompleted()
        }
    }

    // Test helpers
    fun setRobotState(state: RobotState) {
        robotState.value = state
    }

    fun getReceivedCommands(): List<ControlCommand> = receivedCommands.toList()

    fun clearCommands() {
        receivedCommands.clear()
    }
}
```

### CI/CD Test Configuration

```yaml
# .github/workflows/android-tests.yml
name: Android Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          java-version: '17'
      - name: Run unit tests
        run: ./gradlew :apps:continuonxr:testDebugUnitTest

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          java-version: '17'
      - name: Run integration tests
        run: ./gradlew :apps:continuonxr:connectedDebugAndroidTest
        env:
          MOCK_ROBOT: true

  e2e-tests:
    runs-on: [self-hosted, snapdragon]  # Requires real device
    steps:
      - uses: actions/checkout@v4
      - name: Run E2E tests
        run: ./gradlew :apps:continuonxr:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.continuonxr.app.E2ETestSuite
```

---

## Offline Mode Specification

### Offline Capabilities Matrix

| Feature | Online | Offline | Notes |
|---------|--------|---------|-------|
| Camera preview | ✓ | ✓ | Always local |
| VLM inference | ✓ | ✓ | On-device NexaSDK |
| Voice commands | ✓ | ✓ | On-device ASR |
| RLDS recording | ✓ | ✓ | Local storage |
| Behavior teaching | ✓ | ✓ | Local storage |
| Behavior playback | ✓ | ✗ | Requires robot |
| Robot control | ✓ | ✗ | Requires robot |
| Episode upload | ✓ | Queued | Auto-upload when online |
| Model updates | ✓ | ✗ | Requires internet |

### Offline Mode Architecture

```kotlin
class OfflineModeManager(
    private val context: Context,
    private val episodeStore: EpisodeStore,
    private val behaviorStore: BehaviorStore
) {
    private val connectivityManager = context.getSystemService<ConnectivityManager>()

    sealed class ConnectionState {
        object Online : ConnectionState()
        object RobotConnected : ConnectionState()
        object OfflineWithLocalData : ConnectionState()
        object OfflineNoData : ConnectionState()
    }

    val connectionState: Flow<ConnectionState> = combine(
        networkState(),
        robotConnectionState()
    ) { network, robot ->
        when {
            robot -> ConnectionState.RobotConnected
            network -> ConnectionState.Online
            episodeStore.hasLocalEpisodes() -> ConnectionState.OfflineWithLocalData
            else -> ConnectionState.OfflineNoData
        }
    }.stateIn(scope, SharingStarted.WhileSubscribed(), ConnectionState.Online)

    // Offline-first RLDS recording
    suspend fun saveEpisode(episode: RldsEpisode) {
        // Always save locally first
        episodeStore.saveLocal(episode)

        // Queue for upload if online
        if (connectionState.value == ConnectionState.Online) {
            uploadQueue.enqueue(episode.id)
        }
    }

    // Auto-upload when connectivity restored
    init {
        connectionState
            .filter { it == ConnectionState.Online }
            .onEach { processUploadQueue() }
            .launchIn(scope)
    }

    private suspend fun processUploadQueue() {
        while (uploadQueue.isNotEmpty()) {
            val episodeId = uploadQueue.peek() ?: break
            try {
                val episode = episodeStore.getLocal(episodeId)
                uploadService.upload(episode)
                uploadQueue.remove()
                episodeStore.markUploaded(episodeId)
            } catch (e: NetworkException) {
                break // Stop processing, will retry when online
            }
        }
    }
}
```

### Local Storage Schema

```kotlin
// Room database for offline data
@Database(
    entities = [
        LocalEpisode::class,
        LocalBehavior::class,
        UploadQueueEntry::class,
        CachedModel::class
    ],
    version = 1
)
abstract class OfflineDatabase : RoomDatabase() {
    abstract fun episodeDao(): EpisodeDao
    abstract fun behaviorDao(): BehaviorDao
    abstract fun uploadQueueDao(): UploadQueueDao
    abstract fun modelCacheDao(): ModelCacheDao
}

@Entity(tableName = "episodes")
data class LocalEpisode(
    @PrimaryKey val id: String,
    val metadata: String,           // JSON metadata
    val stepsFilePath: String,      // Path to JSONL file
    val stepCount: Int,
    val durationMs: Long,
    val createdAt: Long,
    val uploadedAt: Long?,          // null if not uploaded
    val uploadStatus: UploadStatus
)

enum class UploadStatus {
    PENDING,
    QUEUED,
    UPLOADING,
    UPLOADED,
    FAILED
}

@Entity(tableName = "behaviors")
data class LocalBehavior(
    @PrimaryKey val name: String,
    val actionsJson: String,        // JSON array of actions
    val frameCount: Int,
    val durationMs: Long,
    val createdAt: Long,
    val lastUsedAt: Long?
)
```

### Offline UI Indicators

```kotlin
@Composable
fun ConnectionStatusBar(
    connectionState: ConnectionState,
    modifier: Modifier = Modifier
) {
    val backgroundColor = when (connectionState) {
        ConnectionState.RobotConnected -> Color(0xFF4CAF50) // Green
        ConnectionState.Online -> Color(0xFF2196F3)         // Blue
        ConnectionState.OfflineWithLocalData -> Color(0xFFFFC107) // Amber
        ConnectionState.OfflineNoData -> Color(0xFFF44336)  // Red
    }

    val text = when (connectionState) {
        ConnectionState.RobotConnected -> "Connected to robot"
        ConnectionState.Online -> "Online (no robot)"
        ConnectionState.OfflineWithLocalData -> "Offline - Local data available"
        ConnectionState.OfflineNoData -> "Offline - No data"
    }

    Surface(
        color = backgroundColor,
        modifier = modifier.fillMaxWidth()
    ) {
        Text(
            text = text,
            color = Color.White,
            modifier = Modifier.padding(8.dp),
            textAlign = TextAlign.Center
        )
    }
}
```

### Model Caching

```kotlin
class ModelCacheManager(
    private val context: Context,
    private val modelCacheDao: ModelCacheDao
) {
    private val cacheDir = File(context.filesDir, "nexa_models")

    data class CachedModelInfo(
        val modelName: String,
        val version: String,
        val sizeBytes: Long,
        val downloadedAt: Long,
        val localPath: String
    )

    suspend fun ensureModelAvailable(modelName: String): File {
        // Check local cache first
        val cached = modelCacheDao.getModel(modelName)
        if (cached != null && File(cached.localPath).exists()) {
            return File(cached.localPath)
        }

        // Download if online
        if (isOnline()) {
            return downloadModel(modelName)
        }

        // Offline and no cache - throw
        throw ModelUnavailableException(
            "Model $modelName not available offline. " +
            "Please connect to download."
        )
    }

    private suspend fun downloadModel(modelName: String): File {
        val modelFile = File(cacheDir, modelName)

        withContext(Dispatchers.IO) {
            nexaModelService.downloadModel(modelName, modelFile)
        }

        modelCacheDao.insert(CachedModel(
            name = modelName,
            version = "latest",
            sizeBytes = modelFile.length(),
            downloadedAt = System.currentTimeMillis(),
            localPath = modelFile.absolutePath
        ))

        return modelFile
    }

    fun getCacheSize(): Long = cacheDir.walkTopDown().sumOf { it.length() }

    suspend fun clearCache() {
        cacheDir.deleteRecursively()
        cacheDir.mkdirs()
        modelCacheDao.deleteAll()
    }
}
```

---

## Debug & Observability Infrastructure

### Logging Architecture

```kotlin
object AppLogger {
    private const val TAG_PREFIX = "ContinuonXR"

    enum class LogLevel { VERBOSE, DEBUG, INFO, WARN, ERROR }

    // Structured logging with context
    fun log(
        level: LogLevel,
        component: String,
        message: String,
        extras: Map<String, Any> = emptyMap()
    ) {
        val tag = "$TAG_PREFIX/$component"
        val formattedMessage = if (extras.isNotEmpty()) {
            "$message | ${extras.entries.joinToString(", ") { "${it.key}=${it.value}" }}"
        } else {
            message
        }

        when (level) {
            LogLevel.VERBOSE -> Log.v(tag, formattedMessage)
            LogLevel.DEBUG -> Log.d(tag, formattedMessage)
            LogLevel.INFO -> Log.i(tag, formattedMessage)
            LogLevel.WARN -> Log.w(tag, formattedMessage)
            LogLevel.ERROR -> Log.e(tag, formattedMessage)
        }

        // Also write to file for debug builds
        if (BuildConfig.DEBUG) {
            fileLogger.write(level, tag, formattedMessage)
        }
    }

    // Component-specific loggers
    val teleop = ComponentLogger("Teleop")
    val vision = ComponentLogger("Vision")
    val voice = ComponentLogger("Voice")
    val rlds = ComponentLogger("RLDS")
    val grpc = ComponentLogger("gRPC")
}

class ComponentLogger(private val component: String) {
    fun d(message: String, vararg extras: Pair<String, Any>) =
        AppLogger.log(LogLevel.DEBUG, component, message, extras.toMap())

    fun i(message: String, vararg extras: Pair<String, Any>) =
        AppLogger.log(LogLevel.INFO, component, message, extras.toMap())

    fun w(message: String, vararg extras: Pair<String, Any>) =
        AppLogger.log(LogLevel.WARN, component, message, extras.toMap())

    fun e(message: String, throwable: Throwable? = null, vararg extras: Pair<String, Any>) {
        AppLogger.log(LogLevel.ERROR, component, message, extras.toMap())
        throwable?.let { Log.e("$TAG_PREFIX/$component", "Exception", it) }
    }
}

// Usage
AppLogger.teleop.d("Command sent", "type" to "drive", "direction" to "forward")
AppLogger.grpc.i("Stream connected", "host" to config.host, "port" to config.port)
AppLogger.vision.w("Frame dropped", "reason" to "backpressure", "queueSize" to 3)
```

### Metrics Collection

```kotlin
class MetricsCollector {
    private val metrics = ConcurrentHashMap<String, AtomicLong>()
    private val timings = ConcurrentHashMap<String, MutableList<Long>>()

    // Counters
    fun increment(metric: String, delta: Long = 1) {
        metrics.getOrPut(metric) { AtomicLong(0) }.addAndGet(delta)
    }

    // Timing measurements
    inline fun <T> time(metric: String, block: () -> T): T {
        val start = SystemClock.elapsedRealtimeNanos()
        try {
            return block()
        } finally {
            val elapsed = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000 // ms
            timings.getOrPut(metric) { Collections.synchronizedList(mutableListOf()) }
                .add(elapsed)
        }
    }

    // Snapshot for debug UI
    fun snapshot(): MetricsSnapshot {
        return MetricsSnapshot(
            counters = metrics.mapValues { it.value.get() },
            timings = timings.mapValues { list ->
                if (list.isEmpty()) TimingStats.EMPTY
                else TimingStats(
                    count = list.size,
                    min = list.minOrNull() ?: 0,
                    max = list.maxOrNull() ?: 0,
                    avg = list.average().toLong(),
                    p95 = list.sorted().getOrNull((list.size * 0.95).toInt()) ?: 0
                )
            }
        )
    }
}

// Usage
metrics.increment("commands_sent")
metrics.increment("frames_processed")
metrics.time("vlm_inference") {
    nexaManager.runVlm(frame)
}
```

### Debug Overlay

```kotlin
@Composable
fun DebugOverlay(
    metrics: MetricsSnapshot,
    connectionState: ConnectionState,
    modifier: Modifier = Modifier
) {
    if (!BuildConfig.DEBUG) return

    var expanded by remember { mutableStateOf(false) }

    Box(modifier = modifier.padding(8.dp)) {
        // Floating debug button
        FloatingActionButton(
            onClick = { expanded = !expanded },
            modifier = Modifier.size(40.dp)
        ) {
            Icon(Icons.Default.BugReport, contentDescription = "Debug")
        }

        // Expanded panel
        AnimatedVisibility(visible = expanded) {
            Card(
                modifier = Modifier
                    .padding(top = 48.dp)
                    .width(280.dp)
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text("Debug Info", style = MaterialTheme.typography.titleSmall)
                    Divider()

                    // Connection
                    DebugRow("Connection", connectionState.name)

                    // Latencies
                    metrics.timings["grpc_roundtrip"]?.let {
                        DebugRow("gRPC Latency", "${it.avg}ms (p95: ${it.p95}ms)")
                    }
                    metrics.timings["vlm_inference"]?.let {
                        DebugRow("VLM Latency", "${it.avg}ms")
                    }

                    // Counters
                    DebugRow("Commands", "${metrics.counters["commands_sent"] ?: 0}")
                    DebugRow("Frames", "${metrics.counters["frames_processed"] ?: 0}")
                    DebugRow("Drops", "${metrics.counters["frames_dropped"] ?: 0}")

                    // Memory
                    val runtime = Runtime.getRuntime()
                    val usedMb = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
                    val maxMb = runtime.maxMemory() / 1024 / 1024
                    DebugRow("Memory", "${usedMb}MB / ${maxMb}MB")

                    // Battery
                    DebugRow("Power Mode", PowerProfile.current.name)
                }
            }
        }
    }
}

@Composable
private fun DebugRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, style = MaterialTheme.typography.bodySmall)
        Text(value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold)
    }
}
```

### Remote Debug Bridge

```kotlin
// For development: WebSocket debug interface
class DebugBridge(private val port: Int = 9999) {
    private val server: HttpServer = HttpServer.create(InetSocketAddress(port), 0)

    fun start() {
        server.createContext("/metrics") { exchange ->
            val snapshot = metrics.snapshot()
            val json = Json.encodeToString(snapshot)
            exchange.sendResponseHeaders(200, json.length.toLong())
            exchange.responseBody.write(json.toByteArray())
            exchange.close()
        }

        server.createContext("/logs") { exchange ->
            val logs = fileLogger.getRecentLogs(100)
            val json = Json.encodeToString(logs)
            exchange.sendResponseHeaders(200, json.length.toLong())
            exchange.responseBody.write(json.toByteArray())
            exchange.close()
        }

        server.createContext("/state") { exchange ->
            val state = viewModel.uiState.value
            val json = Json.encodeToString(state)
            exchange.sendResponseHeaders(200, json.length.toLong())
            exchange.responseBody.write(json.toByteArray())
            exchange.close()
        }

        server.start()
        Log.i("DebugBridge", "Debug server started on port $port")
    }

    fun stop() {
        server.stop(0)
    }
}

// Usage from development machine:
// curl http://<phone-ip>:9999/metrics
// curl http://<phone-ip>:9999/logs
// curl http://<phone-ip>:9999/state
```

### Crash Reporting

```kotlin
class CrashReporter : Thread.UncaughtExceptionHandler {
    private val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
    private val crashDir = File(context.filesDir, "crashes")

    override fun uncaughtException(thread: Thread, throwable: Throwable) {
        // Save crash report locally
        val report = CrashReport(
            timestamp = System.currentTimeMillis(),
            threadName = thread.name,
            exception = throwable.stackTraceToString(),
            deviceInfo = getDeviceInfo(),
            appState = captureAppState()
        )

        crashDir.mkdirs()
        val file = File(crashDir, "crash_${report.timestamp}.json")
        file.writeText(Json.encodeToString(report))

        // Call default handler to show system crash dialog
        defaultHandler?.uncaughtException(thread, throwable)
    }

    private fun captureAppState(): AppState {
        return AppState(
            connectionState = viewModel.uiState.value.connectionState.name,
            isRecording = viewModel.uiState.value.recordingState.recording,
            lastCommand = viewModel.uiState.value.teleopState.pendingCommand?.toString(),
            metrics = metrics.snapshot()
        )
    }

    private fun getDeviceInfo(): DeviceInfo {
        return DeviceInfo(
            manufacturer = Build.MANUFACTURER,
            model = Build.MODEL,
            sdkVersion = Build.VERSION.SDK_INT,
            appVersion = BuildConfig.VERSION_NAME,
            availableRamMb = getAvailableRam()
        )
    }
}
```

---

## Success Metrics

### Bounty Judging Criteria

| Criteria | Weight | Our Approach |
|----------|--------|--------------|
| Functionality | High | Real robot control, multimodal AI, RLDS recording |
| Commercialization | High | Home robotics market, clear user pain point |
| Originality | High | First robot trainer with on-device AI |
| Presentation | Medium | Demo with real robot, clear metrics |
| Community | Medium | Robotics + AI community interest |

### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Vision FPS | 15+ | Frames processed per second |
| Voice Latency | <500ms | Speech end → command recognized |
| Control Latency | <50ms | Touch → robot movement |
| RLDS Compliance | 100% | Schema validation pass rate |
| NPU Utilization | 80%+ | Via system profiler |

### User Metrics (Post-Launch)

| Metric | Target | Notes |
|--------|--------|-------|
| Session Length | 10+ min | Indicates engagement |
| Behaviors Taught | 3+/user | Adoption of teaching |
| Episodes Recorded | 50+/user | Training data generation |
| Crash Rate | <1% | Stability |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NexaSDK bugs | Medium | High | Test early, have fallback to CPU |
| Hexagon device unavailable | Low | Critical | Request device from Qualcomm, use emulator |
| ContinuonBrain connection issues | Medium | Medium | Mock mode for demo, robust retry |
| Model too slow for real-time | Medium | High | Use smaller models, async processing |
| Teaching mode complexity | Medium | Medium | Start with simple sequence replay |

---

## Open Questions

1. **Model delivery** — Bundle models in APK or download on first launch?
   - Recommendation: First launch download with progress UI

2. **Robot discovery** — mDNS or manual IP entry?
   - Recommendation: Both, with mDNS preferred

3. **Offline mode** — Support training without robot connection?
   - Recommendation: Yes, record episodes for later upload

4. **Multi-robot** — Support controlling multiple robots?
   - Recommendation: Out of scope for bounty, single robot focus

---

## Appendix

### A. Voice Command Grammar

```
command     := drive_cmd | arm_cmd | teach_cmd | system_cmd
drive_cmd   := ("move" | "go")? direction speed?
              | "stop"
              | "turn" ("left" | "right") degrees?
direction   := "forward" | "back" | "backward" | "left" | "right"
speed       := "slow" | "fast" | number "%"
degrees     := number "degrees"?

arm_cmd     := "arm" arm_action
              | joint_name ("up" | "down" | number)
              | "gripper" ("open" | "close" | number "%")
arm_action  := "up" | "down" | "home" | "ready"
joint_name  := "shoulder" | "elbow" | "wrist" | "j" [1-6]

teach_cmd   := "teach" behavior_name
              | "done" | "finished" | "cancel"
              | ("run" | "do" | "execute") behavior_name
behavior_name := word+

system_cmd  := "describe" | "what do you see"
              | "record" | "stop recording"
              | "help"
```

### B. RLDS Episode Schema (Key Fields)

```protobuf
message Episode {
  EpisodeMetadata metadata = 1;
  repeated Step steps = 2;
}

message Step {
  Observation observation = 1;
  Action action = 2;
  int64 timestamp_ns = 3;
}

message Observation {
  bytes camera_frame = 1;
  RobotState robot_state = 2;
  bytes audio = 3;
  string voice_transcript = 4;
}

message Action {
  CommandMode mode = 1;
  Vector3 command = 2;
  float gripper = 3;
  string source = 4;  // "voice", "touch", "playback"
}
```

### C. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| SoC | Snapdragon 8 Gen 2 | Snapdragon 8 Gen 4 |
| RAM | 8 GB | 12 GB |
| Storage | 8 GB free | 16 GB free |
| Android | 10 (API 29) | 14+ (API 34+) |
| Camera | Rear camera | Wide-angle preferred |
| Network | WiFi 5 | WiFi 6 for low latency |

### D. Related Documents

- `CLAUDE.md` — Development guidance
- `README.md` — Build instructions
- `AGENTS.md` — Agent-specific instructions
- `../../docs/rlds-schema.md` — Full RLDS specification
- `../../proto/continuonxr/` — Proto definitions
- Bounty: https://on-device-bounty-mobile.devpost.com/
- NexaSDK: https://github.com/NexaAI/nexa-sdk

### E. Implementation File Mapping

#### Existing Infrastructure (Leverage)

| Component | Existing File | Purpose | Reuse Strategy |
|-----------|---------------|---------|----------------|
| gRPC Client | `connectivity/ContinuonBrainClient.kt` | Robot state streaming, commands | Extend with NexaSDK integration |
| WebRTC Client | `connectivity/ContinuonBrainWebRtcClient.kt` | Low-latency fallback | Use as-is |
| Control Commands | `connectivity/ControlCommand.kt` | Type-safe command models | Use as-is |
| Robot State | `connectivity/ContinuonBrainClient.kt:359-376` | State data class | Use as-is |
| RLDS Writer | `logging/RldsEpisodeWriter.kt` | Episode file output | Extend with audio/vision |
| RLDS Recorder | `logging/RldsRecorder.kt` | Recording orchestration | Extend with new observations |
| RLDS Validator | `logging/RldsValidator.kt` | Schema validation | Use as-is |
| Config | `config/AppConfig.kt` | App configuration | Extend with NexaSDK config |
| Teleop Controller | `teleop/TeleopController.kt` | Command state machine | Extend with voice/vision |
| Input Fusion | `input/InputFusionEngine.kt` | Sensor fusion | Extend with joystick/voice |
| Audio Capture | `audio/AudioCapture.kt` | Mic recording for RLDS | Use as-is |

#### New Files to Create

| Component | New File | Purpose |
|-----------|----------|---------|
| **NexaSDK Layer** | | |
| Manager | `nexa/NexaManager.kt` | NexaSDK lifecycle, model loading |
| VLM Wrapper | `nexa/VlmWrapper.kt` | Vision-language inference |
| ASR Wrapper | `nexa/AsrWrapper.kt` | Speech recognition |
| **Vision Pipeline** | | |
| Pipeline | `vision/VisionPipeline.kt` | Frame → VLM → detections |
| Object Locator | `vision/ObjectLocator.kt` | Image coords → 3D position |
| Detection Models | `vision/DetectionModels.kt` | Detection data classes |
| **Voice Pipeline** | | |
| Pipeline | `voice/VoicePipeline.kt` | Audio → ASR → transcript |
| Command Parser | `voice/CommandParser.kt` | Transcript → commands |
| Grammar | `voice/VoiceGrammar.kt` | Command grammar definition |
| **UI Components** | | |
| Trainer Screen | `ui/trainer/TrainerScreen.kt` | Main trainer layout |
| Camera Preview | `ui/camera/CameraPreview.kt` | CameraX composable |
| Detection Overlay | `ui/camera/DetectionOverlay.kt` | Bounding box rendering |
| Virtual Joystick | `ui/controls/VirtualJoystick.kt` | Touch joystick |
| Drive Controls | `ui/controls/DriveControls.kt` | Drive panel wrapper |
| Arm Controls | `ui/controls/ArmControls.kt` | Joint sliders |
| Voice Panel | `ui/controls/VoicePanel.kt` | Mic button, transcript |
| **Teaching** | | |
| Teaching Mode | `teaching/TeachingMode.kt` | State machine |
| Behavior Store | `teaching/BehaviorStore.kt` | Persistence |
| Behavior Player | `teaching/BehaviorPlayer.kt` | Playback execution |
| **ViewModel** | | |
| Trainer ViewModel | `viewmodel/TrainerViewModel.kt` | State management |
| Camera ViewModel | `viewmodel/CameraViewModel.kt` | Camera state |
| **Testing** | | |
| Mock Server | `test/MockContinuonBrainServer.kt` | gRPC test server |
| Command Parser Tests | `test/CommandParserTest.kt` | Voice parsing tests |
| RLDS Integration Tests | `test/RldsRecordingIntegrationTest.kt` | Recording tests |

#### Proto Files (Reference)

| Proto | Path | Key Definitions |
|-------|------|-----------------|
| RLDS Episode | `proto/continuonxr/rlds/v1/rlds_episode.proto` | `Episode`, `Step`, `Observation`, `Action` |
| ContinuonBrain Link | `proto/continuonxr/continuonbrain/v1/continuonbrain_link.proto` | `ContinuonBrainBridgeService`, `ControlMode` |

### F. NexaSDK Code Examples

#### VLM Scene Description

```kotlin
// VlmWrapper.kt
class VlmWrapper(private val nexaSdk: NexaSdk) {
    private var model: VlmModel? = null

    suspend fun initialize(modelPath: String) {
        model = nexaSdk.loadVlm(
            VlmCreateInput(
                model_name = "omni-neural",
                model_path = modelPath,
                plugin_id = "npu",  // Hexagon NPU
                config = ModelConfig()
            )
        )
    }

    suspend fun describeScene(frame: Bitmap): String {
        val input = VlmInput(
            image = frame.toByteArray(),
            prompt = "Describe what you see in this image, focusing on objects and their positions."
        )
        return model?.generate(input)?.text ?: "Unable to describe scene"
    }

    suspend fun detectObjects(frame: Bitmap): List<Detection> {
        val input = VlmInput(
            image = frame.toByteArray(),
            prompt = "List all objects visible with their bounding boxes in format: object_name [x1,y1,x2,y2]"
        )
        val response = model?.generate(input)?.text ?: return emptyList()
        return parseDetections(response)
    }

    suspend fun locateObject(frame: Bitmap, objectName: String): BoundingBox? {
        val input = VlmInput(
            image = frame.toByteArray(),
            prompt = "Find the $objectName in this image and return its bounding box as [x1,y1,x2,y2]"
        )
        val response = model?.generate(input)?.text ?: return null
        return parseBoundingBox(response)
    }

    fun release() {
        model?.close()
        model = null
    }
}
```

#### ASR Voice Recognition

```kotlin
// AsrWrapper.kt
class AsrWrapper(private val nexaSdk: NexaSdk) {
    private var model: AsrModel? = null

    suspend fun initialize(modelPath: String) {
        model = nexaSdk.loadAsr(
            AsrCreateInput(
                model_name = "whisper-small",
                model_path = modelPath,
                plugin_id = "npu",
                config = AsrConfig(
                    language = "en",
                    beam_size = 5
                )
            )
        )
    }

    suspend fun transcribe(audio: ShortArray, sampleRate: Int): TranscriptionResult {
        val input = AsrInput(
            audio = audio,
            sample_rate = sampleRate
        )
        val result = model?.transcribe(input)
        return TranscriptionResult(
            text = result?.text ?: "",
            confidence = result?.confidence ?: 0f,
            latencyMs = result?.latency_ms ?: 0
        )
    }

    // Streaming transcription for real-time feedback
    fun transcribeStream(audioFlow: Flow<ShortArray>): Flow<PartialTranscription> {
        return audioFlow.map { chunk ->
            val result = model?.transcribePartial(AsrInput(chunk, 16000))
            PartialTranscription(
                partial = result?.partial_text ?: "",
                isFinal = result?.is_final ?: false
            )
        }
    }

    fun release() {
        model?.close()
        model = null
    }
}

data class TranscriptionResult(
    val text: String,
    val confidence: Float,
    val latencyMs: Long
)

data class PartialTranscription(
    val partial: String,
    val isFinal: Boolean
)
```

#### Command Parser with Fuzzy Matching

```kotlin
// CommandParser.kt
class CommandParser {
    private val driveCommands = mapOf(
        "forward" to DriveDirection.FORWARD,
        "move forward" to DriveDirection.FORWARD,
        "go forward" to DriveDirection.FORWARD,
        "ahead" to DriveDirection.FORWARD,
        "back" to DriveDirection.BACKWARD,
        "backward" to DriveDirection.BACKWARD,
        "reverse" to DriveDirection.BACKWARD,
        "left" to DriveDirection.LEFT,
        "turn left" to DriveDirection.LEFT,
        "right" to DriveDirection.RIGHT,
        "turn right" to DriveDirection.RIGHT,
        "stop" to DriveDirection.STOP,
        "halt" to DriveDirection.STOP
    )

    private val armCommands = mapOf(
        "arm up" to ArmCommand.UP,
        "arm down" to ArmCommand.DOWN,
        "arm home" to ArmCommand.HOME,
        "open gripper" to ArmCommand.GRIPPER_OPEN,
        "close gripper" to ArmCommand.GRIPPER_CLOSE
    )

    private val teachPattern = Regex("""(?:teach|learn|record)\s+(\w+)""", RegexOption.IGNORE_CASE)
    private val runPattern = Regex("""(?:run|do|execute|play)\s+(\w+)""", RegexOption.IGNORE_CASE)

    fun parse(transcript: String): ParsedCommand? {
        val normalized = transcript.trim().lowercase()

        // Check teach commands
        teachPattern.find(normalized)?.let { match ->
            return ParsedCommand.Teach(match.groupValues[1])
        }

        // Check run commands
        runPattern.find(normalized)?.let { match ->
            return ParsedCommand.RunBehavior(match.groupValues[1])
        }

        // Check done/finished
        if (normalized in listOf("done", "finished", "stop teaching", "cancel")) {
            return ParsedCommand.DoneTeaching
        }

        // Check drive commands with fuzzy matching
        driveCommands.entries.minByOrNull { levenshtein(normalized, it.key) }?.let { (pattern, direction) ->
            if (levenshtein(normalized, pattern) <= 2) {  // Allow up to 2 typos
                return ParsedCommand.Drive(direction)
            }
        }

        // Check arm commands
        armCommands.entries.minByOrNull { levenshtein(normalized, it.key) }?.let { (pattern, command) ->
            if (levenshtein(normalized, pattern) <= 2) {
                return ParsedCommand.Arm(command)
            }
        }

        // System commands
        return when {
            "describe" in normalized || "what do you see" in normalized ->
                ParsedCommand.DescribeScene
            "record" in normalized ->
                ParsedCommand.StartRecording
            "stop recording" in normalized ->
                ParsedCommand.StopRecording
            else -> null
        }
    }

    private fun levenshtein(a: String, b: String): Int {
        val dp = Array(a.length + 1) { IntArray(b.length + 1) }
        for (i in 0..a.length) dp[i][0] = i
        for (j in 0..b.length) dp[0][j] = j
        for (i in 1..a.length) {
            for (j in 1..b.length) {
                dp[i][j] = minOf(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + if (a[i-1] == b[j-1]) 0 else 1
                )
            }
        }
        return dp[a.length][b.length]
    }
}

sealed class ParsedCommand {
    data class Drive(val direction: DriveDirection) : ParsedCommand()
    data class Arm(val command: ArmCommand) : ParsedCommand()
    data class Teach(val behaviorName: String) : ParsedCommand()
    data class RunBehavior(val behaviorName: String) : ParsedCommand()
    object DoneTeaching : ParsedCommand()
    object DescribeScene : ParsedCommand()
    object StartRecording : ParsedCommand()
    object StopRecording : ParsedCommand()
}
```

### G. Dependency Versions

```kotlin
// build.gradle.kts additions for Phase 1

dependencies {
    // NexaSDK (on-device AI)
    implementation("ai.nexa:core:0.0.19")

    // CameraX (camera capture)
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // Compose (UI)
    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // gRPC (already in project)
    implementation("io.grpc:grpc-okhttp:1.64.0")
    implementation("io.grpc:grpc-stub:1.64.0")
    implementation("io.grpc:grpc-protobuf-lite:1.64.0")

    // Room (offline storage)
    val roomVersion = "2.6.1"
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")

    // Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")

    // Testing
    testImplementation("junit:junit:4.13.2")
    testImplementation("io.mockk:mockk:1.13.13")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
}

android {
    defaultConfig {
        // Extract native libs for NexaSDK
        ndk {
            abiFilters += listOf("arm64-v8a")  // Hexagon NPU only on arm64
        }
    }
}
```
