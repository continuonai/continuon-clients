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
