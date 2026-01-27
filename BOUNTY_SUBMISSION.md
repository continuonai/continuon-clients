# ContinuonXR — Nexa Bounty Submission

## Project Overview

**ContinuonXR** is a robot trainer app that enables users to teach personal robots through voice commands, camera vision, and direct control — all powered by on-device AI via Nexa SDK. It records training sessions as RLDS (Reinforcement Learning from Demonstration Sequences) episodes, creating a bridge between human demonstration and robot learning without cloud dependency.

## Nexa SDK Integration

### Models & NPU Features Used

| Component | Nexa SDK Feature | Details |
|-----------|-----------------|---------|
| **Vision Pipeline** | Nexa Vision (NPU) | Real-time camera frame analysis for object detection and scene understanding |
| **Voice Pipeline** | Nexa Voice (NPU) | On-device voice command recognition for hands-free robot control |
| **NexaManager** | Core SDK orchestration | Manages model lifecycle, NPU allocation, and inference scheduling |

### How Nexa SDK is Used

1. **NexaManager**: Central orchestrator (`NexaManager.kt`) that initializes and manages Nexa SDK models, handling NPU resource allocation across vision and voice pipelines.
2. **VisionPipeline**: Processes CameraX frames through Nexa's vision models for real-time object detection, scene understanding, and visual context — used during robot training to understand the environment.
3. **VoicePipeline**: Captures and processes voice commands through Nexa's on-device ASR for hands-free robot control during training sessions.
4. **RLDS Recording**: Training episodes capture Nexa's vision detections and voice inputs alongside robot actions, creating rich multimodal training data.

### NPU Acceleration Benefits
- **Real-time vision** at camera frame rate for responsive robot control
- **Simultaneous pipelines** — vision + voice running concurrently on NPU
- **Edge-first** — robots operate in environments without reliable internet

## Demo Instructions

1. Download the APK: [ContinuonXR.apk](https://github.com/continuonai/ContinuonXR/releases/latest/download/ContinuonXR.apk)
2. Install on an Android device (enable "Install from unknown sources")
3. Grant camera and microphone permissions
4. Use the Trainer screen:
   - **Drive Controls**: WASD-style pad to move the robot
   - **Voice Commands**: Say commands like "forward", "teach patrol", "done"
   - **Camera Feed**: Live WebRTC stream from the robot with Nexa vision overlay
   - **RLDS Recording**: Start a session to record training episodes
5. All AI processing runs on-device via Nexa SDK

## APK Download

📥 [Download ContinuonXR APK](https://github.com/continuonai/ContinuonXR/releases/latest/download/ContinuonXR.apk)

## Screenshots / Video

<!-- TODO: Add screenshots and demo video -->
- [ ] Trainer screen with drive controls and camera feed
- [ ] Voice command recognition in action
- [ ] RLDS episode recording indicator
- [ ] Vision pipeline detections overlay

## How It Meets Bounty Criteria

- **On-Device AI**: All vision and voice processing via Nexa SDK — no cloud AI calls
- **NPU Utilization**: Concurrent vision + voice pipelines on Qualcomm Hexagon NPU
- **Real-World Impact**: Enables anyone to train robots through demonstration, democratizing robotics
- **Novel Architecture**: "One Brain, Many Shells" — a single AI brain controls diverse robot hardware
- **Production Quality**: Native Android Kotlin app with Jetpack Compose, CameraX, WebRTC, protobuf/gRPC
- **Open Source**: Fully open-source with RLDS-compatible training data output

## Technical Stack

- **Language**: Kotlin + Jetpack Compose
- **AI Runtime**: Nexa SDK 0.0.19 (NPU/GPU/CPU)
- **Camera**: CameraX for frame capture
- **Comms**: gRPC + protobuf for robot communication, WebRTC for video streaming
- **XR**: AndroidX XR SceneCore for spatial interfaces
- **Platform**: Android 10+ (API 29), Android XR
