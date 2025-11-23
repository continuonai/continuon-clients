# Dev Setup (Draft)

Prereqs:
- Install JDK 17 (set `JAVA_HOME` and add `JAVA_HOME/bin` to PATH).
- Android Studio Koala or later with Android SDK 35 + build-tools 35.0.x; ensure commandline tools and platform-tools are installed.
- Kotlin 1.9.x tooling (bundled with Studio is fine).
- Gradle wrapper included (Gradle 8.7); protoc if generating protos manually.

Build/test:
- macOS/Linux: `./gradlew :apps:continuonxr:assembleDebug`
- Windows: `.\gradlew.bat :apps:continuonxr:assembleDebug`
- Unit tests: `./gradlew :apps:continuonxr:testDebugUnitTest`
- Generate Kotlin proto stubs: `./gradlew :apps:continuonxr:generateDebugProto`

XR app bring-up (local stub):
1. Open the project in Android Studio; let it sync with the Gradle wrapper.
2. Create or select an Android 35 device (emulator) or connect a physical device with developer mode + USB debugging enabled.
3. Build `:apps:continuonxr:assembleDebug` and deploy/run from Studio (UI is stubbed but starts teleop session in trainer mode).
4. Configure ContinuonBrain connectivity in `AppConfigLoader` (host/port + auth token for gRPC, or `signalingUrl`/`iceServers` for WebRTC).
5. Wear the XR headset, grant microphone + BLE permissions, and start the teleop session. Jetpack XR/SceneCore feeds headset/hand poses, gaze rays, and short PCM audio snippets into `TeleopController` for RLDS logging and command mapping.
6. Confirm robot state streaming end-to-end:
   - gRPC: robot state envelopes flow from `ContinuonBrainBridge/StreamRobotState` over OkHttp gRPC; commands return `CommandAck` via the same stub.
   - WebRTC: a WebSocket/WebRTC gateway at `signalingUrl` should echo `continuonbrain_link.proto` frames for robot state and accept binary command envelopes.

 Notes:
- Jetpack XR/SceneCore dependencies are now in `apps/continuonxr/build.gradle.kts`; sync the project once to download `androidx.xr` artifacts.
- ContinuonBrain/OS gRPC stubs use shared `proto/` definitions; ensure server and client are on the same schema revision when testing a headset build.
- RLDS uploader is stubbed (Noop); set `uploadOnComplete=true` only after wiring a real uploader.
- For BLE glove testing, plan to use a simulator or canned payloads before hardware is available.
- If the wrapper ever breaks, regenerate from a working Gradle install: `gradle wrapper --gradle-version 8.7 --distribution-type bin` at repo root, then re-run `./gradlew`.
