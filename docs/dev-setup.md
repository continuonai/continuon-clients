# Dev Setup (Draft)

Prereqs:
- Install JDK 17 (set `JAVA_HOME` and add `JAVA_HOME/bin` to PATH).
- Android Studio Koala or later with Android SDK 35 + build-tools 35.0.x; ensure commandline tools and platform-tools are installed.
- Kotlin 1.9.x tooling (bundled with Studio is fine).
- Gradle wrapper included (Gradle 8.7); protoc if generating protos manually.

Build/test:
- macOS/Linux: `./gradlew :apps:xr:assembleDebug`
- Windows: `.\gradlew.bat :apps:xr:assembleDebug`
- Unit tests: `./gradlew :apps:xr:testDebugUnitTest`
- Generate Kotlin proto stubs: `./gradlew :apps:xr:generateDebugProto`

XR app bring-up (local stub):
1. Open the project in Android Studio; let it sync with the Gradle wrapper.
2. Create or select an Android 35 device (emulator) or connect a physical device with developer mode + USB debugging enabled.
3. Build `:apps:xr:assembleDebug` and deploy/run from Studio (UI is stubbed but starts teleop session in trainer mode).
4. Enable `useMockContinuonBrain=true` in config (default) to stream mock robot state until gRPC/WebRTC is implemented.

Notes:
- Jetpack XR/SceneCore dependencies are not yet added; expect to wire them into `apps/xr/build.gradle.kts`.
- ContinuonBrain/OS gRPC stubs use shared `proto/` definitions; keep proto and Kotlin types aligned. A mock stream is currently emitted from `ContinuonBrainClient` when `useMockContinuonBrain=true`.
- RLDS uploader is stubbed (Noop); set `uploadOnComplete=true` only after wiring a real uploader.
- For BLE glove testing, plan to use a simulator or canned payloads before hardware is available.
- If the wrapper ever breaks, regenerate from a working Gradle install: `gradle wrapper --gradle-version 8.7 --distribution-type bin` at repo root, then re-run `./gradlew`.
