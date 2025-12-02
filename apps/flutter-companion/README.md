# ContinuonXR Flutter Companion

A Flutter module that targets Android/iOS host apps and provides:

- Teleoperation against the ContinuonBrain bridge gRPC/WebRTC service with TLS + bearer auth.
- Platform channel hook points for native SDKs to share transport/auth state.
- RLDS task recording and Cloud upload using signed URLs.
- Multimodal logging (audio, egocentric video/depth, gaze, voice command text) to mirror the Android XR app contract for Gemma and robot alignment.
- Minimal UI flows for connect, control, manual driving, and record/testing.

## Project layout

- `lib/main.dart` – entry-point wiring the connect, control, and record screens.
- `lib/services/brain_client.dart` – ContinuonBrain gRPC/WebRTC bridge with TLS/auth helpers.
- `lib/services/brain_client.dart` – thin ContinuonBrain gRPC/platform channel bridge.
- `lib/screens/manual_mode_screen.dart` – web-friendly manual driving surface with acceleration/gripper presets.
- `lib/services/controller_input.dart` – PS3 controller to `ControlCommand` mapper via platform channels.
- `lib/services/cloud_uploader.dart` – signed URL broker helper using `googleapis_auth`.
- `lib/services/multimodal_inputs.dart` – helper to stamp RLDS observations with synchronized audio, egocentric video/depth, gaze, and voice metadata.
- `lib/services/task_recorder.dart` – in-memory RLDS step collection.
- `lib/models` – data structures for teleop commands and RLDS metadata.
- `integration_test/` – basic UI smoke test covering connect/record flows.

## Requirements

- Flutter 3.19+ (`flutter upgrade` recommended).
- Android Studio or Xcode for embedding the module into the host app.
- Access to the ContinuonBrain endpoint and upload broker for RLDS episodes.

## Getting started

1. Fetch dependencies:
   ```bash
   flutter pub get
   ```
2. Configure ContinuonBrain connectivity in the Connect screen:
   - Default host: `brain.continuon.ai` port `443` with TLS enabled.
   - Optional bearer token: passed as the `Authorization: Bearer <token>` gRPC header.
   - Enable "platform WebRTC bridge" to route transport through the native host for
     lower-latency data channels when available.
   The same parameters can be provided programmatically via `BrainClient.connect`,
   including custom root certificates when connecting to staging clusters.
3. (Optional) If native hosts own the transport stack, implement the platform channel
   methods defined in `lib/services/platform_channels.dart` in the Android/iOS host
   shells to forward gRPC requests to ContinuonBrain.
4. Run the integration smoke test:
   ```bash
   flutter test integration_test/connect_and_record_test.dart
   ```
5. Build the module for Android:
4. Drive manually from a browser surface:
   ```bash
   flutter run -d chrome --web-port 8080
   ```
   Then open `/#/manual` to expose the operator controls. The same surface is available from
   the mobile shell via the Manual mode chips on the control screen.
4. Build the module for Android:
   ```bash
   flutter build aar
   ```
   or for iOS:
   ```bash
   flutter build ios-framework --cocoapods
   ```

## ContinuonBrain bindings

The client uses the same service and message shapes as `proto/continuonbrain_link.proto`.
By default the module uses JSON payloads over gRPC for portability, but mobile hosts can
swap in native protobuf-backed implementations through the platform channel to interop
with existing ContinuonBrain deployments. Commands and recording triggers are sent over
the `ContinuonBrainBridge` gRPC service with TLS by default; bearer tokens are attached
as gRPC metadata. When the platform WebRTC bridge is enabled, the Flutter layer defers
transport negotiation to the host and only serializes the payloads.

## RLDS upload

`CloudUploader` expects a broker endpoint that returns a signed upload URL for the RLDS
payload. The upload helper writes an `episode.json` based on the RLDS schema documented
in `proto/rlds_episode.proto` and issues a PUT to the signed URL. Replace the broker URL
with your environment-specific endpoint and scopes when authenticating with a service
account.

## Manual mode and controller input

- Use the **Manual driving** chip from the control screen to load the manual mode surface.
- Acceleration profiles (Precision/Nominal/Aggressive) scale the velocity commands before
  they are sent through `BrainClient.sendCommand`.
- Gripper presets include direct open/close toggles and a position slider mapping to
  `GripperCommand` payloads.
- RLDS records created from manual mode seed `control_role=manual_driver`; the Record screen
  lets you override the role for autonomous/automatic runs.
- A PlayStation 3 controller can be bridged via the `ps3_controller` platform channel. The
  listener maps left stick translation, right stick yaw, and Cross/Circle gripper buttons to
  `ControlCommand` fields. Implement the native channel on Android/iOS hosts or reuse an
  existing Flutter gamepad plugin that emits the same event keys.

## Embedding

This module is created with the Flutter module template (see `pubspec.yaml` `flutter`
section). Follow the official [Add Flutter to existing app](https://docs.flutter.dev/add-to-app)
workflow to wire the generated AAR/CocoaPods artifacts into the host app.
