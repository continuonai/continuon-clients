# ContinuonXR Flutter Companion

A Flutter module that targets Android/iOS host apps and provides:

- Teleoperation against the ContinuonBrain bridge gRPC service.
- Platform channel hook points for native SDKs to share transport/auth state.
- RLDS task recording and Cloud upload using signed URLs.
- Minimal UI flows for connect, control, and record/testing.

## Project layout

- `lib/main.dart` – entry-point wiring the connect, control, and record screens.
- `lib/services/brain_client.dart` – thin ContinuonBrain gRPC/platform channel bridge.
- `lib/services/cloud_uploader.dart` – signed URL broker helper using `googleapis_auth`.
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
2. (Optional) If native hosts own the transport stack, implement the platform channel
   methods defined in `lib/services/platform_channels.dart` in the Android/iOS host
   shells to forward gRPC requests to ContinuonBrain.
3. Run the integration smoke test:
   ```bash
   flutter test integration_test/connect_and_record_test.dart
   ```
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
with existing ContinuonBrain deployments.

## RLDS upload

`CloudUploader` expects a broker endpoint that returns a signed upload URL for the RLDS
payload. The upload helper writes an `episode.json` based on the RLDS schema documented
in `proto/rlds_episode.proto` and issues a PUT to the signed URL. Replace the broker URL
with your environment-specific endpoint and scopes when authenticating with a service
account.

## Embedding

This module is created with the Flutter module template (see `pubspec.yaml` `flutter`
section). Follow the official [Add Flutter to existing app](https://docs.flutter.dev/add-to-app)
workflow to wire the generated AAR/CocoaPods artifacts into the host app.
