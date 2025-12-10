# ContinuonAI Flutter App

A single Flutter app (web/iOS/Android/Linux) for Continuon teleop, RLDS capture, and cloud upload. This module now lives under `continuonai/` alongside the consolidated `continuon-cloud/` docs.

- Teleoperation against the ContinuonBrain bridge gRPC/WebRTC service with TLS + bearer auth.
- Platform channel hook points for native SDKs to share transport/auth state.
- RLDS task recording and Cloud upload using signed URLs.
- Multimodal logging (audio, egocentric video/depth, gaze, voice command text) to mirror the Android XR app contract for Gemma and robot alignment.
- Minimal UI flows for connect, control, manual driving, and record/testing.
- Integrated RLDS browser/annotation surfaces (WorldTape) now ride inside this Flutter module for consumer review and robot control without a separate web repo.
- Robot registration/ownership, subscription status, and OTA orchestration are managed here: a registered robot with an active subscription can receive signed OTA bundles (edge models and safety manifests).

## Project layout

- `lib/main.dart` – entry-point wiring the connect, control, and record screens.
- `lib/services/brain_client.dart` – ContinuonBrain gRPC/WebRTC bridge with TLS/auth helpers.
- `lib/services/platform_channels.dart` – thin ContinuonBrain gRPC/platform channel bridge.
- `lib/screens/manual_mode_screen.dart` – web-friendly manual driving surface with acceleration/gripper presets.
- `lib/services/controller_input.dart` - PS3 controller to `ControlCommand` mapper via platform channels.
- `lib/services/cloud_uploader.dart` - signed URL broker helper using `googleapis_auth`.
- `lib/services/multimodal_inputs.dart` - helper to stamp RLDS observations with synchronized audio, egocentric video/depth, gaze, and voice metadata.
- `lib/services/task_recorder.dart` - in-memory RLDS step collection.
- `lib/models` - data structures for teleop commands and RLDS metadata.
- `integration_test/` - basic UI smoke test covering connect/record flows.

## Integrated RLDS portal (WorldTape)

- The consumer-facing RLDS browser/annotation experience formerly tracked under `worldtapeai.com/` is consolidated into this Flutter app; use the web build for the portal surfaces and keep implementation here.
- Reuse the existing brain client and signed upload helpers to gate any review/export flows; ingest contracts stay aligned with `proto/rlds_episode.proto` and the upload checklist in `docs/upload-readiness-checklist.md`.
- The `worldtapeai.com/` folder in the repo now serves as a redirect stub only; updates to portal behavior should land in this module and its docs.

## Requirements

- Flutter 3.19+ (`flutter upgrade` recommended).
- Android Studio or Xcode for embedding the module into the host app.
- Access to the ContinuonBrain endpoint and upload broker for RLDS episodes.

## Getting started

1. From repo root, enter the app module:
   ```bash
   cd continuonai
   ```
2. Fetch dependencies:
   ```bash
   flutter pub get
   ```
3. Configure ContinuonBrain connectivity in the Connect screen:
   - Default host: `brain.continuon.ai` port `443` with TLS enabled.
   - Optional bearer token: passed as the `Authorization: Bearer <token>` gRPC header.
   - Enable "platform WebRTC bridge" to route transport through the native host for
     lower-latency data channels when available.
   The same parameters can be provided programmatically via `BrainClient.connect`,
   including custom root certificates when connecting to staging clusters.
4. (Optional) If native hosts own the transport stack, implement the platform channel
   methods defined in `lib/services/platform_channels.dart` in the Android/iOS host
   shells to forward gRPC requests to ContinuonBrain.
5. Run the integration smoke test:
   ```bash
   flutter test integration_test/connect_and_record_test.dart
   ```
6. Drive manually from a browser surface:
   ```bash
   flutter run -d chrome --web-port 8080
   ```
   Then open `/#/manual` to expose the operator controls. The same surface is available from
   the mobile shell via the Manual mode chips on the control screen.
7. Build the module for Android:
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

## OTA & seed model flow
- OTA delivery uses the signed edge bundle path (`docs/bundle_manifest.md`). The app gates download/apply on paid subscription status and robot ownership.
- The initial OTA payload is the **v0 seed model** trained via the Pi 5 + AI HAT pipeline (`docs/seed-model-plan.md`); subsequent updates follow the same OTA flow.
- UX expectations: surface eligibility/error states (`subscription required`, `signature/checksum failed`, `bundle unavailable`) to the operator; do not silently drop OTA attempts.

## Ownership & connectivity (local/remote)
- Registration/claim is **local-only**: the owner must be on the same LAN as the robot to claim it and perform the first ContinuonBrain install (v0 seed bundle).
- Initial ContinuonBrain install is **local-only**; no remote install before claim.
- After claim + initial install, remote control/OTA/Robot Editor access is allowed for the owner over TLS; enforce ownership + subscription on remote sessions.
- Local LAN access remains available for the owner even if remote reachability is blocked; discovery/claim always requires local presence.

### UX states to surface
- **Local claim required**: Robot visible on LAN, not owned; show “claim locally to proceed” and disable remote/OTA actions.
- **Initial install pending**: Claim succeeded but seed bundle not applied; force local install flow.
- **Remote allowed**: Ownership + subscription valid; enable control/OTA/Editor.
- **Blocked**: Show explicit reasons—`subscription required`, `ownership mismatch`, `signature/checksum failed`, `bundle unavailable`.

### Flow essentials
- Discovery: LAN-only for claim; remote discovery blocked until owned.
- Claim + seed install: download + verify signed bundle (checksums/signature) locally; record ownership token/device ID.
- Remote sessions: require ownership token + subscription before enabling control/OTA/Editor; allow read-only status view if subscription lapses.
- OTA UI: show states for download, verify (checksum/signature), apply, rollback on failure; display current + last-known-good bundle versions and provenance.

## UI checklist (Connect/Claim/OTA)
- Connect screen should:
  - Detect LAN presence; disable claim if not local.
  - Show robot ownership/subscription status; if unowned, show “claim locally to proceed.”
- Claim flow:
  - Scan LAN, select robot, claim; fetch/apply seed bundle locally; show checksum/signature verification result and rollback on failure.
  - Persist ownership token/device ID after successful claim/install.
- Remote session:
  - Require ownership + subscription before enabling control/OTA/Editor; allow read-only status if subscription lapses.
  - Surface errors clearly (subscription required, ownership mismatch, signature/checksum failed, bundle unavailable).

## Cloud docs

The Continuon-Cloud staging docs are now located under `continuonai/continuon-cloud/`. Keep those specs aligned with `docs/monorepo-structure.md` and the root README when flows span XR and Cloud.

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
