# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ContinuonAI is a Flutter companion app (web/iOS/Android/Linux) for robot teleoperation, RLDS (Robot Learning and Demonstrations) data capture, and cloud upload. It's part of the Continuon self-learning robotics ecosystem using a "One Brain, Many Shells" architecture.

The app communicates with ContinuonBrain (edge runtime on robots) via gRPC/WebRTC using the RCAN protocol for robot discovery, authentication, and control.

## Build Commands

```bash
# Install dependencies
flutter pub get

# Run analysis/linting
flutter analyze

# Run unit/widget tests
flutter test

# Run integration tests (connect/record flow)
flutter test integration_test/connect_and_record_test.dart

# Run web dev server
flutter run -d chrome --web-port 8080

# Build Android module
flutter build aar

# Build iOS module
flutter build ios-framework --cocoapods

# Deploy Firestore rules
firebase deploy --only firestore

# Update Firebase config (if project changes)
flutterfire configure --project=continuonai --platforms=web,android,ios --yes
```

## Architecture

### State Management: BLoC Pattern
- `lib/blocs/auth/` - Firebase authentication state
- `lib/blocs/robot/` - Robot discovery and connectivity state
- `lib/blocs/thought/` - Brain thought streaming

### Key Services
- **BrainClient** (`lib/services/brain_client.dart`) - gRPC/WebRTC bridge to ContinuonBrain with TLS + bearer auth. Contains embedded `RCANClient` for robot control.
- **RCANClient** (`lib/services/rcan_client.dart`) - RCAN protocol implementation for discovery, authentication, claim/release control.
- **ScannerService** (`lib/services/scanner_service.dart`) - Platform-agnostic robot discovery (mDNS on native, HTTP probing on web).
- **CloudUploader** (`lib/services/cloud_uploader.dart`) - Signed URL RLDS upload to cloud storage.
- **TaskRecorder** (`lib/services/task_recorder.dart`) - In-memory RLDS episode collection.

### Data Models
- `lib/models/user_role.dart` - 9-level role hierarchy (Creator → Guest) with capability matrix
- `lib/models/teleop_models.dart` - ControlCommand, Vector3, AccelerationProfile
- `lib/models/rlds_models.dart` - RLDS episode schema

### Entry Point
`lib/main.dart` - Firebase init, BLoC setup, route configuration (16 named routes)

## RCAN Protocol

Robot Communication & Addressing Network protocol for robot discovery and control.

```dart
// Claim robot control
final session = await brainClient.claimRobotRcan(
  userId: 'user-uuid',
  role: UserRole.owner,
);

// Send command
await brainClient.sendRcanCommand(
  command: 'teleop',
  parameters: {'action': 'move_forward'},
);

// Release control
await brainClient.releaseRobotRcan();
```

Key endpoints: `/rcan/v1/discover`, `/rcan/v1/status`, `/rcan/v1/auth/claim`, `/rcan/v1/auth/release`, `/rcan/v1/command`

mDNS service type: `_rcan._tcp`

## Ownership Model

- **Local-only claim**: Robot must be on same LAN for initial claim + seed model install
- **Remote allowed after claim + subscription**: Owner can control/OTA/edit remotely if subscription active
- **QR pairing**: Robot UI shows QR + 6-digit code; app scans and confirms via `POST /api/ownership/pair/confirm`

UX states to handle: `Local claim required`, `Initial install pending`, `Remote allowed`, `Blocked` (with reasons: subscription required, ownership mismatch, signature/checksum failed, bundle unavailable)

## Testing Expectations

- Run `flutter analyze` for all code changes
- Run integration test for connect/record flow changes
- Platform builds (`flutter build aar` / `flutter build ios-framework --cocoapods`) optional unless embedding configs change
- Doc-only edits require no tests

## Key Conventions

- Keep UI wiring thin; avoid duplicating business logic in widgets
- Align gRPC/JSON payloads with `proto/continuonbrain_link.proto`
- Keep upload/RLDS metadata consistent with `docs/rlds-schema.md`
- Platform channels are transport/auth only - no business logic in native shims
- Staging/prod endpoints go in README comments, not source defaults

## Public RLDS Episodes

Public listings require complete `share` block: `public`, `slug`, `title`, `license`, `tags`, `content_rating` (general/13+/18+), `intended_audience`, `pii_attested`. Only list when `pii_cleared=true` and `pending_review=false`.

## Related Documentation

- `../docs/rcan-protocol.md` - RCAN protocol specification
- `../docs/rlds-schema.md` - RLDS data contract
- `../docs/bundle_manifest.md` - OTA bundle format
- `../docs/ownership-hierarchy.md` - Role-based access control
- `continuon-cloud/README.md` - Cloud staging docs (GCP TPU setup)
