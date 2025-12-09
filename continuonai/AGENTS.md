# Agent Instructions (ContinuonAI Flutter App + Cloud Docs)

Scope: `continuonai/` (Flutter app hosted on web/iOS/Android/Linux + consolidated Continuon-Cloud docs).

- Keep the Flutter module layout consistent with `continuonai/README.md` (brain client, cloud uploader, task recorder, models, integration tests). Favor thin UI wiring; avoid duplicating business logic in widgets.
- Align gRPC/JSON payloads with `proto/continuonbrain_link.proto`; keep upload/RLDS metadata consistent with `docs/rlds-schema.md` and Cloud contracts moved under `continuonai/continuon-cloud/`.
- Keep OTA/subscription notes in sync with root README and `continuonai/README.md`: OTA bundle delivery is gated on robot ownership + paid subscription and uses the signed bundle contract (`docs/bundle_manifest.md`).
- For platform channels, keep them transport/auth only; no business logic in the native shims. Document any staging/prod endpoints in README comments, not source defaults.
- Cloud docs here remain staging for the dedicated Continuon-Cloud repo—keep them minimal, versioned, and secret-free; cross-link root README/`docs/monorepo-structure.md` when flows span products.
- Testing expectations for code changes:
  - Run `flutter analyze`.
  - Run `flutter test integration_test/connect_and_record_test.dart` for flow changes.
  - Platform builds (`flutter build aar` / `flutter build ios-framework --cocoapods`) are optional unless embedding configs change.
  Note skipped commands and why (e.g., SDKs unavailable). Doc-only edits require no tests.
