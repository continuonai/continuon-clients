# Agent Instructions (Flutter Companion)

Scope: `apps/flutter-companion/`.

- Keep the module entry points and service layout consistent with `README.md` (brain client, cloud uploader, task recorder, models, integration tests). Prefer thin UI wiring over duplicating business logic in widgets.
- Use platform channels only for transport/auth bridging; keep gRPC/JSON shapes aligned with `proto/continuonbrain_link.proto`.
- Apply Dart style via `dart format .` before committing; honor analyzer hints from `analysis_options.yaml`.
- When touching RLDS upload/recording code, document required metadata (`episode.json`, `steps/*.jsonl`) and keep broker endpoints configurable.
- Testing expectations:
  - Run `flutter analyze` for changes to Dart code.
  - Run `flutter test integration_test/connect_and_record_test.dart` for flow changes.
  - Build commands (`flutter build aar` / `flutter build ios-framework --cocoapods`) are optional unless you modify embedding configs.
  Note any skipped commands with the reason (e.g., mobile SDKs unavailable).
