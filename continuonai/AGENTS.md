# Agent Instructions (ContinuonAI Flutter App + Cloud Docs)

Scope: `continuonai/` (Flutter app hosted on web/iOS/Android/Linux + consolidated Continuon-Cloud docs).

- Keep the Flutter module layout consistent with `continuonai/README.md` (brain client, cloud uploader, task recorder, models, integration tests). Favor thin UI wiring; avoid duplicating business logic in widgets.
- Align gRPC/JSON payloads with `proto/continuonbrain_link.proto`; keep upload/RLDS metadata consistent with `docs/rlds-schema.md` and Cloud contracts moved under `continuonai/continuon-cloud/`.
- Keep OTA/subscription notes in sync with root README and `continuonai/README.md`: OTA bundle delivery is gated on robot ownership + paid subscription and uses the signed bundle contract (`docs/bundle_manifest.md`).
- Document and preserve the local-only registration + initial install rule; remote access/OTA is only after claim/seed install, with ownership + subscription enforced on remote sessions.
- Mirror the UX states from `continuonai/README.md` (local claim required, initial install pending, remote allowed, blocked) and keep error strings explicit (`subscription required`, `ownership mismatch`, `signature/checksum failed`).
- For platform channels, keep them transport/auth only; no business logic in the native shims. Document any staging/prod endpoints in README comments, not source defaults.
- Cloud docs here remain staging for the dedicated Continuon-Cloud repo—keep them minimal, versioned, and secret-free; cross-link root README/`docs/monorepo-structure.md` when flows span products.
- Testing expectations for code changes:
  - Run `flutter analyze`.
  - Run `flutter test integration_test/connect_and_record_test.dart` for flow changes.
  - Platform builds (`flutter build aar` / `flutter build ios-framework --cocoapods`) are optional unless embedding configs change.
  Note skipped commands and why (e.g., SDKs unavailable). Doc-only edits require no tests.

- Upcoming: Continuon AI web now has a public RLDS viewer stub. When the Continuon Cloud public-episodes API is live, wire it with signed URLs and expose only uploads that include a `share` block (public flag, slug, license, tags). Keep README/AGENTS notes in sync when enabling.
- Public sharing safety: public listings must remain gated until content rating + PII scans pass. Require `share` block fields: `public`, `slug`, `title`, `license`, `tags`, `content_rating` (general/13+/18+), `intended_audience`, and `pii_attested`. Only list when `pii_cleared=true` and `pending_review=false`; serve only redacted assets when `pii_redacted=true`.

- Local pairing: prefer QR pairing (robot UI shows QR + 6-digit code; app scans QR + posts `POST /api/ownership/pair/confirm`). Avoid biometric identification.

Context: Conversation on 2025-12-10 about Pi5 startup/training is logged at `../docs/conversation-log.md` (headless Pi5 boot defaults, optional background trainer, tuned Pi5 training config, RLDS origin tagging).