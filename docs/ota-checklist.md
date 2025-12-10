# OTA Readiness Checklist (subscription-gated)

Single-stop reference tying together OTA requirements across modules.

- **Manifest contract**: Use `docs/bundle_manifest.md` (artifacts, checksums, signature, safety manifest).
- **Seed source**: Initial bundle is the v0 seed model from Pi 5 + AI HAT (`docs/seed-model-plan.md`); future bundles follow the same contract.
- **Gate**: ContinuonAI Flutter app enforces robot ownership + paid subscription before offering OTA; device verifies signature/checksums and keeps a fallback bundle.
- **Safety**: Ship safety manifest with immutable base rules; merge on-device and keep last-known-good.
- **Path hygiene**: Upload manifest + validation reports + artifacts together; delete/rotate temporary scripts (e.g., startup scripts) from buckets after use.

