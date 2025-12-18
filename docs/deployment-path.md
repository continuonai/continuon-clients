# XR to ContinuonOS Deployment Path

This guide narrates the capture-to-deployment pathway: XR capture → RLDS curation → cloud training/export → signed bundle creation → OTA delivery → Memory Plane merge/rollback. It complements the existing RLDS schema and bundle manifest contracts.

## 1. XR capture
- **Sources:** XR headset streams (RGB/depth, gaze, gloves), teleop UI events, and robot state/diagnostics.
- **Contract:** Log using the RLDS schema (`docs/rlds-schema.md`) with per-field `valid` flags and `episode_metadata` describing shell, location, ownership, and consent.
- **Provenance gates:** Require ownership/pairing confirmation and PII/safety flags before upload. Episodes with missing consent remain local-only.

## 2. RLDS curation
- **Filtering:** Drop or mark spans with missing IMU/odometry; preserve diagnostics in `step_metadata` to retain drift/latency context.
- **Labeling:** Add annotations (`action.annotation`, `episode_metadata.tags`) for tasks/goals; mark synthetic/backfilled fields explicitly.
- **Safety review:** Ensure PII redaction status is stored in metadata; only `pii_cleared=true` episodes progress to cloud ingest.

## 3. Cloud training/export
- **Ingest:** Upload curated RLDS to Continuon Cloud ingest. Maintain provenance fields (`source`, `shell_id`, `ownership_token`, `pii_cleared`).
- **Training:** Run HOPE/CMS loops to produce policy + world-model checkpoints; training outputs record dataset hashes and trainer config.
- **Export:** Package the selected checkpoint into an edge bundle following `docs/bundle_manifest.md`. Include RLDS hashes, safety eval summaries, and capability envelopes for the target shell.

## 4. Signed bundle creation
- **Signing:** Sign the bundle manifest + payload with the Continuon signing key. Embed:
  - `bundle_manifest` (model paths, capability matrix, required sensors, safety guardrails)
  - `provenance` (dataset hashes, trainer build, timestamp, signer identity)
  - `safety_gates` (PII attestation, eval thresholds, ownership requirements)
- **Validation:** Reject unsigned or mismatched manifests; ensure manifest capabilities align with the shell’s HAL contract.

## 5. OTA delivery
- **Transport:** Deliver via Vertex Edge or direct OTA. The Continuon Brain runtime validates signature, manifest schema, and capability compatibility before staging.
- **Safety kernel handoff:** The on-device safety kernel loads the bundle only if safety gates match the current shell and ownership token. Otherwise, the bundle remains staged and inactive.

## 6. Memory Plane merge / rollback
- **Merge:** New bundles merge into the Memory Plane with versioned slots. The runtime writes a `staged` slot, runs health checks, then promotes to `active`.
- **Rollback:** If health checks, watchdogs, or operator signals fail, the runtime reverts to the previous `active` slot and logs the failure reason for cloud triage.
- **Hot-swap:** Mid-loop planners can switch between `active` and `staged` policies when permitted by the safety kernel; swaps are logged to RLDS diagnostics.

## End-to-end example
1. XR session recorded with gloves + gaze → RLDS episode stamped with ownership + consent.
2. Curation marks two missing depth segments as `valid=false` and tags `pii_cleared=true` after blur.
3. Cloud training exports `bundle_manifest` referencing the RLDS hash and safety eval metrics.
4. Bundle is signed and pushed OTA; device validates signature and shell capability match.
5. Memory Plane stages the bundle, runs fast-loop watchdog, then promotes. In case of capability mismatch, it remains staged and exposes rollback UI/API.

## Related contracts
- RLDS schema: [`docs/rlds-schema.md`](./rlds-schema.md)
- Bundle manifest: [`docs/bundle_manifest.md`](./bundle_manifest.md)
- HAL streaming contract: [`docs/hal-streaming-contract.md`](./hal-streaming-contract.md)

## Provenance and safety gates
- Maintain chain-of-custody by carrying dataset hashes, signer identity, and eval scores in the bundle manifest.
- Enforce PII/safety gates at each step: capture (consent), curation (redaction), cloud training (eval thresholds), OTA (signature + capability match), and runtime (safety kernel guardrails + watchdogs).
