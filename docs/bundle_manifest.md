# Edge Bundle Manifest & OTA Checklist (subscription-gated)

This note captures the required fields for the signed edge bundle manifest and the minimal OTA checklist used when delivering models to registered, paid robots via the ContinuonAI Flutter app.

## Manifest requirements (`edge_manifest.json`)
- `version`: semantic version of the bundle (e.g., `1.0.0`).
- `model_name`: human-readable name of the policy/skill.
- `artifacts`:
  - `cpu_bundle`: path to the CPU/JAX export.
  - `hailo_bundle`: path to the Hailo ONNX/HEF export (placeholder HEF allowed when compiler absent).
  - `safety_manifest`: path to the safety rules/version file.
- `checksums`: SHA-256 for each artifact listed above.
- `signature`: detached signature over the manifest and artifacts (signed with cloud key).
- `created_at_unix_ms`: creation time for audit/rollback.
- `source`: producer identifier (e.g., `continuon.brain_runtime` or `continuon.cloud`).

## Precision / quantization notes (non-normative)
This repo currently uses **two related manifests**:
- **OTA edge bundle manifest** (`edge_manifest.json`, this doc): integrity + artifact routing for deployments.
- **On-device runtime manifest** (e.g., `continuonbrain/model/manifest.pi5.example.json`): how the local runtime loads base model + adapters.

To stay backward-compatible, prefer **documenting quantization in the runtime manifest** (`runtime.quantization`) and keep the OTA manifest focused on signed artifacts + checksums.

For Nested Learning / HOPE-style systems with **mutable fast state**, treat that state as:
- **Local runtime state** (Memory Plane) rather than an OTA-delivered artifact, unless explicitly packaged and signed.
- **Telemetry/RLDS metadata**: log precision/format and any fast-state resets/promotions in RLDS `step_metadata` for auditability.

## OTA checklist (subscription-gated)
1. **Ownership + subscription**: Robot is registered in the ContinuonAI app and marked paid/active before OTA is offered.
2. **Bundle integrity**: All artifacts are checksummed; manifest is signed. Reject OTA if signature or checksums fail.
3. **Safety manifest present**: Safety rules shipped alongside the model; enforce merging with immutable base rules on device.
4. **Fallback policy**: Ensure a last-known-good bundle remains on device; OTA applies only after download+verify passes.
5. **Versioning**: Increment `version`; never reuse signatures across different artifact sets.
6. **Provenance**: Embed model/source/version metadata; keep validation reports from the export step adjacent to the bundle.
7. **Apply gate**: OTA apply is blocked unless steps 1–4 succeed; subscription/ownership gate is enforced in the Flutter app.

## Minimal manifest example
```json
{
  "version": "1.0.0",
  "model_name": "core-seed-pi5",
  "artifacts": {
    "cpu_bundle": "models/core_model_inference/",
    "hailo_bundle": "models/core_model_hailo/",
    "safety_manifest": "models/safety/manifest.json"
  },
  "checksums": {
    "cpu_bundle_sha256": "<hex>",
    "hailo_bundle_sha256": "<hex>",
    "safety_manifest_sha256": "<hex>"
  },
  "signature": "<base64-detached>",
  "created_at_unix_ms": 1733790000000,
  "source": "continuon.cloud"
}
```

## Packaging flow (reference)
1. Export CPU + Hailo bundles (or placeholder HEF).
2. Generate validation reports (schema, alignment) and place them next to the bundle.
3. Write `edge_manifest.json`, compute checksums, sign manifest+artifacts.
4. Upload bundle + manifest + reports to the OTA bucket path for the registered robot(s).
5. The ContinuonAI app gates download/apply on ownership + paid status; device verifies signature/checksums before swap.

## Where the gate lives
- Subscription/ownership gate: ContinuonAI Flutter app (`continuonai/README.md`).
- OTA manifest contract: this doc.
- Seed model source: `docs/seed-model-plan.md`.

