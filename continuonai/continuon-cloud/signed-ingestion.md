# Signed ingestion expectations (staging)

This note describes the staging-only Continuon Cloud ingest contract for the XR/WorldTape data path. It documents the minimum signing and provenance requirements for uploads and the verification hooks that run in the staging bucket pipeline. Production implementations still belong in the dedicated Continuon-Cloud repo.

## Upload contract

- **Signed uploads are mandatory.** Every submission must include a detached signature over the payload archive or a bundled signature inside the manifest. Operators should rotate signing keys per environment (field test vs. lab) and track key IDs in the manifest.
- **Provenance metadata accompanies each submission.** Upload manifests must include:
  - Software versions for the Continuon Brain runtime and the Continuon AI app shell that produced the episodes.
  - Environment or deployment identifier (lab rig ID, field device tag, or staging fleet name).
  - Content checksums (SHA-256) for the archive and any nested episode blobs.
  - Timestamps for capture window and packaging time.
- **Transport is TLS-only.** Clients must present their signing key ID and checksum in headers; unsigned or mismatched uploads are rejected before storage.

### Example manifest (minimal)

```json
{
  "package_id": "xr-run-2024-06-01T12:00Z",
  "environment_id": "staging-rig-07",
  "continuon_brain_runtime": "0.10.3",
  "continuon_ai_app": "1.4.0",
  "capture_window_utc": {
    "start": "2024-06-01T11:12:00Z",
    "end": "2024-06-01T11:47:00Z"
  },
  "checksums": {
    "archive_sha256": "<hex>",
    "episodes": [
      { "path": "episodes/0001.jsonl", "sha256": "<hex>" },
      { "path": "episodes/0002.jsonl", "sha256": "<hex>" }
    ]
  },
  "signing": {
    "key_id": "staging-uploader-ed25519-02",
    "signature": "<base64-detached>"
  }
}
```

## Staging bucket verification

Uploads land in a staging bucket with a hook that runs before promotion to training storage:

1. Validate that a signature is present and that the advertised signing key is allowed for the declared environment ID.
2. Recompute archive checksum and cross-check against the manifest; fail on any mismatch.
3. Verify the signature against the archive hash and manifest digest.
4. Ensure required provenance fields (runtime/app versions, environment ID, timestamps) are populated.
5. Emit structured errors (HTTP 400/409) back to the client on failure and tag the object with a rejection reason to block downstream trainers.

Clients should treat any 4xx response as a hard failure and leave the local copy intact until a signed, checksum-matched re-upload succeeds.
