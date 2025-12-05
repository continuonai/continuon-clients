# Upload Readiness Checklist (Manual/Opt-In)

Use this checklist whenever enabling uploads from field hardware. It keeps offline defaults intact while making sure curated data reaches the ContinuonAI/WorldTape portal with provenance and consent.

1. **Default to offline logging**
   - Confirm RLDS episodes are stored locally and upload cron/daemon is disabled by default.
   - Verify disk space and rotation policy for local episodes.
2. **Confirm operator intent**
   - Present an explicit opt-in toggle in the client (XR or Flutter companion) before any upload is attempted.
   - Capture who authorized the upload (user ID or device tag) in the upload manifest.
3. **Curate the payload**
   - Run the local curation filter: drop unsafe/redacted frames, trim to approved segments, and attach episode-level quality flags.
   - Generate a manifest containing episode count, duration, and curation outcomes.
4. **Prepare network and credentials**
   - Ensure the device has outbound connectivity to the configured ingest endpoint (default: ContinuonAI/WorldTape portal URL).
   - Install the upload token/API key in the secure keystore or environment file; never bake credentials into images.
5. **Package for ContinuonAI/WorldTape portal**
   - Zip the curated RLDS episodes and include the manifest plus device/environment identifiers.
   - Capture provenance in the manifest: Continuon Brain runtime version, Continuon AI app version, environment/deployment ID, and per-episode checksums. See the staging ingest sample in `continuonai/continuon-cloud/signed-ingestion.md`.
6. **Provenance and security gates**
   - Hash the archive (SHA-256) and record the checksum alongside the upload request.
   - Sign the archive or request with the environment’s signing key; unsigned uploads are rejected by the staging bucket hook.
   - Verify TLS is enforced end-to-end; reject if certificate validation fails.
7. **Send and verify**
   - Perform a dry-run (HEAD or small sample) if bandwidth is constrained, then send the full package.
   - Confirm server receipt, compare the returned checksum, and check for signature/manifest validation errors; log the result locally.
8. **Post-upload hygiene**
   - Keep the local copy until cloud acknowledgment succeeds; mark episodes as exported only after verification.
   - Rotate tokens and audit logs periodically; revoke credentials on operator request.
