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
   - If requesting public listing, set a `share` block with: `public` (bool), `slug`, `title`, `license`, `tags`, `content_rating` (general/13+/18+), `intended_audience`, `pii_attested=true`. Skip listing until rating + PII checks pass.
4. **Prepare network and credentials**
   - Ensure the device has outbound connectivity to the configured ingest endpoint (default: ContinuonAI/WorldTape portal URL).
   - Install the upload token/API key in the secure keystore or environment file; never bake credentials into images.
5. **Package for ContinuonAI/WorldTape portal**
   - Zip the curated RLDS episodes and include the manifest plus device/environment identifiers.
   - Capture provenance in the manifest: Continuon Brain runtime version, Continuon AI app version, environment/deployment ID, per-episode checksums, and capture timestamps. See the staging ingest sample in `continuonai/continuon-cloud/signed-ingestion.md` for the required fields.
   - Pin a deterministic `package_id` in the manifest for traceability across retries.
6. **Provenance and security gates**
   - Hash the archive (SHA-256) and record the checksum alongside the upload request; include per-episode hashes in the manifest.
   - Sign the archive plus manifest digest with the environment’s signing key before initiating the upload; enforce the signing step in the client so unsigned payloads never leave the device.
   - Verify TLS is enforced end-to-end; reject if certificate validation fails.
   - Run automated safety/PII scans prior to public listing: face/license-plate detection + blur, OCR for IDs/text, audio ASR + PII/toxicity filter, metadata profanity/PII check. Mark `pii_cleared=true` only after redaction or manual review; block public listing if `pii_cleared=false`.
7. **Send and verify**
   - Perform a dry-run (HEAD or small sample) if bandwidth is constrained, then send the full package.
   - Confirm server receipt, compare the returned checksum, and check for signature/manifest validation errors; log the result locally and keep the manifest alongside the audit record.
8. **Post-upload hygiene**
   - Keep the local copy until cloud acknowledgment succeeds; mark episodes as exported only after verification.
   - Rotate tokens and audit logs periodically; revoke credentials on operator request.
