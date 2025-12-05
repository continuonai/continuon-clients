# Continuon-Cloud (placeholder)

Cloud ingest/train/packaging should live in a dedicated repo. This `continuonai/continuon-cloud/` folder is a staging area for specs only (Google Cloud-only backend assumptions); do not add production pipelines here. Refer to `docs/monorepo-structure.md` for the decoupled architecture and keep RLDS/schema updates coordinated via `proto/`.

## Staging ingest expectations

- All uploads into the staging bucket must be signed client-side and include a manifest with provenance metadata (Continuon Brain runtime + Continuon AI app versions), environment ID, deterministic package ID, and SHA-256 checksums for the archive and every episode blob.
- The ingestion service rejects unsigned, unverifiable, or checksum-mismatched artifacts before they ever reach the staging bucket and surfaces 4xx errors to clients; keep local copies until a signed re-upload succeeds.
- See [signed-ingestion.md](./signed-ingestion.md) for the required manifest fields and the verification steps enforced before promotion to training storage.
