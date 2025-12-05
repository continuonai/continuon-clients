# Studio Slow-loop Artifact Cache Plan

This note outlines how the Continuon Brain Studio server caches Slow-loop artifacts on-device and when it revalidates against Continuon Cloud. The intent is to keep critical panels usable while offline without leaking telemetry that should stay transient.

## Artifacts and default handling

| Artifact | Local cache policy | Online refresh strategy | TTL | Privacy constraints |
| --- | --- | --- | --- | --- |
| **Policy bundle IDs** (Slow → Mid/Fast lineage and active bundle fingerprints) | Persisted in the Studio cache directory so the Skill Composer and Safety Head panels can enumerate installed bundles offline. | Re-fetched from Continuon Cloud when the cache is missing or older than the TTL. | 24 hours | Store only bundle identifiers, lineage metadata, and signatures—no training traces or prompts. |
| **Memory Plane merge state** (which Memory Planes are merged, last merge time, conflict notes) | Cached locally to drive the Memory Plane panel when the device is disconnected. | Pulled from Cloud when the cache is missing or stale; merge diffs are merged locally so the panel can reconcile once back online. | 6 hours | Strip any embedded spans or user utterance payloads before persisting. |
| **Drift alerts** (Slow-loop drift summaries sent back to devices) | Cached locally with aggressive redaction so Safety Head and Diagnostics panels can show the last known alerts during outages. | Re-polled from Cloud once per TTL; failed fetches fall back to cached alerts even if stale. | 1 hour | Never write raw traces, feature vectors, or frame snippets—only keep the alert IDs, severity, short summaries, and timestamps. |

## Cache directories and file layout

- Cache files live under the Studio server config directory: `cache/slow_loop.json` for the aggregated snapshot.
- Each artifact is stored with `cached_at`, an `expires_at` derived from the TTL, and a list of redactions applied before persistence.
- Responses returned by the Studio server include a `freshness` block per artifact:
  - `source`: `cache` or `cloud`
  - `cached_at` and `expires_at` timestamps (UTC)
  - `stale`: whether the TTL has elapsed
  - `privacy_redactions`: which fields were removed before writing to disk

## Offline and reconciliation flows

1. **Offline use:** When Continuon Cloud access is disabled or unavailable, the Studio server serves cached artifacts even if they are marked stale, while preserving freshness metadata so panels can warn operators.
2. **Revalidation:** Once connectivity returns and cloud fetches are allowed, the server refreshes any stale or missing artifacts, updates the cache file, and responds with `source: cloud` and `stale: false` for the refreshed items.
3. **Partial failures:** If the cloud fetch partially fails (e.g., drift alerts timeout), the server keeps prior cached values for that artifact and marks them stale; other artifacts that refreshed successfully still return as fresh.
4. **Privacy:** Redaction happens before persistence; only the already-redacted payloads are written to disk. Panels receive the redacted payloads along with the `privacy_redactions` list to make it clear what was omitted.

## TTL and redaction rationale

- **Policy bundles (24h):** Bundle fingerprints rarely change more than once a day; a 24-hour TTL avoids unnecessary network calls while keeping SKU/lineage displays current.
- **Memory Plane merge state (6h):** Merge activity is more frequent during tuning, so we revalidate twice a day to keep merge resolutions and conflict notes current without churning the network.
- **Drift alerts (1h):** Alerts can escalate quickly; a 1-hour TTL keeps diagnostics reasonably up to date while still allowing offline safety awareness.

## Integration checklist for Studio server

- Load cached artifacts on startup so initial panel renders do not depend on the network.
- Annotate every Slow-loop API response with the per-artifact `freshness` block.
- Log when the cache is served while offline so operators can inspect how stale the data was.
- Keep redaction lists alongside the code to ensure future fields are handled deliberately.
