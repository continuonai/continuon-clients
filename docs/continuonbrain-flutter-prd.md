# PRD: ContinuonBrain Flutter Package (ContinuonAI Host App)

## 1. Problem Statement
ContinuonAI (Flutter app) currently embeds ContinuonBrain connectivity, RLDS capture, and upload helpers directly in-app. For fleet safety and resilience, we need a **private Flutter package** that can be reused across ContinuonAI shells (mobile, desktop, web) to provide:
- Co-existence with (or a light wrapper around) the browser-hosted failover client so we can evaluate when local model access, offline caching, RLDS serialization, and OTA bundle ingestion should stay in the browser versus crossing into Flutter shells.
- An offline-capable fallback brain client when on-device inference or OTA bundles fail.
- A consistent RLDS capture/queue + upload path that can run browser-hosted or embedded in native shells.
- Safety-critical controls (emergency stop, rollback-to-known-good bundle) that remain available even when Google Cloud/Vertex calls degrade.

## 2. Target Users
- **Robot operators and safety reviewers** using ContinuonAI for teleop, emergency stops, and safety violation escalation.
- **Developers/researchers** who need a reusable client to prototype policies and OTA bundles without reimplementing RLDS/logging.
- **Continuon platform engineers** maintaining ContinuonBrain/OS and Continuon Cloud contracts, ensuring the Flutter surfaces stay aligned with `continuonai/README.md` and the RLDS schema in `docs/rlds-schema.md`.

## 3. Goals (v1–v3)
- **v1 – Extraction & Parity:** Extract existing `BrainClient`, `TaskRecorder`, and `CloudUploader` logic from ContinuonAI into a private package with API parity and sample wiring in the host app, while preserving the browser failover client as a first-class peer.
- **v2 – Offline & Safety Hardened:** Add dual-path RLDS handling (local serialization + Python relay), offline detection, retry/backoff queues, and emergency-stop/bundle rollback hooks; document when browser-hosted JS in-webview is preferred over native bindings for offline capture or OTA ingestion to align with the “one shell, many devices” principle.
- **v3 – Cloud-Aligned Expansion:** Plug into Google Cloud defaults (signed URLs via Cloud Storage broker, IAM/service-account auth, optional Pub/Sub alerts for safety flags, future Vertex job triggers) without hard-coding cloud dependencies into the host UI layer, and ensure the browser failover path reuses the same cloud contracts.

### Hybrid Assessment Outcomes
- **Webview-first (JS/browser failover) paths:** Use the browser-hosted client when OTA bundle ingestion, offline caching, or IndexedDB-backed RLDS serialization needs to run without native dependencies, keeping policy evaluation consistent across the “one shell, many devices” fleet.
- **Native-binding paths:** Prefer Flutter/Dart bindings when local model access, file-system-backed RLDS buffering, or native plugin access (camera/sensors) is required; keep interop shims thin so the browser fallback can be swapped in without changing cloud alignment.
- **Cloud contract alignment:** Both execution modes must honor the same Google Cloud Storage signing, provenance manifests, and Pub/Sub alert hooks so browser failover remains a drop-in alternative during outages.

## 4. Non-Goals (for now)
- Replacing the ContinuonBrain/OS runtime (lives in `continuonos`); this package is a client façade only.
- Shipping a public Flutter plugin on pub.dev; distribution remains private within the monorepo.
- Full WebRTC stack ownership in Flutter; native/platform-channel transports remain acceptable for low-latency paths.

## 5. Functional Requirements
1. **Connectivity & Control Facade**
   - Provide gRPC/WebRTC client bindings equivalent to `lib/services/brain_client.dart` with TLS + bearer metadata support.
   - Expose emergency stop and mode switch calls, mirroring the existing `/api/mode/*` semantics referenced in `GEMMA_INTEGRATION.md`.

2. **RLDS Capture & Storage**
   - Package `TaskRecorder`-style APIs to stamp observations/actions per `docs/rlds-schema.md` and tag `episode_metadata.continuon.xr_mode`.
   - Support two modes: (a) local serialization + IndexedDB/SQLite/FileSystem adapter for offline-first capture, and (b) relay-to-Python bridge when the ContinuonBrain API server is present.
   - Queue episodes and auto-upload when connectivity/auth return; ensure safety-violation episodes are prioritized.

3. **Upload & Cloud Alignment**
   - Integrate a signed-URL broker (Cloud Storage) with checksum verification and provenance manifest per `continuon-lifecycle-plan.md`.
   - Optional Pub/Sub notification emitters for emergency-stop events and safety violations; keep them behind feature flags.
   - Leave OTA bundle/model fetch pluggable; provide integrity checks (hash/signature) and a rollback-to-last-good pointer stored locally.

4. **Offline Detection & Retry**
   - Detect network state changes; gate uploads until readiness checklist in `docs/upload-readiness-checklist.md` is satisfied.
   - Implement exponential backoff with jitter for uploads and bundle fetches; surface progress/error callbacks to the host app.

5. **Developer Experience**
   - Provide a minimal sample app in `continuonai/` demonstrating initialization, connect, record, upload, and emergency stop.
   - Publish API docs and migration notes from in-app services to the package façade.

## 6. Non-Functional Requirements
- **Portability:** Must run on Flutter mobile/desktop/web; storage layer chosen via adapters (IndexedDB for web, SQLite/file IO for mobile/desktop).
- **Security:** TLS by default; bearer tokens passed through; signed URL uploads must verify checksums and reject unsigned endpoints.
- **Performance:** RLDS recording must not drop frames at 30 FPS egocentric video + sensor streams; retries must not block UI thread.
- **Resilience:** Package must operate offline for ≥24 hours, preserving RLDS buffers and bundle rollback pointers without data loss.

## 7. Data Contracts & Compliance
- Adhere to RLDS schema in `docs/rlds-schema.md`; keep parity with `proto/rlds_episode.proto` and the ContinuonBrain link proto.
- Preserve provenance metadata and upload gating steps defined in `continuon-lifecycle-plan.md` and `docs/upload-readiness-checklist.md`.
- Store safety-violation markers with timestamps and device IDs for HQ review; default to auto-upload on reconnect.

## 8. Success Metrics
- **Reliability:** 99% success for emergency-stop invocation under offline/packet-loss simulation.
- **Data Retention:** ≥95% of completed offline episodes survive a 24-hour offline window and upload successfully on reconnect.
- **Interop:** 100% API parity with existing ContinuonAI flows (connect, teleop, record, upload) measured by migration test suite.
- **Performance:** <200 ms control command round-trip on Wi-Fi when using gRPC/WebRTC bridge; <5% drop rate on 30 FPS RLDS capture.

## 9. Milestones
- **M1 (Extraction, 1–2 weeks):** Create `packages/continuon_brain_client` with exported facades; wire ContinuonAI host app to consume it without functional regressions; add sample.
- **M2 (Offline/Safety, 3–4 weeks):** Implement offline detection, RLDS queue adapters, emergency stop + rollback hooks, and signed-URL upload retries with manifest verification.
- **M3 (Cloud Alignment, 4–6 weeks):** Add Cloud Storage broker helpers, optional Pub/Sub emitters, and OTA metadata integrity checks; document Vertex/AutoML handoff hooks for future training jobs.
- **M4 (Hardening, 6–8 weeks):** End-to-end soak tests (offline/online churn, safety violation auto-upload), performance profiling, API docs, and migration guide for host apps.

## 10. Risks & Mitigations
- **Offline storage fragmentation across platforms** – Mitigate with a storage adapter interface and conformance tests for web/mobile/desktop backends.
- **Safety feature drift vs ContinuonBrain/OS** – Align APIs with `continuonbrain/` scaffolding and `continuonai/README.md`; add contract tests against mock ContinuonBrain endpoints.
- **Cloud dependency bloat** – Keep GCP SDK usage modular; default to HTTP signed-URL flow to avoid pulling large dependencies into the host app.

## 11. Documentation & Launch
- Maintain package README with API surface, storage adapter guidance, and cloud configuration examples; cross-link `continuonai/README.md` and `docs/monorepo-structure.md` when describing flows spanning ContinuonAI, ContinuonBrain/OS, and Continuon Cloud.
- Add migration notes for removing direct imports of `BrainClient`, `TaskRecorder`, and `CloudUploader` from host code.
- Launch criteria: v2 features complete, migration of ContinuonAI to the package completed, and soak tests meeting success metrics.
