# Continuon Monorepo Strategy (One Brain, Many Shells)

This repo remains the **ContinuonXR** product, but it now documents how it fits into a decoupled, multi-repo architecture for the `@continuonai` org. Each repo keeps strict boundaries and versioned contracts.

## Core repos
- **continuonos** (ContinuonBrain/OS runtime): platform-neutral core (`src/core/`), HAL adapters (`platform/`), backends (`backends/`), configs, and `rlds_logger/`.
- **ContinuonXR** (this repo): spatial UI + data capture rig; XR shell, glove BLE, RLDS logging, upload.
- **Continuon-Cloud**: ingest/train/package; owns bundle manifest and definitive RLDS schema.
- **ContinuonAI**: business/orchestration; master specs (`episode-schema.md`, `cloud-api-spec.md`, `xr-app-spec.md`).
- **worldtapeai.com**: RLDS browser/annotation front-end.

## Contracts
- **RLDS schema**: versioned; producers (ContinuonXR, continuonos) and consumers (Cloud, worldtapeai) must track MAJOR.MINOR.PATCH. Align `proto/rlds_episode.proto` here with the canonical spec in `ContinuonAI/specs`.
- **ContinuonBrain bridge**: gRPC/WebRTC API; defined in `proto/continuonbrain_link.proto` here and mirrored in `continuonos`.
- **Edge bundles**: manifest owned by Continuon-Cloud; consumed by continuonos.

## Repo boundaries for this project
- Keep XR app code, glove BLE, XR inputs, RLDS logging, and upload client in this repo.
- Do not embed continuonos core or Cloud training code; integrate via APIs/protos only.
- Place shared docs/specs here only when XR-specific; reference canonical specs in `ContinuonAI/specs` for org-wide contracts.
