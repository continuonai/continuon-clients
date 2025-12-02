# Continuon Monorepo Strategy (One Brain, Many Shells)

This repo remains the **ContinuonXR** product, but it now documents how it fits into a decoupled, multi-repo architecture for the `@continuonai` org. Each repo keeps strict boundaries and versioned contracts.

## Core repos
- **continuonos** (ContinuonBrain/OS runtime): platform-neutral core (`src/core/`), HAL adapters (`platform/`), backends (`backends/`), configs, and `rlds_logger/`.
- **ContinuonXR** (this repo): spatial UI + data capture rig; XR shell, glove BLE, RLDS logging, upload.
- **ContinuonAI**: business/orchestration; master specs (`episode-schema.md`, `cloud-api-spec.md`, `xr-app-spec.md`). The Flutter companion app now lives under `continuonai/` in this repo.
- **Continuon-Cloud**: ingest/train/package; owns bundle manifest and definitive RLDS schema. Staging docs are consolidated under `continuonai/continuon-cloud/` here.
- **worldtapeai.com**: RLDS browser/annotation front-end.

## Contracts
- **RLDS schema**: versioned; producers (ContinuonXR, continuonos) and consumers (Cloud, worldtapeai) must track MAJOR.MINOR.PATCH. Align `proto/rlds_episode.proto` here with the canonical spec in `ContinuonAI/specs`.
- **ContinuonBrain bridge**: gRPC/WebRTC API; defined in `proto/continuonbrain_link.proto` here and mirrored in `continuonos`.
- **Edge bundles**: manifest owned by Continuon-Cloud; consumed by continuonos.

## Repo boundaries for this project
- Keep XR app code, glove BLE, XR inputs, RLDS logging, and upload client in this repo.
- Flutter companion/ContinuonAI app ships from `continuonai/` (multi-platform web/iOS/Android/Linux). Treat it as its own product slice while keeping contracts aligned with `proto/`.
- Cloud ingestion/training code still belongs in the dedicated Continuon-Cloud repo; the docs under `continuonai/continuon-cloud/` are staging-only. Do not land production pipelines here.
- Do not embed continuonos core; integrate via APIs/protos only. Reference canonical specs in `ContinuonAI/specs` for org-wide contracts.
