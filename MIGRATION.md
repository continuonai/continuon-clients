# ContinuonXR Migration Plan

This document tracks directories/files that need to move to other repositories as part of the ContinuonAI org reorganization.

**Goal:** ContinuonXR becomes the XR client app only (Android XR / Kotlin).

| Source | Destination | Status |
|--------|-------------|--------|
| `continuonbrain/` | continuonos | pending |
| `brain_b/` | continuonos | pending |
| `continuonai-backend/` | continuon-cloud | pending |
| `continuonai-web/` | continuon-web | pending |
| `trainer_ui/` | continuon-web | pending |
| `proto/` | continuon-proto | pending |
| `rlds/` | continuon-proto (schemas) + continuonos (implementation) | pending |
| `models/` | continuon-cloud/packaging | pending |
| `HOPE_*.md` | continuonos/docs | pending |
| `conductor/` | evaluate (orchestration docs) | pending |
| `edge_manifest.json` | continuon-proto | pending |
| `packaging/` | continuon-cloud | pending |

## Notes

- Phase 1 (continuon-proto) may already cover `proto/` — verify before migrating.
- Phase 3 will extract brain code into ContinuonOS.
- Do not delete these directories until migration is confirmed in the destination repo.
