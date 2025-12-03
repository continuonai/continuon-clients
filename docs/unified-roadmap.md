# Unified Roadmap & Milestones (Source of Truth)

This roadmap is the shared reference for ContinuonXR, ContinuonBrain/OS, and Continuon Cloud (docs/staging under `continuonai/continuon-cloud/` for the Google Cloud ingest/train/package path). It aligns the phased plan in the repository README, the MVP scope and KPIs in `PRD.md`, and the Pi 5 lifecycle sequencing in `continuon-lifecycle-plan.md`. All schedule, ownership, and milestone updates should be made here first and linked elsewhere.

## Current Phase
- **Phase:** **Phase 1 – MVP Data Capture (Current)**
- **Target window:** 2025-02-28 (mirrors Pi 5 Week 3 milestone)
- **KPI anchors (from PRD):** 95% valid Mode A RLDS episodes, 100 Hz glove streaming, and stable gRPC/WebRTC link between XR and ContinuonBrain/OS.

## Phases 0–4 Summary
The table below is the canonical mapping for phase status, owners, and target dates across modules. README references to Phases 0–4 should link to this section.

| Phase | Goal | XR Owner / Target | ContinuonBrain/OS Owner / Target | Continuon Cloud Owner / Target | Status |
|------|------|-------------------|----------------------------------|-------------------------------|--------|
| **Phase 0** | Contracts & architecture baselines (RLDS, XR app contract, Robot API) | @continuonxr-team — 2025-01-17 | @continuonbrain-team — 2025-01-17 | @continuoncloud-team — 2025-01-17 | ✅ Complete |
| **Phase 1** | MVP data capture to mock ContinuonBrain/OS with manual RLDS upload | @continuonxr-team — 2025-02-28 | @continuonbrain-team — 2025-02-21 | @continuoncloud-team — 2025-03-07 | 🚧 In progress (current) |
| **Phase 2** | Cloud ingest + automated upload pipeline + first edge bundle | @continuonxr-team — 2025-03-21 | @continuonbrain-team — 2025-03-21 | @continuoncloud-team — 2025-04-04 | ⏳ Planned |
| **Phase 3** | Closed-loop learning with OTA + telemetry feedback | @continuonxr-team — 2025-04-25 | @continuonbrain-team — 2025-04-25 | @continuoncloud-team — 2025-05-02 | ⏳ Planned |
| **Phase 4** | Production scale dashboards, full VLA stack, and fleet tooling | @continuonxr-team — 2025-06-06 | @continuonbrain-team — 2025-06-06 | @continuoncloud-team — 2025-06-20 | ⏳ Planned |

## MVP Scope and KPIs (Phase 1)
This section ties the MVP requirements and success metrics from `PRD.md` to the operational plan.

- **Scope:** Jetpack XR MVP with basic panels + teleop to mock ContinuonBrain/OS, local RLDS episode writer, and manual upload path.
- **KPIs:**
  - 95% of Mode A sessions logged as valid RLDS episodes.
  - 100 Hz reliable glove BLE telemetry.
  - 100% successful gRPC/WebRTC connectivity between XR and ContinuonBrain/OS mock.
- **Owner alignment:** XR delivers capture + manual upload; Brain maintains mock runtime and RLDS schema compatibility; Cloud keeps ingest ready for opt-in uploads.

## Pi 5 Lifecycle Alignment and Weekly Milestones
The Pi 5 lifecycle plan is the reference for Week-by-Week execution. Updates to weekly targets should be mirrored here and in `continuon-lifecycle-plan.md`.

| Week | Focus | XR Owner / Target | ContinuonBrain/OS Owner / Target | Continuon Cloud Owner / Target | Notes |
|------|-------|-------------------|----------------------------------|-------------------------------|-------|
| **Week 1** | Pi configs, HAL stubs, proto regeneration; XR points to Pi mock | XR hooks to Pi mock + SceneCore deps stubbed — 2025-02-07 | Pi HAL/PWM + camera reader stubs — 2025-02-07 | Ingest endpoint skeleton — 2025-02-07 | Mirrors lifecycle "Week 1" checklist |
| **Week 2** | Vehicle loop logging RLDS locally; first opt-in uploads | XR validates local RLDS writer — 2025-02-14 | Pi loop logging camera/IMU/commands — 2025-02-14 | Minimal ingest receives zipped RLDS — 2025-02-14 | Offline-first, opt-in uploads only |
| **Week 3** | XR teleop to Pi over gRPC/WebRTC; start collecting gold data | XR teleop to Pi, glove integration @ 100 Hz — 2025-02-21 | ContinuonBrain/OS runs gRPC/WebRTC bridge — 2025-02-21 | Provenance tagging + checksum in ingest — 2025-02-21 | Current target window |
| **Week 3a** | Hybrid browser/Flutter architecture + packaging recommendations (GCP-aligned) | XR + Flutter owners deliver design doc + sample integration notes for ContinuonAI — 2025-02-24 | ContinuonBrain/OS enumerates OTA bundle serving options and cache/offline detection hooks — 2025-02-24 | Continuon Cloud maps ingest queues to RLDS queueing semantics for both browser and Flutter clients — 2025-02-24 | Artifacts: architecture recommendations doc, OTA bundle serving options, ContinuonAI integration notes tied to hybrid browser/Flutter flows (offline detection/cache sync, RLDS queueing, OTA paths) |
| **Week 4** | Cloud retrains and ships first Edge Bundle; close the loop | XR OTA UI hooks for bundle swap — 2025-02-28 | Edge bundle loader + rollback — 2025-02-28 | Training loop + bundle packaging — 2025-02-28 | Optional stretch goal for Phase 2 handoff |

## Linkage Rules
- README Phase sections, PRD MVP scope/KPIs, and `continuon-lifecycle-plan.md` weekly milestones must link back to this document as the source of truth.
- Offline-first ingest and upload safety gates remain governed by `continuon-lifecycle-plan.md` and `docs/upload-readiness-checklist.md`; roadmap items that involve uploads must honor those controls.
- The Week 3a architecture + packaging deliverable must surface two artifacts: (1) a design document describing hybrid browser/Flutter alignment to Google Cloud packaging (offline detection/cache sync, RLDS queueing semantics, OTA bundle serving modes) and (2) sample ContinuonAI integration notes that show how the browser and Flutter clients queue RLDS consistently and fetch OTA bundles from the same packaging strategy.
