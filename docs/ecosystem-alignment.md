# Ecosystem Alignment and Data Source Coverage

This memo restates how this repo maps onto the broader Continuon architecture and calls out additional lower-fidelity data sources that can be normalized into RLDS episodes to keep the HOPE/Nested Learning stack well fed.

## Where the repo already aligns
- **Monorepo stance:** ContinuonXR stays focused on XR data capture and UI while referencing continuonos (runtime), Continuon-Cloud (ingest/train), ContinuonAI (contracts), and worldtapeai.com (browser/annotation).【F:docs/monorepo-structure.md†L6-L24】 This respects the “One Brain, Many Shells” separation of shells versus the shared brain.
- **RLDS-first contracts:** The RLDS schema covers synchronized poses, gaze, video/depth, audio, glove telemetry, robot state, UI context, and diagnostics with explicit validity flags so sensors can drop without breaking the format.【F:docs/rlds-schema.md†L6-L74】 Human-centric modes already map to RLDS tags for trainer, workstation, observer, and YouTube/TV ingestion.【F:docs/human-centric-data.md†L6-L35】
- **HOPE/CMS routing:** Existing guidance ties RLDS blocks to Fast/Mid/Slow loops and the VisionCore/LanguagePlanner/World Model/SkillPolicies/SafetyHead heads, ensuring XR signals arrive at the right timescale and optimizer.【F:docs/hope-cms-vla.md†L1-L85】

## Gaps to close
- **Storytelling gap:** Repo docs focus on capture and schema but don’t narrate the end-to-end bridge from XR episodes to Cloud training, signed edge bundles, and continuonos hot-swap/rollback.
- **Data quality tiers:** The current schema hints at optional fields and hallucination/backfill, but it lacks concrete pathways for ingesting weaker data sources (front-facing web video, app-based trainers) and grading them before they reach HOPE.

## Lower-fidelity data sources and how to fit them to RLDS
- **Raw web video (e.g., YouTube without RLDS):** Treat each clip as an offline “observer” episode (`xr_mode = "youtube_tv"`) and attach derived timestamps plus synthetic `frame_id` for consistency. Use vision-derived pose estimators (3D keypoints, depth-from-video), ASR for `audio`, and UI/scene labels in `step_metadata` to keep the schema intact even when real-time sensors are absent. Mark inferred blocks via `valid` flags and provenance tags so the Slow loop can down-weight or filter during training.【F:docs/rlds-schema.md†L16-L74】【F:docs/human-centric-data.md†L24-L35】
- **Third-party telemetry or logs (non-RLDS):** Normalize robot/IoT logs into `robot_state` and `action.command` where possible; map event markers into `step_metadata`. If vision is missing, keep `egocentric_video` empty but preserve timing to allow later NeRF/backfill synthesis as described in human-centric data notes.【F:docs/rlds-schema.md†L30-L64】【F:docs/human-centric-data.md†L36-L47】
- **ContinuonAI.com trainer app episodes:** For users without XR, accept desktop/mobile recordings as Mode B-style sessions (`xr_mode = "workstation"`) with `ui_context` and `ui_action` populated from the app’s UI event stream. Pair screen/video capture with optional microphone input and any available pointer/gesture traces as lightweight substitutes for XR hand poses. Align these into steps with monotonic timestamps so Mid/Slow LanguagePlanner and VisionCore training can still learn workflow patterns and tool sequencing.【F:docs/rlds-schema.md†L47-L74】【F:docs/human-centric-data.md†L10-L35】
- **Crowdsourced annotations:** Allow post-hoc polygon/mask labels or text tags to enter via `action.annotation`, enabling observer-mode supervision even when controls were not recorded. Keep these aligned with `frame_id` so they can be fused with synthesized depth or glove signals later.【F:docs/rlds-schema.md†L52-L74】【F:docs/human-centric-data.md†L18-L35】

## Operational safeguards for noisy inputs
- **Quality tags and scoring:** Encourage ingestion pipelines to set `step_metadata` and `episode_metadata.tags` for source type (XR vs YouTube vs app) and confidence tiers so Cloud trainers can stratify batches and SafetyHead retraining can ignore low-trust segments.【F:docs/rlds-schema.md†L6-L24】【F:docs/hope-cms-vla.md†L63-L85】
- **Backfill with provenance:** When hallucinating tactile/audio or estimating pose for missing sensors, persist the original `valid` flags and add provenance labels rather than overwriting fields. This keeps HOPE’s Slow loop aware of uncertainty and allows down-weighting or targeted augmentations.【F:docs/rlds-schema.md†L22-L74】【F:docs/human-centric-data.md†L32-L47】
- **Latency and drift tracking:** Carry `diagnostics` (latency, drop counts, BLE RSSI) into converted episodes so Fast/Mid loops can compensate and Slow loop training can model reliability per source.【F:docs/rlds-schema.md†L61-L74】【F:docs/hope-cms-vla.md†L70-L85】

## Next documentation steps
- Add a deployment path doc that narrates XR → Cloud → signed edge bundle → continuonos hot-swap/rollback, with clear metadata handoffs. This closes the storytelling gap between capture and deployment.
- Provide worked examples of converting YouTube clips and app-based sessions into RLDS episode folders (metadata.json + steps/*.jsonl) to standardize ingestion recipes.

## Related Documentation

The storytelling gap identified in the "Gaps to close" section is now addressed by the comprehensive system architecture document:
- **[System Architecture and Training Lifecycle](./system-architecture.md)** - Covers the complete end-to-end flow: XR capture → WorldTapeAI ingestion → Vertex AI cloud training → signed edge bundle → OTA delivery → Memory Plane merge at boot
