# Agent Instructions (ContinuonXR Android App)

Scope: `apps/continuonxr/`.

- Keep the module layout in sync with `apps/continuonxr/README.md` (config, connectivity, glove, audio, teleop, logging, ui). New features should slot into these domains rather than creating parallel patterns.
- Favor Kotlin coroutines/Flows for async work; avoid blocking XR/SceneCore threads. Prefer extension functions over utility singletons for domain helpers.
- RLDS logging and teleop code should preserve schema tags (`xr_mode`, `action.source`, provenance metadata) consistent with the PRD; document any schema touchpoints in code comments.
- When updating gRPC/WebRTC bridge logic, keep it contract-aligned with `proto/continuonbrain_link.proto` and note any temporary stubs.
- Compose/XR UI code should prioritize small, previewable composables with clear state hoisting; avoid embedding business logic directly in UI nodes.
- Testing/build expectations:
  - Run `./gradlew :apps:continuonxr:assembleDebug` after build.gradle or dependency changes.
  - Run `./gradlew :apps:continuonxr:testDebugUnitTest` for logic changes.
  - Run `./gradlew :apps:continuonxr:generateDebugProto` if proto contracts referenced by the app change.
  If commands are skipped due to environment limits (e.g., Android SDK missing), call that out in the summary.
