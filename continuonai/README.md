# ContinuonAI (placeholder)

Organizational specs/orchestration live in the dedicated ContinuonAI repo. This monorepo holds ContinuonXR and shared contracts; master specs (episode schema, cloud API, XR app spec) should remain versioned there and be referenced here.

## Flutter control companion plan (Pi 5 + iPhone)
- **Scope**: outline how the Flutter companion (iOS + Pi 5 build) talks to the ContinuonBrain/OS bridge for teleop, depth preview, and adapter hot-reload. Runtime code stays in the dedicated app module; this is the coordination spec.
- **Contracts**: use `proto/continuonbrain_link.proto` for command/telemetry; map depth frames + drivetrain feedback into RLDS fields per `docs/rlds-schema.md` and the Pi checklist in `continuonbrain/PI5_CAR_READINESS.md`.
- **On-Pi (Flutter runner)**: embed `flutter_gemma` to load `continuonbrain/model/manifest.pi5.example.json` (or safety variant). Listen for adapter promotions from the sidecar trainer and hot-reload the manifest; expose a WebRTC/gRPC endpoint that mirrors the Robot API for local UI and remote iPhone control.
- **iPhone UX**: minimal panels for depth preview, steering/throttle sliders (bounded to PCA9685 ranges), and an upload toggle. Buffer RLDS episodes locally with `xr_mode="trainer"` + `action.source="human_teleop_xr"` and send over the Robot API when connected.
- **Offline-first**: default to local logging only; uploads require explicit opt-in and should follow the ingestion gates in `continuon-lifecycle-plan.md`. Keep mode/state/version tags in `episode_metadata` so Pi and phone builds stay debuggable.
