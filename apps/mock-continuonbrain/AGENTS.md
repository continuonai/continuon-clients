# Agent Instructions (Mock ContinuonBrain Service)

Scope: `apps/mock-continuonbrain/`.

- This directory stays as a lightweight mock for XR-side contract testing. Do not add production runtimes; keep implementations minimal and telemetry-focused.
- Favor TypeScript with strict types and generated stubs from `proto/continuonbrain_link.proto`; avoid hand-written protobuf definitions.
- Keep latency/jitter knobs and canned trajectories easy to toggle for XR resilience testing; document defaults in README comments when adding flags.
- Testing/build expectations:
  - Run `npm run build` after code or proto changes (includes `generate:proto`).
  - Run `npm run lint:schema` if schema/proto contracts are updated.
  Note environment blockers (e.g., missing Node) in the summary when skipping commands.
