# Agent Instructions (worldtapeai.com placeholder)

Scope: `worldtapeai.com/`.

- This directory now only holds redirect notes for the RLDS browser/annotation surfaces that live inside the ContinuonAI consumer app/robot controller (`continuonai/`). Keep it lightweight; implementation belongs with the Flutter app.
- When updating references to RLDS browsing/annotation flows, align them with the contracts in `proto/`, the root README, and `continuonai/README.md` so the consolidated surfaces stay in sync.
- Avoid adding build artifacts or node_modules here; keep examples framework-agnostic unless explicitly documenting integration steps.
- Testing expectations: documentation changes only-no web build commands required unless executable code is introduced.
