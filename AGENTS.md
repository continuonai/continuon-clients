# Agent Instructions (Repository Root)

Scope: All files in this repository unless a deeper `AGENTS.md` overrides these notes.

- Preserve the product boundaries documented in `README.md` and `PRD.md`; do not merge code across products without updating the relevant product README and contracts.
- Standardize product naming (avoid deprecated labels):
  - **Continuon Brain runtime** (preferred) vs. legacy **ContinuonBrain/OS** label; use "runtime" when referring to the on-device executor.
  - **Continuon Brain Studio**: desktop editor/IDE for authoring and testing experiences.
  - **Continuon AI app**: mobile companion client.
  - **Continuon Cloud**: hosted services, APIs, and orchestration.
- Favor the existing toolchains: Gradle wrapper for Kotlin/Android, Flutter CLI for Dart, Node/TypeScript for mocks, and Python tooling already in `continuonbrain`. Avoid introducing alternate package managers for the same language.
- Proto/schema changes must stay backward-compatible. When editing files under `proto/`, run `./gradlew formatProto validateProtoSchemas` and regenerate language stubs where applicable before submitting.
- Keep placeholder directories (e.g., cloud, org, web) lightweight: update specs and contracts here, but push production implementations to their dedicated repos as described in `README.md`.
- For documentation updates, cross-link the product-specific READMEs when mentioning flows that span multiple products.
- Testing expectations by area:
  - Kotlin/Android XR app: `./gradlew :apps:continuonxr:testDebugUnitTest` (and `assembleDebug` if build files change).
  - Flutter companion: `flutter analyze` and `flutter test integration_test/connect_and_record_test.dart`.
  - Mock ContinuonBrain service: `npm run build` (runs proto generation + TypeScript compile).
  - Trainer scaffolding (Python): run targeted module import checks or a short `python -m continuonbrain.trainer.local_lora_trainer --help` to ensure dependencies resolve.
  If a command is infeasible in the current environment, note the limitation in your summary.
- Do not commit generated binaries, large media files, or secrets. Keep diffs readable and prefer adding comments near non-obvious logic.

- Upcoming requirement (Continuon AI web): wire the public RLDS viewer to a Continuon Cloud public-episodes API using signed URLs; uploads must carry a `share` block (public flag, slug, title, license, tags) and only `public=true` episodes should list. Coordinate README/AGENTS updates when enabling.
- Public safety/PII: for any public listing, require content rating/audience fields, PII attestation, automated PII/safety scans (faces/plates blur, OCR/ASR PII, profanity/toxicity). Only list when `pii_cleared=true` and `pending_review=false`; prefer serving redacted assets when `pii_redacted=true`.

TODO (offline Wikipedia context)
- Once an offline `wikimedia/wikipedia` dump or JSONL corpus is available, wire `continuonbrain/eval/wiki_retriever.py` into HOPE eval flows so prompts can include retrieved snippets; keep provenance in RLDS and default to no-op when corpus is absent.

## On-device WaveCore seed + HOPE eval (JAX path)
- WaveCore loops (fast/mid/slow) run via `POST /api/training/wavecore_loops`; defaults use JSON RLDS at `/opt/continuonos/brain/rlds/episodes`, export seed manifest/checkpoint to `/opt/continuonos/brain/model/adapters/candidate/core_model_seed`, checkpoint dir `/opt/continuonos/brain/trainer/checkpoints/core_model_seed`.
- Model presets: `pi5` (default), `columnar_small`, `wave_only`, `hybrid`; optional `sparsity_lambda` (L1) per loop; JIT off by default.
- HOPE eval: graded Q&A logged as RLDS; endpoint `POST /api/training/hope_eval` or set `run_hope_eval` in wavecore payload. Default questions file `continuonbrain/eval/hope_eval_questions.json`; episodes land in `/opt/continuonos/brain/rlds/episodes/hope_eval_<ts>.json`.
- Fallback model order for eval: prefer `google/gemma-370m`, then `google/gemma-3n-2b`; primary chat runs HOPE agent manager first.

Note: Conversation on 2025-12-10 about Pi5 startup/training is logged at `docs/conversation-log.md` (headless Pi5 boot defaults, optional background trainer, tuned Pi5 training config, RLDS origin tagging).