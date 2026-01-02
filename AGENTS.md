# Agent Instructions (Repository Root)

Scope: All files in this repository unless a deeper `AGENTS.md` overrides these notes.

- Preserve the product boundaries documented in `README.md` and `PRD.md`; do not merge code across products without updating the relevant product README and contracts.
- Standardize product naming (avoid deprecated labels):
  - **Continuon Brain runtime** (preferred) vs. legacy **ContinuonBrain/OS** label; use "runtime" when referring to the on-device executor.
  - **Continuon Brain Studio**: desktop editor/IDE for authoring and testing experiences.
  - **Continuon AI app**: mobile companion client.
  - **Continuon Cloud**: hosted services, APIs, and orchestration.
- Favor the existing toolchains: Gradle wrapper for Kotlin/Android, Flutter CLI for Dart, Node/TypeScript for mocks, and Python tooling already in `continuonbrain`. Avoid introducing alternate package managers for the same language.
- Proto/schema changes must stay backward-compatible. When editing files under `proto/`, run `./gradlew validateProtoSchemas` (and `./gradlew generateProtoKotlin` if stubs need regeneration) before submitting.
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
- Prefer the opt-in “curiosity boot” path for early seed days: `continuonbrain/eval/wiki_curiosity_boot.py` runs bounded offline sessions when `CONTINUON_WIKI_JSONL` is present and `CONTINUON_ENABLE_WIKI_CURIOSITY=1`.

## Curiosity-Driven Learning (Gemini)
- **Pattern**: "Teacher Mode" / Subagent Delegation.
- **Flow**:
  1.  **Agent Manager (HOPE)**: Generates a question based on a "Curiosity Directive".
  2.  **System**: INTERCEPTS the turn if `delegate_model_hint="consult:gemini"`.
  3.  **Subagent (Gemini)**: Receives the question, generates a high-quality answer (using `google-genai` SDK).
  4.  **Agent Manager**: Receives the answer as "User" input (simulating a subagent reply), synthesizes insights, and decides on the next question or action.
- **Tooling**:
  - `continuonbrain/services/brain_service.py` (`RunChatLearn`): orchestrates the loop and handles delegation.
  - `continuonbrain/utils/gemini_cli.py`: Internal utility wrapping the Google GenAI SDK.
  - `scripts/run_learning_session.py`: Operational script to ensure server readiness and trigger the session.

## On-device WaveCore seed + HOPE eval (JAX path)
- WaveCore loops (fast/mid/slow) run via `POST /api/training/wavecore_loops`; defaults use JSON RLDS at `/opt/continuonos/brain/rlds/episodes`, export seed manifest/checkpoint to `/opt/continuonos/brain/model/adapters/candidate/core_model_seed`, checkpoint dir `/opt/continuonos/brain/trainer/checkpoints/core_model_seed`.
- Model presets: `pi5` (default), `columnar_small`, `wave_only`, `hybrid`; optional `sparsity_lambda` (L1) per loop; JIT off by default.
- HOPE eval: graded Q&A logged as RLDS; endpoint `POST /api/training/hope_eval` or set `run_hope_eval` in wavecore payload. Default questions file `continuonbrain/eval/hope_eval_questions.json`; episodes land in `/opt/continuonos/brain/rlds/episodes/hope_eval_<ts>.json`.
- Fallback model order for eval: prefer `google/gemma-370m`, then `google/gemma-3n-2b`; primary chat runs HOPE agent manager first.
- Cloud TPU handoff + re-acquire endpoints (UI-backed, offline-first): `GET /api/training/cloud_readiness`, `POST /api/training/export_zip`, `GET /api/training/exports`, `POST /api/training/install_bundle` (`kind`: `jax_seed_manifest` | `edge_bundle` | `vertex_edge`).
- Training visualization endpoints (UI): `GET /api/training/metrics`, `GET /api/training/eval_summary`, `GET /api/training/data_quality`.
- Chat → RLDS logging is supported but **opt-in** (privacy): enable only with explicit consent via `CONTINUON_LOG_CHAT_RLDS=1`.

## Seed Model Architecture (Universal Initialization)

The **Seed Model** is the universal initialization point for every robot in the ecosystem. It is a permanent, hardware-agnostic core—not a temporary bootstrap phase.

### Current Version: v3.0.0 (January 2026)

| Metric | Value |
|--------|-------|
| **Parameters** | 3.4M |
| **Memory** | 14 MB (float32) |
| **Embedding** | EmbeddingGemma-300m (768-dim) |
| **Inference** | 231 steps/sec (4.3ms/step) |
| **Training Loss** | 0.011 |
| **RLDS Episodes** | 4,218 |

### Seed Model (Permanent Core)
- **Hardware-Agnostic**: Runs on ARM, x64, RISC-V, quantum, neuromorphic
- **WaveCore (JAX)**: 3.4M params Mamba SSM, O(n) complexity
- **CMS Memory**: 3-level hierarchical (Fast/Mid/Slow) with write-back
- **EmbeddingGemma-300m**: 768-dim semantic embeddings
- **Context Graph**: Relational reasoning with entity tracking
- **Decision Traces**: Explainable provenance logging
- **Checkpoint location**: `/opt/continuonos/brain/model/seed_stable/`

### Scaling Roadmap (Golden Rule: <8GB RAM)

| Version | Parameters | Memory | Target | Status |
|---------|------------|--------|--------|--------|
| v2.0 | 1M | 4 MB | Pi 5 (8GB) | ✅ Released |
| **v3.0** | **3.4M** | **14 MB** | **Pi 5 (8GB)** | **✅ Current** |
| v4.0 | 25M | 100 MB | Pi 5 (8GB) | 🔶 Q1 2026 |
| v5.0 | 100M | 200 MB | Pi 5 + Float16 | 🔶 Q2 2026 |
| v6.0 | 500M | 500 MB | 8GB + Int8 | 🔶 Q3 2026 |

**Memory Budget (8GB device):**
- Total RAM: 8.0 GB
- OS + Runtime overhead: 3.3 GB
- **Available for model: 4.7 GB (~1.2B params)**
- Current utilization: **0.3%** (room to grow 350x!)

### Continuous Evolution (Post-Initialization)
After seed initialization, robots learn continuously:
- **Local Learning**: Fast/Mid loops on device (Pi5, Jetson, etc.)
- **Cloud Aggregation**: TPU slow loop trains on aggregated RLDS
- **OTA Updates**: Updated weights distributed to devices
- **Scaffold Evolution**: Gemma 3n provides chat; HOPE WaveCore grows capabilities

### Hardware Portability
| Platform | Runtime | Accelerator | Status |
|----------|---------|-------------|--------|
| ARM64 (Pi5) | JAX CPU | Hailo-8 NPU | ✅ Primary |
| ARM64 (Jetson) | JAX CUDA | Tensor Cores | ✅ Supported |
| x86_64 (PC/Cloud) | JAX TPU/CUDA | TPU/GPU | ✅ Supported |
| RISC-V / Apple Silicon | Planned | Custom | 🔶 Future |
| Quantum / Neuromorphic | Research | QPU/Loihi | 🔮 Research |

### Training Pipeline
1. **RLDS Episodes** → Text extraction → EmbeddingGemma (768-dim)
2. **WaveCore Training** → JAX/Flax, Adam optimizer, gradient clipping
3. **CMS Memory** → 3-level write-back during inference
4. **Hailo Export** → ONNX → HEF (requires DFC on x86)

See `docs/seed-to-hope-evolution.md` for full architecture details.
See `continuonbrain/jax_models/scaling_configs.py` for tier definitions.

## Safety Kernel (Ring 0)

The **Safety Kernel** operates at Ring 0 (highest privilege) like the Unix kernel. It is:
- **First to initialize** on boot, before any other component
- **Cannot be disabled** or bypassed by any other code
- **Veto power** over all actions via `SafetyKernel.allow_action()`
- **Hardware E-Stop** via GPIO (direct motor power cutoff)

### Ring Architecture
```
Ring 0 - SAFETY KERNEL (cannot be bypassed)
Ring 1 - Hardware Abstraction (sensors, actuators)
Ring 2 - Core Runtime (Seed Model, WaveCore, CMS)
Ring 3 - User Space (Chat, API, UI)
```

### Key Components
- `continuonbrain/safety/kernel.py` - Ring 0 SafetyKernel singleton
- `continuonbrain/safety/protocol.py` - Protocol 66 (default safety rules)
- `continuonbrain/safety/bounds.py` - Workspace and motion limits
- `continuonbrain/safety/monitor.py` - Continuous safety monitoring

### Usage
```python
from continuonbrain.safety import SafetyKernel

# All actions must pass through Ring 0
if SafetyKernel.allow_action(action):
    execute(action)

# Emergency stop (always works, cannot be blocked)
SafetyKernel.emergency_stop("Reason")
```

See `continuonbrain/safety/README.md` for full documentation.

## Ownership / pairing (LAN-only, non-biometric)
- Prefer **QR pairing + 6-digit confirm code** for local ownership claim; do **not** implement face recognition / biometric identification.
- Endpoints (robot runtime):
  - `POST /api/ownership/pair/start`, `POST /api/ownership/pair/confirm`, `GET /api/ownership/status`, `GET /api/ownership/pair/qr`, `GET /pair?token=...`.
  - Keep `/api/ownership/status` backward-compatible for existing clients that expect flat keys.

Note: Conversation on 2025-12-10 about Pi5 startup/training is logged at `docs/conversation-log.md` (headless Pi5 boot defaults, optional background trainer, tuned Pi5 training config, RLDS origin tagging).