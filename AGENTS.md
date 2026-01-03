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

The **Seed Model** is the universal initialization point for every robot in the ecosystem. It is a permanent, hardware-agnostic core—not a temporary bootstrap phase. The goal is a fully **self-training embodied AI** that works without external dependencies.

### Current Version: v4.2.0 (January 2026)

| Metric | Value |
|--------|-------|
| **Parameters** | 12.8M |
| **Memory** | 51 MB (model) + 27 MB (self-contained encoder) |
| **Architecture** | WaveCore Mamba SSM + CMS 3-Level Memory |
| **Embedding** | Self-contained (6.7M, 768-dim) or EmbeddingGemma-300m |
| **Inference** | 50+ Hz (20ms/step) - real-time capable |
| **Benchmark** | 0.84 score (17/23 progressive tests) |
| **Highest Level** | ADVANCED (L3 of 6) |
| **RLDS Episodes** | 4,218 |

### Progressive Benchmark (6-Level Embodied AI Test)

| Level | Tests | Score | Capabilities |
|-------|-------|-------|--------------|
| L1 BASIC | 3/3 ✅ | 1.00 | Output stability, inference speed, non-trivial output |
| L2 INTERMEDIATE | 3/3 ✅ | 0.82 | Command differentiation, state evolution, spatial understanding |
| L3 ADVANCED | 3/3 ✅ | 0.84 | Memory persistence, context switching, hierarchical commands |
| L4 EXPERT | 5/5 ⚠️ | 0.71 | Error recovery, multi-step planning, safety, sensor fusion |
| L5 AUTONOMOUS | 5/5 ✅ | 0.92 | Self-monitoring, continuous learning, world model, spatial reasoning |
| L6 SWARM | 6/6 🔶 | TBD | Parts understanding, build planning, coordination, replication |

Run benchmark: `python -m continuonbrain.eval.progressive_benchmark`

### Seed Model (Permanent Core)
- **Hardware-Agnostic**: Runs on ARM, x64, RISC-V, quantum, neuromorphic
- **WaveCore (JAX)**: 12.8M params Mamba SSM, O(n) complexity
- **CMS Memory**: 3-level hierarchical (Fast/Mid/Slow) with write-back
- **Self-Contained Encoder**: 6.7M params, 768-dim (no transformers dependency)
- **HAL Discovery**: Auto-detects USB/I2C/PCIe accessories (cameras, arms, NPUs)
- **Context Graph**: Relational reasoning with entity tracking
- **Decision Traces**: Explainable provenance logging
- **Checkpoint location**: `/opt/continuonos/brain/model/seed_stable/`

### Hardware Abstraction Layer (HAL)
The HAL automatically discovers connected accessories:
```python
from continuonbrain.hal import discover_accessories, AccessoryRegistry
accessories = discover_accessories()  # USB, I2C, PCIe, GPIO scanning
# Returns: [OAK-D Camera, Hailo-8 NPU, SO-ARM100 (when connected), ...]
```

Known accessory registry: `continuonbrain/hal/accessory_registry.py`

### Scaling Roadmap (Golden Rule: <8GB RAM)

| Version | Parameters | Memory | Benchmark | Status |
|---------|------------|--------|-----------|--------|
| v2.0 | 1M | 4 MB | — | ✅ Released |
| v3.0 | 3.4M | 14 MB | — | ✅ Released |
| **v4.2** | **12.8M** | **51 MB** | **0.84** | **✅ Current** |
| v5.0 | 50M | 200 MB | 0.90+ | 🔶 Q1 2026 |
| v6.0 | 200M | 800 MB | 0.95+ | 🔶 Q2 2026 |

**Memory Budget (8GB device):**
```
OS + Python + JAX       2.0 GB
Seed Model (WaveCore)   0.05 GB
Self-Contained Encoder  0.03 GB  ← Replaces 1.2GB EmbeddingGemma
CMS Memory              0.5 GB
RLDS Episodes           1.0 GB   ← Local memories (mid-loop)
Context Graph           0.3 GB
Safety Kernel           0.1 GB
────────────────────────────────
TOTAL                   3.98 GB (50% of budget)
HEADROOM                4.02 GB (for scaling + local memories)
```

### Self-Training Architecture
```
Human Chat ──┬──► HOPE Agent Manager ──► Seed Model (WaveCore)
             │                              ↓
             │                         CMS Memory (Fast/Mid/Slow)
             │                              ↓
             └──► RLDS Episodes ──────► Local Storage (Mid-Loop)
                       │                    │
                       │   [Opt-in]         │
                       ▼                    ▼
                  Continuon Cloud ◄─── Optional Sync
                       │
                       ▼
                  TPU Slow Loop Training
                       │
                       ▼
                  OTA Bundle ──────► Device Update
                                   (preserves local memories)
```

### Continuous Evolution (Post-Initialization)
After seed initialization, robots learn continuously:
- **Fast Loop**: On-device reflexive control (50-100ms)
- **Mid Loop**: Local RLDS episodes, adapter training
- **Slow Loop**: Cloud TPU training on aggregated RLDS (optional sync)
- **OTA Updates**: Updated weights merge with Memory Plane (preserves local)
- **Self-Contained**: No external LLM required (Gemma 3n optional scaffold)

### Hardware Portability
| Platform | Runtime | Accelerator | Status |
|----------|---------|-------------|--------|
| ARM64 (Pi5) | JAX CPU | Hailo-8 NPU | ✅ Primary (verified) |
| ARM64 (Jetson) | JAX CUDA | Tensor Cores | ✅ Supported |
| x86_64 (PC/Cloud) | JAX TPU/CUDA | TPU/GPU | ✅ Supported |
| RISC-V / Apple Silicon | Planned | Custom | 🔶 Future |

### Training Pipeline
1. **RLDS Episodes** → Text extraction → Self-contained encoder (768-dim)
2. **WaveCore Training** → JAX/Flax, Adam optimizer, gradient clipping
3. **CMS Memory** → 3-level write-back during inference (verified in benchmark)
4. **Progressive Benchmark** → 6-level capability testing (includes swarm)
5. **Hailo Export** → ONNX → HEF (requires DFC on x86)

### Key Files
- `continuonbrain/jax_models/core_model.py` - WaveCore Mamba SSM
- `continuonbrain/jax_models/text_encoder.py` - Self-contained encoder
- `continuonbrain/hal/` - Hardware abstraction layer
- `continuonbrain/eval/progressive_benchmark.py` - 6-level benchmark
- `continuonbrain/seed/model.py` - Seed model wrapper
- `continuonbrain/swarm/` - Swarm intelligence (robot building, coordination)
- `continuonbrain/safety/work_authorization.py` - Gray-area safety handling

See `docs/seed-to-hope-evolution.md` for full architecture details.

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

### Gray-Area Safety (Work Authorization)
Some actions are normally prohibited but legitimate in specific contexts (demolition, recycling, data deletion). The work authorization system handles these:

- **Work orders** - Signed authorization for specific destructive actions
- **Property claims** - Proof of ownership before destruction
- **Role verification** - Only owner/leasee can authorize
- **Multi-party approval** - Critical actions need 2+ approvers
- **Audit trail** - All actions logged in tamper-evident log

Key files:
- `continuonbrain/safety/work_authorization.py` - Work order management
- `continuonbrain/safety/anti_subversion.py` - Attack prevention

### Anti-Subversion Layer
Prevents bad actors from bypassing safety:
- **ImmutableSafetyCore** - Hardcoded rules frozen at import
- **Cryptographic signatures** - Forged authorizations rejected
- **Prompt injection defense** - Detects "ignore previous instructions" attacks
- **Rate limiting** - Prevents brute force attempts
- **Tamper-evident logging** - Blockchain-style hash chaining

See `continuonbrain/safety/README.md` for full documentation.

## Swarm Intelligence

Enables robots to build other robots and coordinate as a swarm.

### Capabilities
- **Robot Builder** - Plan construction from available parts
- **Seed Replicator** - Clone seed image to new hardware
- **Swarm Coordination** - Multi-robot discovery and task delegation
- **Experience Sharing** - Share learned skills (not personal data)

### Safety Requirements
| Requirement | Description |
|-------------|-------------|
| Owner Authorization | Only creator/owner/leasee can authorize |
| Parts Ownership | Parts must be owned by authorizing party |
| Multi-Party Approval | Critical builds need 2+ approvers |
| Signed Work Orders | Cryptographic authorization |
| Audit Trail | Tamper-evident logging |

### Key Files
- `continuonbrain/swarm/builder.py` - Parts inventory and build plans
- `continuonbrain/swarm/replicator.py` - Seed image cloning
- `continuonbrain/swarm/coordination.py` - Multi-robot communication
- `continuonbrain/swarm/authorized_builder.py` - Safety-integrated builder

See `continuonbrain/swarm/README.md` for full documentation.

## Ownership / pairing (LAN-only)
- Prefer **QR pairing + 6-digit confirm code** for local ownership claim.
- Endpoints (robot runtime):
  - `POST /api/ownership/pair/start`, `POST /api/ownership/pair/confirm`, `GET /api/ownership/status`, `GET /api/ownership/pair/qr`, `GET /pair?token=...`.
  - Keep `/api/ownership/status` backward-compatible for existing clients that expect flat keys.

## User Recognition (Consent-Based Face Recognition)
- Face recognition is **opt-in only** - users must explicitly consent to be remembered.
- Purpose: Robot recognizes familiar users for personalized interactions and role-based access.
- **Privacy principles**:
  - Only face embeddings stored (not photos) - cannot be reversed to images
  - All processing on-device (never sent to cloud)
  - Users can delete their data anytime (right to be forgotten)
  - Transparent: users know when recognition is active
- **User roles** (in order of privilege): creator, owner, leasee, user, guest, unknown
- **Module**: `continuonbrain/recognition/` - see README.md for full documentation.
- **Endpoints**:
  - `POST /api/recognition/register` - Register face with consent
  - `POST /api/recognition/recognize` - Identify user from image
  - `DELETE /api/recognition/revoke/{user_id}` - Delete user data
  - `GET /api/consent/status/{user_id}` - Check consent status

Note: Conversation on 2025-12-10 about Pi5 startup/training is logged at `docs/conversation-log.md` (headless Pi5 boot defaults, optional background trainer, tuned Pi5 training config, RLDS origin tagging).