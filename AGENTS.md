# Agent Instructions (Repository Root)

**Last Updated:** 2026-01-25

Scope: All files in this repository unless a deeper `AGENTS.md` overrides these notes.

---

## Current Architecture (January 2026)

The system has evolved from the original HOPE architecture to a simpler, more effective design:

```
Chief Scientist (Claude Code)     ← Strategic direction, goal setting
        ↓
Learning Partner                  ← Autonomous 5-phase training loop
        ↓
Brain B                          ← Teachable robot brain (production)
        ↓
Hardware (OAK-D, Arms, Drive)    ← Physical robot control
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Chief Scientist** | Claude Code as strategic brain developer | `scripts/compound/chief_scientist.py` |
| **Learning Partner** | Autonomous learning loop (OBSERVE→SIMULATE→TRAIN→TRANSFER→DEPLOY) | `scripts/compound/learning_partner.py` |
| **Brain B** | Simple teachable brain (production) | `brain_b/` |
| **Trainer UI** | Web interface for training | `trainer_ui/` |
| **Seed Model v4.2.0** | 12.8M param model, 0.84 benchmark | `models/` |

### Deprecated (Legacy References)

- ~~HOPE Architecture~~ → Now use Learning Partner + Chief Scientist
- ~~Pi 5 + Hailo NPU~~ → OAK-D camera is primary perception  
- ~~Gemma on-device~~ → Seed Model v4.2.0
- ~~Manual curiosity sessions~~ → Chief Scientist automates this

---

## Product Naming Standards

- **Continuon Brain runtime** (preferred) vs. legacy ContinuonBrain/OS label
- **Continuon Brain Studio**: desktop editor/IDE
- **Continuon AI app**: mobile companion client
- **Continuon Cloud**: hosted services, APIs

---

## Development Guidelines

### Toolchains
- Gradle wrapper for Kotlin/Android
- Flutter CLI for Dart
- Node/TypeScript for mocks
- Python tooling in `continuonbrain` and `brain_b`

### Testing

```bash
# Android XR app
./gradlew :apps:continuonxr:testDebugUnitTest

# Flutter companion
flutter analyze && flutter test integration_test/

# Python brain
cd brain_b && python -m pytest tests/ -v

# Learning Partner status
python scripts/compound/learning_partner.py --status
```

### Proto/Schema Changes
- Must stay backward-compatible
- Run `./gradlew validateProtoSchemas` before submitting
- Regenerate stubs with `./gradlew generateProtoKotlin` if needed

---

## Training System

### Chief Scientist Goals (Current)

1. **navigate_home** - Navigate rooms without hitting furniture
2. **recognize_family** - Recognize family by face and voice
3. **fetch_items** - Fetch and deliver items
4. **give_reminders** - Remember and deliver reminders
5. **entertain** - Fun interactions, jokes, games

### Learning Partner Phases

```
OBSERVE → SIMULATE → TRAIN → TRANSFER → DEPLOY
   ↑___________________________________________|
```

1. **OBSERVE**: Scan real environment (room_scanner.py)
2. **SIMULATE**: Generate training in HomeScan simulator
3. **TRAIN**: Brain B learns from simulation + real experience
4. **TRANSFER**: Export to neural networks for fine-tuning
5. **DEPLOY**: Apply learned behaviors on robot

### Quick Commands

```bash
# Check goals
python scripts/compound/chief_scientist.py --goals

# Run training cycle
python scripts/compound/learning_partner.py --train

# Check brain health
python scripts/compound/learning_partner.py --status

# Start trainer UI
cd trainer_ui && python server.py
```

---

## Seed Model Architecture

**Version:** v4.2.0 (January 2026)

| Metric | Value |
|--------|-------|
| Parameters | 12.8M |
| Memory | 51 MB (model) + 27 MB (encoder) |
| Architecture | WaveCore Mamba SSM + CMS 3-Level Memory |
| Inference | 50+ Hz (20ms/step) |
| Benchmark | 0.84 score (17/23 tests) |
| Level | ADVANCED (L3 of 6) |
| RLDS Episodes | 4,218+ |

Run benchmark: `python -m continuonbrain.eval.progressive_benchmark`

---

## RLDS Data Pipeline

- Episodes stored in `brain_b_data/` and `trainer_ui/brain_b_data/`
- JSON format for portability
- Chat → RLDS logging is **opt-in** (privacy): `CONTINUON_LOG_CHAT_RLDS=1`

---

## Hardware

### Primary Sensors
- **OAK-D Camera**: RGB + Depth perception
- **Microphone**: Voice commands
- **Encoders**: Wheel odometry

### Actuators
- **6-Axis Arm**: Object manipulation
- **Mecanum Drive**: Omnidirectional movement
- **Speaker**: Audio feedback

### Mechanical Design
- V-slot mast design: `docs/mechanical-design/v-slot-mast-design.md`
- Alternative builds: `docs/mechanical-design/alternative-robot-builds.md`

---

## Android XR Trainer (Qualcomm Bounty)

New feature for XR glasses-based training:

Location: `apps/continuonxr/src/main/java/com/continuonxr/app/trainer/`

Components:
- `TrainerScreen.kt` - Main UI
- `DriveControls.kt` - Mecanum drive via gestures
- `ArmControls.kt` - 6-axis arm control
- `VoicePanel.kt` - Voice commands
- `TrainerRldsExtensions.kt` - Episode recording

---

## Safety & Privacy

- PII scans required for public episodes
- Face/plate blur for public content
- Content rating required for sharing
- Only list when `pii_cleared=true` and `pending_review=false`

---

## Files to Update When Changing Architecture

1. This file (`AGENTS.md`)
2. `README.md` Quick Start section
3. `docs/NEXT_STEPS.md`
4. `docs/TODO_BACKLOG.md`
5. `CLAUDE.md` (agent-specific instructions)

---

*Updated 2026-01-25 - Replaced HOPE references with current Learning Partner + Chief Scientist architecture*
