# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Quick Start

### Brain B (Simple Robot Brain)

```bash
cd brain_b
pip install -r requirements.txt  # or just: pip install fastapi uvicorn
python main.py
```

Talk to it:
```
You: forward
Bot: Moving forward.

You: teach patrol
Bot: Ready to learn 'patrol'. Show me what to do.

You: forward
You: left
You: done
Bot: Learned 'patrol': forward -> left

You: patrol
Bot: Running 'patrol'...
```

### Trainer UI (Web Interface)

```bash
cd trainer_ui
pip install fastapi uvicorn opencv-python face-recognition
python server.py
# Open http://localhost:8000
```

Features: Drive (WASD), Robot arm (6-axis), Camera + face recognition, Voice, Claude Code chat, RLDS recording.

### Full ContinuonBrain (Advanced)

```bash
cd continuonbrain
pip install -r requirements.txt
python -m continuonbrain.startup_manager
# API at http://localhost:8080
```

## Project Structure

```
ContinuonXR/
├── brain_b/                    # Simple teachable robot brain (Option B)
│   ├── actor_runtime/          # Event-sourced agent model
│   ├── conversation/           # Natural language interface
│   ├── hardware/               # Motor/sensor abstraction
│   ├── sandbox/                # Cowork-style isolation gates
│   └── main.py                 # Entry point
│
├── trainer_ui/                 # Web UI for training Brain A
│   ├── server.py               # FastAPI + WebSocket server
│   ├── static/index.html       # Single-page UI
│   └── face_db/                # Face recognition database
│
├── continuonbrain/             # Full HOPE brain (Option A)
│   ├── hope_impl/              # HOPE architecture
│   ├── ralph/                  # Context rotation layers
│   ├── kernel/                 # Safety kernel
│   └── services/               # Domain services
│
├── .claude/                    # Claude Code hooks
│   ├── settings.json           # Hook configuration
│   └── hooks/                  # Hook scripts
│       ├── session-start.sh    # Initialize Brain B
│       ├── validate-robot-action.py   # PreToolUse validation
│       ├── record-for-training.py     # PostToolUse recording
│       └── export-rlds.py      # Stop hook - RLDS export
│
├── continuonai/                # Flutter companion app
├── apps/continuonxr/           # Android XR app
└── docs/architecture/          # Design documents
```

## Brain Architecture

### Two Brains

| Brain | Complexity | Use Case |
|-------|------------|----------|
| **Brain B** | ~500 LOC | Talk to robot, teach behaviors, fast iteration |
| **Brain A** | ~50,000 LOC | Neural network, cloud training, production |

### Training Pipeline

```
Brain B (simple) → generates RLDS episodes → trains Brain A (complex)
```

The Claude Code hooks in `.claude/` automatically:
1. Validate robot actions through Brain B before execution
2. Record outcomes for training
3. Export sessions as RLDS episodes

## Claude Code Hooks

When you start a session in this repo, the hooks:

1. **SessionStart** - Initializes Brain B state, loads guardrails
2. **PreToolUse** - Validates `robot*`, `motor*` commands against safety rules
3. **PostToolUse** - Records actions, auto-generates guardrails from failures
4. **Stop** - Exports session as RLDS episode for Brain A training

## Key Commands

### Brain B

```bash
# Run Brain B CLI
python brain_b/main.py

# With real hardware
python brain_b/main.py --real

# Custom data directory
python brain_b/main.py --data-dir /path/to/data
```

### Trainer UI

```bash
# Start web server
python trainer_ui/server.py

# Custom port
uvicorn trainer_ui.server:app --port 9000
```

### Full Brain (ContinuonBrain)

```bash
# Start with auto-detect
python -m continuonbrain.startup_manager

# Run tests
pytest continuonbrain/tests/

# Train on RLDS episodes
python -m continuonbrain.trainer.local_lora_trainer \
  --episodes ./continuonbrain/rlds/episodes/
```

## Testing

```bash
# Brain B
cd brain_b && python -m pytest

# Trainer UI
cd trainer_ui && python -c "from server import *; print('OK')"

# Full brain
pytest continuonbrain/tests/

# E2E smoke tests
pytest tests/e2e/test_full_stack_smoke.py -s
```

## Design Documents

| Document | Path | Purpose |
|----------|------|---------|
| Actor Runtime | `docs/architecture/ACTOR_RUNTIME_DESIGN.md` | Brain B design |
| Sandbox Pattern | `docs/architecture/SANDBOXED_AGENT_RUNTIME_PATTERN.md` | Cowork-style isolation |
| ContinuonVM | `docs/architecture/CONTINUON_VM_PATTERN.md` | Unified Ralph+OpenProse+Sandbox |

## Conventions

- **Brain B**: Pattern-based, markdown state, fast iteration
- **Brain A**: Neural network, RLDS training, production
- **Hooks**: Brain B validates before Brain A learns
- **RLDS**: Universal training data format

## Hardware

Brain B supports:
- **Motors**: GPIO PWM or mock
- **Camera**: OpenCV + face_recognition
- **Voice**: Web Speech API + system TTS

Trainer UI supports:
- **Browser camera/mic**: getUserMedia API
- **Face recognition**: dlib via face-recognition package
- **Claude Code**: CLI subprocess

## Related Files

- `brain_b/README.md` - Brain B documentation
- `trainer_ui/README.md` - Trainer UI documentation
- `continuonbrain/README.md` - Full brain documentation
- `continuonai/CLAUDE.md` - Flutter app guidance
