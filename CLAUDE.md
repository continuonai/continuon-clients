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

### Learning Partner (Autonomous Learning)

The Learning Partner helps the robot continuously improve by understanding its capabilities and running training autonomously.

```bash
# Check brain health and learning phase
python scripts/compound/learning_partner.py --status

# See prioritized learning goals
python scripts/compound/learning_partner.py --goals

# Run one training cycle
python scripts/compound/learning_partner.py --train

# Run continuous learning daemon
python scripts/compound/learning_partner.py --interval 600
```

Learning Phases:
```
OBSERVE → SIMULATE → TRAIN → TRANSFER → DEPLOY
   ↑___________________________________________|
```

1. **OBSERVE**: Scan home environment → 3D training world
2. **SIMULATE**: Generate training episodes in HomeScan
3. **TRAIN**: Train Brain B and ContinuonBrain from episodes
4. **TRANSFER**: Adapt simulation skills to real hardware
5. **DEPLOY**: Run learned behaviors on robot

## Project Structure

```
ContinuonXR/
├── brain_b/                    # Simple teachable robot brain
│   ├── actor_runtime/          # Event-sourced agent model
│   ├── conversation/           # Natural language interface
│   ├── hardware/               # Motor/sensor abstraction
│   ├── sandbox/                # Cowork-style isolation gates
│   └── main.py                 # Entry point
│
├── trainer_ui/                 # Web UI for training ContinuonBrain
│   ├── server.py               # FastAPI + WebSocket server
│   ├── static/index.html       # Single-page UI
│   └── face_db/                # Face recognition database
│
├── continuonbrain/             # Full HOPE brain (production)
│   ├── hope_impl/              # HOPE architecture
│   ├── ralph/                  # Learning loops (fast/mid/slow)
│   ├── mambawave/              # SSM + Spectral architecture (skill)
│   ├── wavecore/               # Spectral foundation (genesis)
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
├── scripts/
│   └── compound/               # Autonomous learning system
│       ├── learning_partner.py # Main learning daemon
│       ├── analyzer.py         # Codebase analyzer
│       └── daemon.py           # Bug-fix daemon
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
| **ContinuonBrain** | ~50,000 LOC | Neural network, cloud training, production |

### Training Pipeline

```
Brain B (simple) → generates RLDS episodes → trains ContinuonBrain (complex)
```

The Claude Code hooks in `.claude/` automatically:
1. Validate robot actions through Brain B before execution
2. Record outcomes for training
3. Export sessions as RLDS episodes

## ContinuonBrain Components

ContinuonBrain is composed of integrated skills and components:

### MambaWave (Skill)

MambaWave is the core neural architecture - an evolution of WaveCore that combines:
- **WaveCore spectral design**: FFT-based filtering with learnable complex filters
- **Mamba SSM architecture**: Efficient state space model for sequences

```python
from continuonbrain.mambawave import MambaWaveModel, MambaWaveConfig, MambaWaveSkill

# Three loop variants for different compute budgets
fast_config = MambaWaveConfig.fast_loop()   # Real-time inference (64-dim, 1 layer)
mid_config = MambaWaveConfig.mid_loop()     # Online learning (128-dim, 4 layers)
slow_config = MambaWaveConfig.slow_loop()   # Batch training (256-dim, 8 layers)

# Use as a skill
skill = MambaWaveSkill()
result = skill.execute("predict_world", joint_pos=[0.1, 0.2, 0.3], joint_delta=[0.01, 0, 0])
```

**MambaWave Structure:**
```
continuonbrain/mambawave/
├── config.py           # Unified config (WaveCore + Mamba SSM params)
├── skill.py            # Skill interface for ContinuonBrain
├── world_model.py      # State prediction for planning
├── layers/
│   ├── ssm_spectral.py     # Core SSM + Spectral fusion
│   └── mamba_wave_block.py # Full block with optional attention
└── models/
    └── mambawave_model.py  # Full sequence model
```

### Ralph (Component)

Ralph provides learning loops that help the agent learn through context rotation:
- **Fast Loop**: Real-time inference
- **Mid Loop**: Online learning with context
- **Slow Loop**: Batch training and consolidation

```python
from continuonbrain.ralph import MetaRalph

ralph = MetaRalph()
# Ralph orchestrates fast/mid/slow loops
```

### WaveCore (Genesis)

WaveCore is the foundational spectral architecture that MambaWave evolved from:
- FFT-based spectral blocks
- Hybrid attention + spectral blocks
- Training utilities

## Claude Code Hooks

When you start a session in this repo, the hooks:

1. **SessionStart** - Initializes Brain B state, loads guardrails
2. **PreToolUse** - Validates `robot*`, `motor*` commands against safety rules
3. **PostToolUse** - Records actions, auto-generates guardrails from failures
4. **Stop** - Exports session as RLDS episode for ContinuonBrain training

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

### MambaWave

```bash
# Test MambaWave
python -c "from continuonbrain.mambawave import MambaWaveModel, MambaWaveConfig; print('OK')"

# Create and test model
python -c "
from continuonbrain.mambawave import MambaWaveModel, MambaWaveConfig
model = MambaWaveModel(MambaWaveConfig.fast_loop())
print(f'Parameters: {model.count_parameters():,}')
"
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
- **ContinuonBrain**: Neural network, RLDS training, production
- **MambaWave**: SSM + Spectral skill for sequence/world modeling
- **Ralph**: Learning loop component for context rotation
- **Hooks**: Brain B validates before ContinuonBrain learns
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
