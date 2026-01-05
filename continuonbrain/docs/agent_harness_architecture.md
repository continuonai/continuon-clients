# Agent Harness Architecture for ContinuonBrain

This document maps the ContinuonBrain architecture to the Agent Harness (OS) pattern, establishing a clear separation between the Brain as the operating system layer and the HOPE Agent as the application layer.

## Overview: Agent Harness Pattern

Based on the Agent Harness architecture:

| Computer Component | AI Equivalent | ContinuonBrain Implementation |
|-------------------|---------------|-------------------------------|
| **CPU** | Model (computation) | WaveCore/Mamba SSM + CMS |
| **RAM** | Context Window | Fast/Mid/Slow CMS Memory Levels |
| **Operating System** | Agent Harness | BrainService + Infrastructure |
| **Application** | Agent | HOPEAgent + Chat Manager Interface |

## Architectural Layers

### Layer 0: Hardware Abstraction (Ring 1)

The HAL layer provides unified interfaces to physical hardware:

```
continuonbrain/hal/              - Hardware Abstraction Layer
continuonbrain/sensors/          - Sensor interfaces (OAK-D, IMU, etc.)
continuonbrain/actuators/        - Actuator interfaces (PCA9685, Drivetrain)
continuonbrain/safety/kernel.py  - Ring 0 Safety Kernel (cannot be bypassed)
```

**Responsibilities:**
- Hardware auto-detection and registration
- Unified sensor/actuator APIs
- Safety constraints enforcement (e-stop, limits)

### Layer 1: Brain Runtime OS (Agent Harness)

The BrainService acts as the operating system layer:

```
continuonbrain/services/brain_service.py  - Main OS orchestrator
continuonbrain/services/world_model_integration.py - Sensory fusion
continuonbrain/hope_impl/                 - HOPE Brain core (CMS, SSM)
continuonbrain/03_mamba_brain/            - Mamba world model
continuonbrain/jax_models/                - JAX model infrastructure
```

**OS Responsibilities (from Agent Harness pattern):**

1. **Context Engineering**
   - CMS memory manages Fast/Mid/Slow timescales
   - Memory compaction and decay (built into CMS)
   - State persistence via Memory Plane

2. **Tool Call Handling**
   - `tool_registry` manages available tools
   - Safety kernel gates all tool calls
   - Resource monitor throttles under pressure

3. **Lifecycle Management**
   - Startup manager initializes subsystems
   - Mode manager (autonomous, teleop, sleep)
   - Background learner orchestrates training

4. **Durability Monitoring**
   - Resource monitor tracks CPU/RAM/temperature
   - Stability monitor tracks model drift
   - Experience logger tracks learning progress

### Layer 2: HOPE Agent Application

The HOPEAgent is the application layer running on top of the OS:

```
continuonbrain/services/agent_hope.py         - Main agent interface
continuonbrain/services/hope_agent_wrapper.py - Chat manager integration
continuonbrain/gemma_chat.py                  - Language interface
```

**Agent Responsibilities:**
- Process user queries (text/voice/gesture)
- Generate responses using learned knowledge
- Identify knowledge gaps and ask questions
- Learn from corrections and demonstrations

## Multi-Modal Input Architecture

```
                    ┌─────────────────────────────────────────┐
                    │          INPUT MODALITIES               │
                    │                                         │
                    │  Vision ─┬─ OAK-D RGB+Depth            │
                    │          ├─ SAM3 Segmentation           │
                    │          └─ Hailo-8 Pose Estimation     │
                    │                                         │
                    │  Audio ──┬─ Speech-to-Text (Whisper)   │
                    │          └─ Voice Activity Detection    │
                    │                                         │
                    │  Text ───┬─ Chat Interface             │
                    │          └─ Command Parser              │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │       WORLD MODEL INTEGRATION           │
                    │    (WorldModelIntegration class)        │
                    │                                         │
                    │  • Multi-modal fusion                   │
                    │  • Object tracking (persistent IDs)     │
                    │  • Spatial relationship inference       │
                    │  • Scene description generation         │
                    │  • SSM-based temporal predictions       │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │         BRAIN OS (Harness)              │
                    │                                         │
                    │  ┌───────────────────────────────────┐  │
                    │  │      CMS MEMORY SYSTEM            │  │
                    │  │  Fast (50-100ms) │ Mid (0.5-10s) │  │
                    │  │  Slow (mins-hrs via cloud)        │  │
                    │  └───────────────────────────────────┘  │
                    │                                         │
                    │  ┌───────────────────────────────────┐  │
                    │  │      SSM/MAMBA WORLD MODEL        │  │
                    │  │  State prediction, rollouts       │  │
                    │  └───────────────────────────────────┘  │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │       HOPE AGENT (Application)          │
                    │                                         │
                    │  • Confidence assessment                │
                    │  • Response generation                  │
                    │  • Knowledge gap identification         │
                    │  • Learning from corrections            │
                    │                                         │
                    │  Fallback: Gemma-3n Chat               │
                    └─────────────────────────────────────────┘
```

## Claude Code as Teacher Interface

The TeacherInterface (in `world_model_integration.py`) enables Claude Code to act as a teaching agent:

```python
class TeacherInterface:
    """
    Teacher interface for Claude Code to guide HOPE learning.

    Capabilities:
    1. provide_answer()     - Answer HOPE's questions
    2. provide_correction() - Correct HOPE's mistakes
    3. demonstrate_action() - Show how to perform actions
    4. get_pending_questions() - Get HOPE's knowledge gaps
    5. suggest_teaching_focus() - Identify what to teach
    """
```

### Teaching Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     TEACHING ARCHITECTURE                       │
│                                                                 │
│   Claude Code (Teacher)                                         │
│         │                                                       │
│         ├─── API: /api/teacher/questions ───▶ get_pending_questions()
│         │                                                       │
│         ├─── API: /api/teacher/answer ─────▶ provide_answer()   │
│         │                                                       │
│         ├─── API: /api/teacher/correct ────▶ provide_correction()
│         │                                                       │
│         └─── API: /api/teacher/demo ───────▶ demonstrate_action()
│                                                                 │
│         ▼                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              HOPE AGENT LEARNING                        │   │
│   │                                                         │   │
│   │  1. Store teaching in ExperienceLogger                  │   │
│   │  2. Update HOPE brain CMS memory                        │   │
│   │  3. Log RLDS episode for Slow-loop training             │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## CMS Memory System Integration

The Continuum Memory System (CMS) implements multi-timescale learning aligned with HOPE:

### Fast Loop (50-100ms)
- Reflexive control, safety overrides
- Liquid Neural Network / SSM on Pi CPU
- Minimal parameters, constant-time inference

### Mid Loop (0.5-10s)
- Skill sequencing, intent inference
- Compact SSM/conv blocks
- On-device adapter training from RLDS

### Slow Loop (minutes-hours)
- Cloud TPU training on aggregated RLDS
- Updates core model weights
- Returns OTA bundles preserving local memories

```python
# CMS Memory Levels (from hope_impl/cms.py)
class CMSRead:
    """Content-addressable hierarchical retrieval across memory levels"""

class CMSWrite:
    """Discrete jump map for memory updates with decay"""

# Each level has:
# - M: Memory matrix [N_slots, d_level]
# - K: Key matrix [N_slots, d_key]
# - decay: Level-specific decay rate
```

## Implementation Recommendations

### 1. Strengthen Agent Harness OS Layer

The BrainService already implements most Agent Harness responsibilities. Formalize by:

```python
# Add explicit harness lifecycle hooks
class AgentHarness:
    """OS-layer abstraction for HOPE agents"""

    def __init__(self, brain_service):
        self.brain = brain_service
        self.context_manager = ContextManager(brain_service.cms)
        self.tool_handler = ToolCallHandler(brain_service.tool_registry)
        self.durability_monitor = DurabilityMonitor()

    def manage_context(self, agent_state):
        """Context engineering: compaction, offloading, isolation"""
        pass

    def handle_tool_call(self, tool_name, args):
        """Opinionated tool call handling with safety gates"""
        pass

    def check_durability(self, step_count):
        """Detect model drift after extended execution"""
        pass
```

### 2. Multi-Modal Input Hub

Create a unified input hub that routes all modalities:

```python
class MultiModalInputHub:
    """Unified input processing for all modalities"""

    def __init__(self, world_model_integration):
        self.world_model = world_model_integration
        self.speech_recognizer = WhisperSTT()
        self.command_parser = CommandParser()

    def process_vision(self, rgb, depth) -> SensoryFrame:
        """Process camera input through SAM + pose estimation"""

    def process_audio(self, audio_buffer) -> Optional[str]:
        """Convert speech to text via Whisper"""

    def process_text(self, text) -> Dict:
        """Parse text commands and queries"""

    def fuse_inputs(self) -> WorldState:
        """Merge all inputs into coherent world state"""
```

### 3. Enhanced Teacher Protocol

Extend the TeacherInterface for richer Claude Code interactions:

```python
class EnhancedTeacherInterface(TeacherInterface):
    """Extended teacher protocol for Claude Code integration"""

    def get_learning_curriculum(self) -> List[LearningGoal]:
        """Get structured curriculum for HOPE learning"""

    def validate_knowledge(self, topic) -> ValidationResult:
        """Test HOPE's knowledge on a topic"""

    def provide_reasoning_trace(self, problem, solution_steps):
        """Demonstrate step-by-step reasoning"""

    def request_clarification(self, context) -> str:
        """Ask Claude Code for clarification"""
```

## File Organization

```
continuonbrain/
├── harness/                    # NEW: Agent Harness OS layer
│   ├── __init__.py
│   ├── context_manager.py      # Context engineering
│   ├── tool_handler.py         # Tool call handling
│   └── durability_monitor.py   # Model drift detection
│
├── services/
│   ├── brain_service.py        # Main OS orchestrator
│   ├── world_model_integration.py  # Multi-modal fusion (ENHANCED)
│   ├── agent_hope.py           # HOPE Agent application
│   └── teacher_interface.py    # NEW: Separated teacher protocol
│
├── inputs/                     # NEW: Multi-modal input hub
│   ├── __init__.py
│   ├── vision_input.py         # Camera processing
│   ├── audio_input.py          # Speech processing
│   └── text_input.py           # Chat/command processing
│
├── hope_impl/                  # HOPE Brain core
│   ├── brain.py
│   ├── cms.py                  # CMS memory system
│   └── ...
│
└── 03_mamba_brain/             # SSM/Mamba world model
    ├── world_model.py
    └── ...
```

## API Endpoints for Claude Code Teaching

```
# Existing endpoints used by TeacherInterface
POST /api/chat            - Main chat endpoint (routes to HOPE)
POST /api/training/learn  - Learn from correction

# Recommended new endpoints for enhanced teaching
GET  /api/teacher/questions      - Get HOPE's pending questions
POST /api/teacher/answer         - Provide answer to question
POST /api/teacher/correct        - Correct a HOPE response
POST /api/teacher/demonstrate    - Demonstrate an action sequence
GET  /api/teacher/summary        - Get teaching interaction summary
GET  /api/teacher/suggestions    - Get suggested teaching focus areas
POST /api/teacher/validate       - Validate HOPE's knowledge on topic
```

## Summary

The ContinuonBrain architecture naturally aligns with the Agent Harness pattern:

1. **Brain Runtime = Operating System (Harness)**
   - BrainService manages lifecycle, context, tools
   - CMS handles multi-timescale memory
   - Safety Kernel ensures Ring 0 protection

2. **HOPE Agent = Application**
   - HOPEAgent handles user interaction
   - Falls back to Gemma for unknown queries
   - Learns from corrections via TeacherInterface

3. **Claude Code = Teacher**
   - Uses TeacherInterface for guidance
   - Provides answers, corrections, demonstrations
   - Helps identify and fill knowledge gaps

4. **Multi-Modal Inputs = Sensory Stream**
   - WorldModelIntegration fuses vision/audio/text
   - Maintains coherent world state
   - Feeds into HOPE Agent for reasoning
