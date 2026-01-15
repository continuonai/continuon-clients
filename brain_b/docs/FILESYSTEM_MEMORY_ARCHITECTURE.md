# Filesystem Memory Architecture for Brain B

## Overview

This document maps the filesystem-based memory hierarchy to Brain B's architecture,
aligning with the CMS (Continuous Memory System) three-level model.

```
Memory = Filesystem
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Procedural Memory (Blue)                            │   │
│  │  → AGENTS.md     (agent behavior definitions)       │   │
│  │  → mcp.json      (tool/capability configurations)   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Semantic Memory (Green)                             │   │
│  │  → skills/                                          │   │
│  │     └─ foo/SKILL.md       (skill definitions)       │   │
│  │     └─ bar/knowledge1.md  (domain knowledge)        │   │
│  │     └─ bar/knowledge2.md  (domain knowledge)        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Episodic Memory (Red)                               │   │
│  │  → conversations/                                   │   │
│  │     └─ 01_10_2025.json   (session transcript)       │   │
│  │     └─ 02_10_2025.json   (session transcript)       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## CMS Level Mapping

| CMS Level | Memory Type | Decay | Brain B Filesystem | Purpose |
|-----------|-------------|-------|-------------------|---------|
| **L2** | Procedural | 0.999 | `procedural/` | How to act (stable skills) |
| **L1** | Semantic | 0.99 | `semantic/` | What things mean (knowledge) |
| **L0** | Episodic | 0.9 | `episodic/` | What happened (sessions) |

### Decay Interpretation for Filesystem

- **Procedural (decay=0.999)**: Rarely changes, persists across all sessions
- **Semantic (decay=0.99)**: Accumulates slowly, updated when learning
- **Episodic (decay=0.9)**: Decays quickly, only recent sessions retained

## Brain B Directory Structure

```
brain_b_data/
├── procedural/                 # L2: Stable, rarely-changing
│   ├── agents.md               # Agent behavior definitions
│   ├── capabilities.json       # Available tools/actions
│   ├── guardrails.json         # Safety constraints
│   └── sandbox_rules.json      # Permission policies
│
├── semantic/                   # L1: Accumulated knowledge
│   ├── skills/                 # Learned skills
│   │   ├── navigation/
│   │   │   └── SKILL.md        # Navigation skill definition
│   │   ├── manipulation/
│   │   │   └── SKILL.md        # Object manipulation skill
│   │   └── communication/
│   │       └── SKILL.md        # Dialog patterns
│   │
│   ├── knowledge/              # Domain knowledge
│   │   ├── objects.md          # Known object types
│   │   ├── locations.md        # Known places
│   │   └── patterns.md         # Recognized patterns
│   │
│   └── behaviors/              # Taught behaviors
│       └── behaviors.json      # From teaching system
│
├── episodic/                   # L0: Session-specific, decaying
│   ├── conversations/          # Chat transcripts
│   │   ├── 2025-01-15_session1.json
│   │   └── 2025-01-15_session2.json
│   │
│   ├── events/                 # Event logs
│   │   └── events.jsonl        # All events (append-only)
│   │
│   ├── rlds_episodes/          # Training episodes
│   │   ├── episode_001/
│   │   │   ├── metadata.json
│   │   │   └── steps.jsonl
│   │   └── episode_002/
│   │
│   └── checkpoints/            # State snapshots
│       ├── ckpt_latest.json
│       └── ckpt_backup.json
│
└── session_state.json          # Current session metadata
```

## Memory Operations

### Read Operation (Query)

```python
def read_memory(query: str, levels: List[str] = ["episodic", "semantic", "procedural"]) -> Context:
    """
    Read from filesystem memory with hierarchical mixing.

    1. Search each level for relevant files
    2. Score by relevance (semantic similarity)
    3. Mix with level-specific weights
    """
    contexts = {}

    for level in levels:
        level_path = data_dir / level
        matches = search_files(level_path, query)
        contexts[level] = aggregate(matches)

    # Hierarchical mixing (procedural overrides semantic overrides episodic)
    return mix_contexts(contexts, weights=LEVEL_WEIGHTS)
```

### Write Operation

```python
def write_memory(content: dict, level: str, key: str) -> None:
    """
    Write to filesystem memory at appropriate level.

    - Procedural: Only on explicit skill learning
    - Semantic: On knowledge consolidation
    - Episodic: On every event (append-only)
    """
    level_path = data_dir / level

    if level == "episodic":
        # Append to event log
        append_jsonl(level_path / "events.jsonl", content)
    elif level == "semantic":
        # Update knowledge file
        update_or_create(level_path / "knowledge" / f"{key}.md", content)
    elif level == "procedural":
        # Update skill definition (requires explicit confirmation)
        if confirm_procedural_update(content):
            update_skill(level_path / "skills" / key / "SKILL.md", content)
```

### Consolidation (Episodic → Semantic → Procedural)

```python
def consolidate_memory() -> None:
    """
    Consolidate memories up the hierarchy (like sleep).

    Called periodically or on session end:
    1. Aggregate episodic events into patterns
    2. Update semantic knowledge with patterns
    3. (Rarely) Promote stable patterns to procedural
    """
    # Episodic → Semantic
    recent_episodes = load_recent_episodes(max_age_days=7)
    patterns = extract_patterns(recent_episodes)

    for pattern in patterns:
        if pattern.frequency > SEMANTIC_THRESHOLD:
            write_memory(pattern, "semantic", pattern.key)

    # Semantic → Procedural (very rare)
    stable_knowledge = find_stable_knowledge(min_age_days=30)
    for knowledge in stable_knowledge:
        if knowledge.consistency > PROCEDURAL_THRESHOLD:
            propose_procedural_update(knowledge)  # Requires human approval
```

## Integration with RobotGrid Simulator

The filesystem memory integrates with the game simulator:

```
Game Session
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Episodic (L0): Game events, moves, RLDS episodes            │
│   → episodic/rlds_episodes/robotgrid_*.json                 │
│   → episodic/events/game_events.jsonl                       │
└────────────────────────┬────────────────────────────────────┘
                         │ consolidation
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Semantic (L1): Level strategies, successful patterns        │
│   → semantic/knowledge/level_strategies.md                  │
│   → semantic/behaviors/behaviors.json (taught behaviors)    │
└────────────────────────┬────────────────────────────────────┘
                         │ consolidation (rare)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Procedural (L2): Game rules, action definitions             │
│   → procedural/capabilities.json (available actions)        │
│   → procedural/skills/navigation/SKILL.md                   │
└─────────────────────────────────────────────────────────────┘
```

## File Formats

### SKILL.md (Procedural/Semantic)

```markdown
# Navigation Skill

## Capability
Move robot through grid world avoiding obstacles.

## Actions
- forward: Move one tile in facing direction
- backward: Move one tile opposite to facing
- left: Turn 90° counter-clockwise
- right: Turn 90° clockwise

## Constraints
- Cannot enter WALL tiles
- Cannot enter LAVA tiles (sandbox denied)
- Must have KEY to pass DOOR

## Patterns
- Wall-following: Turn right when blocked, then forward
- Key-seeking: Navigate to nearest KEY before DOOR
```

### Conversation JSON (Episodic)

```json
{
  "session_id": "2025-01-15_session1",
  "start_time": "2025-01-15T10:30:00Z",
  "end_time": "2025-01-15T11:45:00Z",
  "turns": [
    {
      "timestamp": "2025-01-15T10:30:15Z",
      "role": "user",
      "content": "teach patrol",
      "intent": "START_TEACHING"
    },
    {
      "timestamp": "2025-01-15T10:30:16Z",
      "role": "assistant",
      "content": "Ready to learn 'patrol'. Show me what to do.",
      "action": "teach_start"
    }
  ],
  "behaviors_taught": ["patrol"],
  "sandbox_denials": 0,
  "success": true
}
```

### Knowledge MD (Semantic)

```markdown
# Level Strategies

## Tutorial Levels (1-3)
- Direct path to goal
- No obstacles require strategy

## Key/Door Levels (4-6)
- Always collect KEY before approaching DOOR
- Pattern: Explore periphery first for keys

## Lava Levels (7-10)
- Maintain 1-tile buffer from LAVA
- If stuck, backtrack and try alternate path
- Success rate: 85% with buffer strategy

## Last Updated
2025-01-15 (from 47 episodes)
```

## Decay Implementation

Files don't literally decay, but we simulate decay through:

1. **Episodic**: Auto-archive after N days, limit to M recent files
2. **Semantic**: Version with timestamps, weight recent updates higher
3. **Procedural**: Require explicit human approval to modify

```python
RETENTION_POLICY = {
    "episodic": {
        "conversations": {"max_age_days": 30, "max_files": 100},
        "events": {"max_age_days": 7, "max_size_mb": 100},
        "rlds_episodes": {"max_age_days": 90, "max_files": 1000},
    },
    "semantic": {
        "knowledge": {"version_history": 10},
        "behaviors": {"no_auto_delete": True},
    },
    "procedural": {
        "all": {"require_approval": True, "no_auto_delete": True},
    },
}
```

## API Endpoints

```python
# Memory query
GET  /api/memory/search?q=<query>&levels=episodic,semantic
GET  /api/memory/read/<level>/<path>

# Memory write
POST /api/memory/write/<level>/<path>
POST /api/memory/consolidate

# Memory stats
GET  /api/memory/stats
GET  /api/memory/retention
```

## Benefits of Filesystem Memory

1. **Inspectable**: Human-readable markdown and JSON files
2. **Debuggable**: Easy to examine memory state
3. **Portable**: Copy directory to transfer knowledge
4. **Versionable**: Git-compatible for tracking changes
5. **Familiar**: Standard filesystem semantics

## Alignment with Brain A Training

The filesystem memory generates RLDS-compatible episodes:

```
episodic/rlds_episodes/  →  Brain A slow loop training
semantic/behaviors/      →  Seed knowledge for new robots
procedural/skills/       →  Universal skill definitions
```

This enables the training loop:
1. Brain B generates episodic memories (RLDS)
2. Brain A trains on curated episodes (slow loop)
3. Trained weights update procedural skills
4. Skills propagate to all Brain B instances
