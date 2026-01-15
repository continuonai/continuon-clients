# RobotGrid Multi-Level Game Design

## Overview

This document outlines the enhancement of RobotGrid into a multi-level game system that:
1. Generates RLDS episodes for Brain A training
2. Implements world modeling with surprise metrics
3. Provides semantic search over game state history
4. Enables next-action prediction training

## Alignment with RLDS Slow Loop Training

### Three-Timescale Integration

| Timescale | Game Component | Brain Integration |
|-----------|----------------|-------------------|
| **Fast (τ=10ms)** | Collision detection, sandbox denial | Safety reflex patterns |
| **Mid (τ=100ms)** | Action execution, state updates | Working memory context |
| **Slow (τ=1s+)** | Episode export, world model training | Cloud-based RLDS training |

### RLDS Episode Schema for RobotGrid

```json
{
  "schema_version": "1.1",
  "episode_id": "robotgrid_<timestamp>_<uuid>",
  "robot_id": "simulator_v1",
  "robot_model": "RobotGrid/GridBot",
  "capabilities": ["navigation", "manipulation", "planning"],

  "metadata": {
    "level_id": "lava_maze",
    "level_difficulty": 3,
    "initial_state": { "robot": {...}, "grid": [...] },
    "final_state": { ... },
    "success": true,
    "total_moves": 42,
    "sandbox_denials": 2,
    "behaviors_used": ["scout", "fetch_key"]
  },

  "steps": [
    {
      "frame_id": 0,
      "timestamp_us": 1234567890,
      "observation": {
        "robot_state": { "x": 1, "y": 1, "direction": "EAST", "inventory": [] },
        "visible_tiles": [...],
        "look_ahead": { "tile": "FLOOR", "distance": 3 }
      },
      "action": {
        "command": "forward",
        "intent": "MOVE_FORWARD",
        "params": {}
      },
      "world_model": {
        "predicted_state": { "x": 2, "y": 1, ... },
        "actual_state": { "x": 2, "y": 1, ... },
        "surprise": 0.0,
        "belief_confidence": 0.95
      },
      "reward": 1.0,
      "done": false
    }
  ]
}
```

## World Model Architecture

### Predictor Interface

```python
class GridWorldModel:
    """Predicts next game state given current state and action."""

    def predict(self, state: WorldState, action: Action) -> Prediction:
        """
        Returns:
            predicted_state: Expected next state
            uncertainty: Model confidence (0-1)
            surprise: Actual vs predicted divergence (computed post-hoc)
        """

    def encode_state(self, world: GridWorld) -> LatentState:
        """Convert grid world to latent representation for CMS storage."""

    def decode_state(self, latent: LatentState) -> WorldState:
        """Reconstruct world state from latent tokens."""
```

### Wave-Particle Duality in Game Context

**Wave Path (Global Planning):**
- Encodes entire grid as spectral state
- Long-range path planning
- Level completion strategies

**Particle Path (Local Reactions):**
- Immediate collision detection
- Sandbox gate responses
- Item pickup reactions

### Surprise Metrics

```python
surprise = 1.0 - cosine_similarity(predicted_state, actual_state)
```

High surprise indicates:
- Unexpected sandbox denial
- Hidden mechanics (buttons, boxes)
- Novel level configurations

## Semantic Search System

### State Embedding

```python
class StateEmbedder:
    """Embeds game states for semantic similarity search."""

    def embed(self, state: WorldState) -> np.ndarray:
        """
        Encodes:
        - Robot position and direction (normalized)
        - Inventory contents (one-hot)
        - Visible tile patterns (conv features)
        - Level progress (goal distance)
        """

    def search(self, query: WorldState, history: List[WorldState], k: int = 5) -> List[Match]:
        """Find k most similar past states."""
```

### Use Cases

1. **Similar Situations:** "I've been stuck here before"
2. **Strategy Recall:** "What worked last time?"
3. **Anomaly Detection:** "This state is unusual"
4. **Curriculum Learning:** "Start from similar solved states"

## Next-Action Prediction

### Training Loop

```
For each episode in RLDS:
    For each (state, action, next_state) triple:
        predicted_action = model.predict_action(state)
        loss = cross_entropy(predicted_action, actual_action)

        predicted_state = world_model.predict(state, action)
        surprise = compute_surprise(predicted_state, next_state)

        Log metrics: {loss, surprise, accuracy}
```

### Action Space

```python
ACTIONS = [
    "forward",    # Move in facing direction
    "backward",   # Move opposite to facing
    "left",       # Turn 90° left
    "right",      # Turn 90° right
    "look",       # Observe ahead (no state change)
    "wait",       # Do nothing (useful for timing puzzles)
]
```

## Progressive Difficulty Levels

### Level Tiers

| Tier | Levels | New Mechanics | Brain Capabilities Tested |
|------|--------|--------------|--------------------------|
| 1 - Tutorial | 1-3 | Basic movement | Intent classification |
| 2 - Keys | 4-6 | Keys, doors | State tracking, planning |
| 3 - Hazards | 7-10 | Lava, sandbox | Safety gates, risk assessment |
| 4 - Puzzles | 11-15 | Boxes, buttons | Multi-step planning |
| 5 - Challenge | 16-20 | Combined, timed | Full capability integration |

### Procedural Generation

```python
class LevelGenerator:
    """Generates levels with controlled difficulty."""

    def generate(self, difficulty: int, seed: int = None) -> GridWorld:
        """
        Parameters:
            difficulty: 1-20 (controls complexity)
            seed: Random seed for reproducibility

        Returns:
            New level with guaranteed solution
        """
```

### Curriculum Learning Integration

1. Start with Tier 1 levels
2. Advance when success rate > 80%
3. Regress if failure rate > 50%
4. Mix difficulties for robustness

## CMS Memory Integration

### Three-Level Mapping

| CMS Level | Game Context | Decay | Slots |
|-----------|-------------|-------|-------|
| 0 (Episodic) | Current game session | 0.9 | 64 |
| 1 (Working) | Level strategies | 0.99 | 128 |
| 2 (Semantic) | General game knowledge | 0.999 | 256 |

### Memory Operations

**Write (on action):**
```python
cms.write(level=0, key=state_embedding, value=action_taken)
```

**Read (for prediction):**
```python
context = cms.read(query=current_state_embedding, levels=[0, 1, 2])
predicted_action = policy(current_state, context)
```

**Consolidate (on level complete):**
```python
cms.consolidate(source_level=0, target_level=1, threshold=0.8)
```

## Implementation Phases

### Phase 1: RLDS Export (This PR)
- [ ] Episode logger for game sessions
- [ ] Step-by-step observation/action recording
- [ ] Metadata with level info and success metrics
- [ ] Export to JSONL format

### Phase 2: World Model
- [ ] State encoder/decoder
- [ ] Simple predictor (rule-based initially)
- [ ] Surprise metric computation
- [ ] Integration with game loop

### Phase 3: Semantic Search
- [ ] State embedding model
- [ ] Vector store for history
- [ ] Similarity search API
- [ ] UI for exploring similar states

### Phase 4: Training Loop
- [ ] Action prediction model
- [ ] Training script using RLDS episodes
- [ ] Evaluation metrics
- [ ] Model checkpoint management

### Phase 5: Advanced Levels
- [ ] Procedural level generator
- [ ] Difficulty progression system
- [ ] Curriculum learning integration
- [ ] Multi-agent scenarios

## API Extensions

```python
# New endpoints for training integration

GET  /api/episodes              # List recorded episodes
GET  /api/episodes/{id}         # Get episode details
POST /api/episodes/export       # Export to RLDS format

GET  /api/world-model/predict   # Predict next state
GET  /api/world-model/surprise  # Get surprise for last action

GET  /api/search/similar        # Find similar past states
GET  /api/search/strategies     # Find successful strategies

POST /api/train/step            # Run one training step
GET  /api/train/metrics         # Get training metrics
```

## Success Criteria

1. **RLDS Compatibility:** Episodes pass `rlds-acceptance-checklist`
2. **World Model Accuracy:** Surprise < 0.1 for 90% of actions
3. **Search Relevance:** Top-5 similar states include solution hints
4. **Prediction Accuracy:** > 70% on held-out episodes
5. **Training Integration:** Episodes usable by Brain A trainer
