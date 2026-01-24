# Codebase Consolidation Plan

## Executive Summary

After analyzing the repository structure, I've identified the training and simulation systems have evolved from multiple parallel implementations to a more unified architecture. This document outlines what to keep, migrate, and archive.

---

## Architecture Analysis

### Current Training Pipeline (Active)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     trainer_ui/server.py (UNIFIED)                   │
│    - All training UIs served from one server                         │
│    - Integrates: simulator_training, home_scan, room_scanner,       │
│                  hailo_scanner, oakd_camera, house_api               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 brain_b/simulator/ (Core Training)                   │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────────┐  │
│  │  2D Grid    │   │  3D Home     │   │  3D House (NEW)          │  │
│  │  world.py   │   │  home_world  │   │  house_3d/               │  │
│  │  339 LOC    │   │  1298 LOC    │   │  ~4000 LOC               │  │
│  └─────────────┘   └──────────────┘   └─────────────────────────┘  │
│        │                 │                       │                   │
│  game_handler.py   home_handler.py    training_integration.py       │
│  rlds_logger.py    home_rlds_logger.py                              │
│  training.py       home_training.py                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### World Systems (Keep All - Not Duplicates)

| System | File | LOC | Purpose | Status |
|--------|------|-----|---------|--------|
| 2D Grid | `world.py` | 539 | Simple puzzle training | ACTIVE |
| 3D Home | `home_world.py` | 1298 | Voxel home exploration | ACTIVE |
| 3D House | `house_3d/` | ~4000 | Photorealistic POV training | NEW |

**These are complementary, not duplicates:**
- 2D Grid = Fast iteration on pathfinding/puzzles
- 3D Home = Realistic object interaction training
- 3D House = Photorealistic visual perception training

---

## Files to Archive (Dead Code)

### Standalone Servers (Superseded by trainer_ui)

| File | LOC | Reason | Action |
|------|-----|--------|--------|
| `brain_b/simulator/server.py` | 339 | Superseded by trainer_ui/server.py | ARCHIVE |
| `brain_b/simulator/home_server.py` | 850 | Superseded by trainer_ui/server.py | ARCHIVE |

**Note:** These contain unique endpoints that should be migrated first:

```python
# From server.py - migrate to trainer_ui
"/api/sandbox-audit"  # Sandbox audit trail

# From home_server.py - migrate to trainer_ui
"/api/load_scanned_room"  # Load scanned room
"/api/scanned_levels"     # List scanned levels
"/api/observation"        # Robot observation data
"/api/episodes"           # RLDS episodes list
```

### Game Generators (Both Active)

| File | LOC | Purpose | Status |
|------|-----|---------|--------|
| `training_games.py` | ~800 | Unified curriculum generator | KEEP |
| `home_robot_games.py` | ~1000 | High-fidelity game generator | KEEP |

**Note:** `home_robot_games.py` is actively used by `high_fidelity_trainer.py`.
These have different APIs and serve complementary purposes.

---

## Files to Keep (Active & Critical)

### Core Training (Essential)

```
brain_b/simulator/
├── __init__.py              ✓ KEEP - Unified exports
├── world.py                 ✓ KEEP - 2D grid system
├── home_world.py            ✓ KEEP - 3D home system
├── game_handler.py          ✓ KEEP - 2D game logic
├── home_handler.py          ✓ KEEP - 3D home logic
├── training_games.py        ✓ KEEP - Unified game generator
├── level_generator.py       ✓ KEEP - Procedural 2D levels
├── rlds_logger.py           ✓ KEEP - 2D training data
├── home_rlds_logger.py      ✓ KEEP - 3D training data
├── training.py              ✓ KEEP - 2D model training
├── home_training.py         ✓ KEEP - 3D model training
├── simulator_training.py    ✓ KEEP - Main trainer entry
├── perception_system.py     ✓ KEEP - Visual perception
├── high_fidelity_trainer.py ✓ KEEP - Advanced training
├── perception_trainer.py    ✓ KEEP - Perception training
├── expert_navigation.py     ✓ KEEP - Navigation model
├── semantic_search.py       ✓ KEEP - State search
├── generate_training_data.py ✓ KEEP - Data generation
└── house_3d/                ✓ KEEP - New 3D renderer
```

### Trainer UI (Essential)

```
trainer_ui/
├── server.py              ✓ KEEP - Main unified server
├── home_scan.py           ✓ KEEP - Home scanning
├── room_scanner.py        ✓ KEEP - Room to 3D pipeline
├── hailo_scanner.py       ✓ KEEP - NPU acceleration
├── oakd_camera.py         ✓ KEEP - Depth camera
├── brain_b_integration.py ✓ KEEP - Brain B connection
├── house_api.py           ✓ KEEP - 3D house API
├── simulator_utils.py     ✓ KEEP - Simulator helpers
└── static/
    ├── index.html         ✓ KEEP - Main trainer UI
    ├── homescan.html      ✓ KEEP - Home scan UI
    ├── home_explore.html  ✓ KEEP - 3D exploration
    ├── room_scanner.html  ✓ KEEP - Room scanner UI
    └── house_viewer.html  ✓ KEEP - 3D house POV
```

---

## Migration Steps

### Step 1: Migrate Unique Endpoints

Before archiving old servers, migrate these to `trainer_ui/server.py`:

```python
# Add to trainer_ui/server.py

# From server.py
@app.get("/api/sandbox-audit")
async def get_sandbox_audit():
    """Return sandbox audit trail."""
    # Migrate implementation from brain_b/simulator/server.py:228

# From home_server.py
@app.get("/api/observation")
async def get_observation():
    """Return robot observation data for training."""
    # Migrate from brain_b/simulator/home_server.py:384
```

### Step 2: Archive Old Servers

```bash
# Create archive directory
mkdir -p brain_b/simulator/_archive

# Move deprecated servers
mv brain_b/simulator/server.py brain_b/simulator/_archive/
mv brain_b/simulator/home_server.py brain_b/simulator/_archive/
mv brain_b/simulator/home_robot_games.py brain_b/simulator/_archive/

# Remove from __init__.py exports (none exported currently)
```

### Step 3: Update Static Files

The 2D grid game UI needs migration:

```bash
# Move 2D game UI to trainer_ui
mv brain_b/simulator/static/index.html trainer_ui/static/robotgrid.html

# Update trainer_ui navigation to include it
# Add route: /robotgrid → trainer_ui/static/robotgrid.html
```

### Step 4: Consolidate Game Generators

Keep `training_games.py` as the single entry point. Verify it includes all tiers from `home_robot_games.py`:

```python
# training_games.py already has:
GameType.NAVIGATION   # Tier 1 - Basic Movement
GameType.PUZZLE       # Tier 2 - Object Interaction
GameType.EXPLORATION  # Tier 3 - Multi-Room Navigation
GameType.INTERACTION  # Tier 4 - Task Completion
GameType.MULTI_OBJECTIVE  # Tier 5 - Complex Scenarios
```

---

## Savings Summary

| Category | Before | After | Saved |
|----------|--------|-------|-------|
| Standalone servers | 1189 LOC | 0 LOC | 1189 LOC |
| **Total** | | | **~1200 LOC** |

**Files archived:** 2 files (server.py, home_server.py)
**Migrated:** 2D RobotGrid UI to trainer_ui, unique endpoints preserved
**Restored:** home_robot_games.py (actively used by high_fidelity_trainer.py)

---

## Unified Training Entry Points

After consolidation, these are the main entry points:

### Web Training
```bash
# Start unified trainer server
python trainer_ui/server.py
# Open http://localhost:8000

# Available pages:
# /                 - Main trainer (arms, camera, voice)
# /homescan         - 3D home scanning
# /home-explore     - 3D home exploration game
# /room-scanner     - Room image to 3D pipeline
# /house-viewer     - Photorealistic 3D POV training
# /robotgrid        - 2D grid puzzle game (after migration)
```

### Headless Training
```bash
# Generate training data
python -m brain_b.simulator.generate_training_data

# Run training
python scripts/compound/training_with_inference.py

# Visual 3D training
python scripts/compound/training_with_inference.py --visual

# Learning partner daemon
python scripts/compound/learning_partner.py --train
```

---

## Verification Commands

After consolidation, verify all systems work:

```bash
# 1. Trainer UI starts
python trainer_ui/server.py &
sleep 2
curl -s http://localhost:8000/api/status | jq .
kill %1

# 2. 2D world works
python -c "from brain_b.simulator import GridWorld, load_level; w = load_level('tutorial'); print(f'2D world: {w.width}x{w.height}')"

# 3. 3D home world works
python -c "from brain_b.simulator import HomeWorld, get_home_level; w = get_home_level('simple_apartment'); print(f'3D home: {len(w.rooms)} rooms')"

# 4. 3D house renderer works
python -c "from brain_b.simulator.house_3d import HouseScene; s = HouseScene.from_template('studio_apartment'); print(f'3D house: {len(s.rooms)} rooms')"

# 5. Training games generate
python -c "from brain_b.simulator.training_games import TrainingGamesGenerator; g = TrainingGamesGenerator(); print(f'Games: {len(g.game_types)} types')"
```

---

## Notes

1. **No breaking changes** - All active imports continue to work
2. **Gradual migration** - Archive files still exist for reference
3. **Single server** - trainer_ui/server.py handles all web training
4. **Three world systems** - 2D, 3D voxel, 3D photorealistic coexist

The architecture is clean - the apparent duplication is actually a layered training system with increasing visual fidelity.
