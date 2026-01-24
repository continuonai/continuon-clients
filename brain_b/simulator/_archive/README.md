# Archived Files

These files have been superseded by the unified `trainer_ui/server.py` implementation.

## Archived Date
2026-01-23

## Files

### server.py (339 LOC)
- **Purpose**: Standalone 2D RobotGrid web server with WebSocket
- **Superseded by**: `trainer_ui/server.py`
- **Unique endpoints migrated**:
  - `/api/sandbox-audit` -> `trainer_ui/server.py`

### home_server.py (850 LOC)
- **Purpose**: Standalone 3D Home exploration web server
- **Superseded by**: `trainer_ui/server.py`
- **Unique endpoints migrated**:
  - `/api/scanned_levels` -> `trainer_ui/server.py`
  - `/api/episodes` -> `trainer_ui/server.py`
  - `/api/observation` -> `trainer_ui/server.py`

### home_robot_games.py - NOT ARCHIVED
- **Status**: RESTORED - actively used by `high_fidelity_trainer.py`
- **Exports**: HomeRobotGameGenerator, GameState, ActionType, ObjectType, RoomType, TaskType
- **Note**: Different API than training_games.py, kept for compatibility

### static/index.html
- **Purpose**: 2D RobotGrid game UI
- **Migrated to**: `trainer_ui/static/robotgrid.html`
- **New route**: `/robotgrid` in trainer_ui

## Restoration

If you need to restore these files, they can be moved back:

```bash
mv brain_b/simulator/_archive/server.py brain_b/simulator/server.py
mv brain_b/simulator/_archive/home_server.py brain_b/simulator/home_server.py
mv brain_b/simulator/_archive/home_robot_games.py brain_b/simulator/home_robot_games.py
mv brain_b/simulator/_archive/static brain_b/simulator/static
```

## Why Archived?

1. **Consolidation**: trainer_ui/server.py serves as the unified web server for all training UIs
2. **Reduced maintenance**: Single server to maintain instead of three
3. **Feature parity**: All unique functionality has been migrated
4. **Better integration**: trainer_ui has hardware detection, Brain B integration, etc.
