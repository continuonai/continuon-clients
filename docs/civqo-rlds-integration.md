# Civqo RLDS Integration

## Overview

ContinuonBrain can import RLDS episodes from Civqo agent sessions for slow loop training. This integration enables:

- **Training Data from Real Sessions**: Agent interactions captured in Civqo's Code City visualization
- **Tool Call Learning**: Claude Code tool usage patterns (file operations, bash commands, etc.)
- **Decision Traces**: Agent reasoning and confidence for each action
- **Anonymized PII**: All sensitive data hashed before import

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CIVQO                                    │
├─────────────────────────────────────────────────────────────────┤
│  Agent Sessions → Trajectory Events → RLDS Export API           │
│                                                                  │
│  packages/harness/src/rlds/                                      │
│  ├── types.ts        # RLDS type definitions                    │
│  ├── transformer.ts  # Event → Step conversion                  │
│  ├── exporter.ts     # Session export logic                     │
│  ├── anonymizer.ts   # PII removal                              │
│  └── validator.ts    # Schema validation                        │
│                                                                  │
│  apps/api/src/routes/rlds-export.ts                             │
│  ├── GET  /sessions   # List eligible sessions                  │
│  ├── POST /session/:id # Export single session                  │
│  ├── POST /batch      # Batch export                            │
│  └── GET  /stats      # Export statistics                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS / JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CONTINUONBRAIN                               │
├─────────────────────────────────────────────────────────────────┤
│  continuonbrain/rlds/civqo_importer.py                          │
│  ├── fetch_episodes_from_api()  # Direct API import             │
│  ├── load_episodes_from_files() # File-based import             │
│  ├── normalize_episode()        # Schema normalization          │
│  └── write_episode()            # Directory format output       │
│                                                                  │
│  Output: continuonbrain/rlds/episodes/                          │
│  ├── civqo_session_xxx/                                         │
│  │   ├── metadata.json                                          │
│  │   └── steps/                                                  │
│  │       └── 000000.jsonl                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

### Import from Civqo API

```bash
# Set API key
export CIVQO_API_KEY="your_api_key"

# Import all eligible sessions
python -m continuonbrain.rlds.civqo_importer --api-key $CIVQO_API_KEY

# Import specific sessions
python -m continuonbrain.rlds.civqo_importer \
  --api-key $CIVQO_API_KEY \
  --session-ids session_123,session_456

# Limit episodes
python -m continuonbrain.rlds.civqo_importer \
  --api-key $CIVQO_API_KEY \
  --max-episodes 100 \
  --max-steps 500
```

### Import from Pre-Exported Files

```bash
# Import from directory containing JSON/JSONL exports
python -m continuonbrain.rlds.civqo_importer --input-dir ./civqo_exports/
```

### Programmatic Usage

```python
from continuonbrain.rlds import CivqoImportConfig, import_civqo_episodes

config = CivqoImportConfig(
    api_key="your_api_key",
    output_dir=Path("./episodes"),
    max_episodes=50,
    max_steps_per_episode=200,
)

written_paths = import_civqo_episodes(config)
print(f"Imported {len(written_paths)} episodes")
```

## Episode Structure

### Metadata

```json
{
  "episode_id": "civqo_session_abc123",
  "environment_id": "civqo:workspace_xyz",
  "xr_mode": "trainer",
  "control_role": "human_supervisor",
  "tags": [
    "origin:civqo",
    "outcome:completed",
    "tools:file_read,bash,file_write"
  ],
  "software": {
    "xr_app": "civqo.com",
    "continuonbrain_os": "continuonbrain-civqo-importer",
    "glove_firmware": "absent"
  },
  "schema_version": "1.1"
}
```

### Step Structure

Each step represents a trajectory event (tool call, user input, or assistant output):

```json
{
  "observation": {
    "headset_pose": { "position": [0,0,0], "orientation_quat": [0,0,0,1], "valid": false },
    "right_hand_pose": { "position": [0,0,0], "orientation_quat": [0,0,0,1], "valid": false },
    "left_hand_pose": { "position": [0,0,0], "orientation_quat": [0,0,0,1], "valid": false },
    "gaze": { "origin": [0,0,0], "direction": [0,0,1], "confidence": 0 },
    "robot_state": {
      "timestamp_nanos": 1704067200000000000,
      "joint_positions": [0,0],
      "joint_velocities": [0,0],
      "frame_id": "civqo_session_abc_000001"
    },
    "glove": { "flex": [0,0,0,0,0], "fsr": [0,0,0,0,0,0,0,0], "valid": false },
    "video_frame_id": "civqo_session_abc_000001",
    "depth_frame_id": "civqo_session_abc_000001",
    "diagnostics": { "latency_ms": 150 },
    "command": "Read the file src/index.ts",
    "ui_context": { "active_panel": "chat" }
  },
  "action": {
    "command": [1.0, 1.0],
    "source": "agent",
    "annotation": {
      "kind": "tool_call",
      "fields": {
        "tool_name": "file_read",
        "success": "true"
      }
    },
    "tool_calls": [
      {
        "name": "file_read",
        "parameters": { "path": "src/index.ts" },
        "result": "...",
        "success": true
      }
    ]
  },
  "is_terminal": false,
  "step_metadata": {
    "source_session": "session_abc123",
    "source_workspace": "workspace_xyz",
    "event_type": "tool_call"
  }
}
```

## XR Placeholder Strategy

Since Civqo sessions don't have XR hardware signals, we use schema-stable placeholders:

| Field | Placeholder Value | Notes |
|-------|-------------------|-------|
| `headset_pose` | Zero position, identity quat, `valid: false` | No VR headset |
| `hand_poses` | Zero position, identity quat, `valid: false` | No hand tracking |
| `gaze` | Origin zero, direction [0,0,1], confidence 0 | No eye tracking |
| `robot_state` | Zero joints, identity pose | No robot arm |
| `glove` | All zeros, `valid: false` | No haptic glove |

This allows episodes to pass validation while preserving the rich tool call and decision trace data.

## Export Eligibility

Sessions must meet criteria to be eligible for export:

- **Minimum Events**: At least 10 trajectory events
- **Completion Status**: `completed` or `failed` (not `active` or `abandoned`)
- **Successful Actions**: At least one successful tool call

## Anonymization

All episodes are anonymized before export:

- **User IDs**: SHA256 hashed with salt
- **Email addresses**: Dropped entirely
- **API keys**: Pattern-matched and redacted
- **Message content**: Optionally hashed
- **Metadata tags**: User-identifying tags hashed

## Future Integration

- [ ] Wire into Ralph slow loop for automated training
- [ ] R2 storage integration for episode archival
- [ ] Real-time streaming export for active sessions
- [ ] Episode quality scoring for training set curation
