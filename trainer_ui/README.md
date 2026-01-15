# Brain A Trainer - Web UI

A simple web interface for training your robot brain.

## Features

- **Drive Controls** - WASD keyboard or click controls
- **Robot Arm** - 6-axis joint control with sliders + gripper
- **Camera Feed** - Live video with face recognition
- **Voice** - Speech-to-text input, text-to-speech output
- **Claude Code** - Ask Claude for help with robot tasks
- **Recording** - Save sessions as RLDS episodes for Brain A training

## Quick Start

```bash
cd trainer_ui
pip install fastapi uvicorn opencv-python face-recognition
python server.py
```

Open http://localhost:8000

## Face Recognition Setup

1. Click "Register Face" in the camera panel
2. Look at the camera and enter your name
3. The system will recognize you in future sessions

Face data is stored in `face_db/` as JSON files.

## Controls

| Key | Action |
|-----|--------|
| W | Forward |
| S | Backward |
| A | Turn left |
| D | Turn right |
| Space | Stop |

## Recording for Training

1. Click "Start Recording"
2. Operate the robot (drive, arm, chat with Claude)
3. Click "Stop Recording"
4. Episode is saved to `continuonbrain/rlds/episodes/`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Browser                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Camera    │  │   Drive     │  │    Chat     │         │
│  │  + Faces    │  │  + Arm      │  │  + Voice    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                   │
│                    WebSocket                                 │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │
┌──────────────────────────┼───────────────────────────────────┐
│                    server.py                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Face DB    │  │  Robot      │  │  Training   │          │
│  │  (dlib)     │  │  State      │  │  Recorder   │          │
│  └─────────────┘  └─────────────┘  └──────┬──────┘          │
│                                           │                  │
│                                    RLDS Episodes             │
│                                           │                  │
└───────────────────────────────────────────┼──────────────────┘
                                            │
                                            ▼
                              continuonbrain/rlds/episodes/
                                            │
                                            ▼
                                    Brain A Training
```

## Dependencies

Required:
- Python 3.10+
- fastapi
- uvicorn
- websockets

Optional (for face recognition):
- opencv-python
- face-recognition (requires dlib)
- numpy

## Hardware Integration

The server has placeholder hooks for real hardware:

```python
# In server.py, search for "TODO: Send to actual"

# Motors
motor_controller.set_speed(self.state.drive_left, self.state.drive_right)

# Arm
arm_controller.set_joint(joint, value)
```

Replace these with your actual hardware drivers.

## Files

```
trainer_ui/
├── server.py           # FastAPI server with WebSocket
├── static/
│   └── index.html      # Single-page UI
├── face_db/            # Face recognition database
│   └── *.json          # Per-person face encodings
└── README.md
```

## Troubleshooting

**Camera not working?**
- Check browser permissions (click the camera icon in address bar)
- Try a different browser (Chrome works best)

**Face recognition not working?**
- Install: `pip install face-recognition`
- Requires dlib: `pip install dlib` (may need cmake)

**Claude not responding?**
- Install Claude Code CLI: `npm install -g @anthropic-ai/claude-code`
- Set API key: `export ANTHROPIC_API_KEY=...`
