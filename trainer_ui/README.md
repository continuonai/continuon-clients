# ContinuonBrain Trainer - Web UI

A simple web interface for training your robot brain with hardware auto-detection and dual SO-ARM101 support.

## Features

- **Drive Controls** - WASD keyboard or click controls
- **Dual Robot Arms** - SO-ARM101 support with 5 joints + gripper per arm
- **Hardware Auto-Detection** - Automatically detects I2C arms, Hailo AI accelerator, audio
- **Camera Feed** - Live video with face recognition
- **Voice** - Speech-to-text input, text-to-speech output (espeak-ng)
- **Claude Code** - Ask Claude for help with robot tasks
- **Recording** - Save sessions as RLDS episodes for ContinuonBrain training

## Quick Start

```bash
cd trainer_ui
pip install fastapi uvicorn opencv-python face-recognition

# With real hardware
python server.py

# Mock mode (no hardware needed)
TRAINER_MOCK_HARDWARE=1 python server.py
```

Open http://localhost:8000

## Hardware Auto-Detection

On startup, the server automatically detects:

| Hardware | Detection Method |
|----------|------------------|
| **SO-ARM101 Arms** | I2C scan for PCA9685 at 0x40, 0x41 |
| **Hailo-8/8L** | lspci for device ID 1e60:2864/2862 |
| **Audio** | Check for espeak-ng, espeak, or say |
| **Cameras** | V4L2 device enumeration |

### Hardware Status Panel

The UI shows real-time hardware status in the header:

```
🧠 Hailo: 26 TOPS    🦾 Arm L: OK    🦾 Arm R: OK    🔊 Audio: espeak-ng
```

- **Green** = Real hardware detected
- **Yellow** = Mock mode
- **Gray** = Not available

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRAINER_MOCK_HARDWARE` | `0` | Set to `1` to force mock mode |
| `TRAINER_ARM_0_ADDRESS` | `0x40` | I2C address for left arm |
| `TRAINER_ARM_1_ADDRESS` | `0x41` | I2C address for right arm |

## Dual Arm Support

The UI supports controlling two SO-ARM101 arms simultaneously:

1. **Arm Tabs** - Click "Left Arm" or "Right Arm" to switch
2. **Joint Sliders** - J1-J5 control arm joints, G controls gripper
3. **Independent Control** - Each arm maintains its own state

### WebSocket Messages

Arm commands include `arm_id` to specify which arm:

```json
{"type": "arm", "arm_id": "arm_0", "joint": 2, "value": 0.5}
{"type": "gripper", "arm_id": "arm_1", "value": 0.75}
```

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

## Health Endpoint

Check server and hardware status:

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "ok",
  "hardware": {
    "arms": ["arm_0", "arm_1"],
    "arm_count": 2,
    "hailo": {"available": true, "model": "Hailo-8", "tops": 26.0},
    "audio": {"available": true, "backend": "espeak-ng"},
    "cameras": ["Webcam"],
    "is_mock": false
  }
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Browser                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Camera    │  │   Drive     │  │    Chat     │         │
│  │  + Faces    │  │  + Arms     │  │  + Voice    │         │
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
│  │  Hardware   │  │  Dual Arm   │  │  Training   │          │
│  │  Detector   │  │  Manager    │  │  Recorder   │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         │         ┌──────┴──────┐         │                  │
│         │         │  PCA9685    │         │                  │
│         │         │  (I2C)      │         │                  │
│         │         └─────────────┘         │                  │
│         │                                 │                  │
│         └─────────────────────────────────┤                  │
│                                           │                  │
│                                    RLDS Episodes             │
│                                           │                  │
└───────────────────────────────────────────┼──────────────────┘
                                            │
                                            ▼
                              continuonbrain/rlds/episodes/
                                            │
                                            ▼
                                  ContinuonBrain Training
```

## Files

```
trainer_ui/
├── server.py           # FastAPI server with WebSocket
├── hardware/           # Hardware management
│   ├── __init__.py     # Package exports
│   ├── detector.py     # Auto-detection for I2C, Hailo, audio
│   ├── arm_manager.py  # DualArmManager + ArmController
│   └── audio_manager.py # TTS wrapper (espeak-ng)
├── static/
│   └── index.html      # Single-page UI
├── face_db/            # Face recognition database
│   └── *.json          # Per-person face encodings
└── README.md
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

Optional (for real hardware):
- adafruit-servokit (for PCA9685 arm control)
- smbus2 (for I2C)

## Troubleshooting

**No arms detected?**
- Check I2C is enabled: `sudo raspi-config` → Interface Options → I2C
- Verify PCA9685 at 0x40: `i2cdetect -y 1`
- Use mock mode for testing: `TRAINER_MOCK_HARDWARE=1`

**Camera not working?**
- Check browser permissions (click the camera icon in address bar)
- Try a different browser (Chrome works best)

**Face recognition not working?**
- Install: `pip install face-recognition`
- Requires dlib: `pip install dlib` (may need cmake)

**Audio not working?**
- Install espeak-ng: `sudo apt install espeak-ng`
- Or espeak: `sudo apt install espeak`

**Claude not responding?**
- Install Claude Code CLI: `npm install -g @anthropic-ai/claude-code`
- Set API key: `export ANTHROPIC_API_KEY=...`

**Port already in use?**
- Kill existing process: `fuser -k 8000/tcp`
- Or use different port: `uvicorn server:app --port 9000`
