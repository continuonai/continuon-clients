# Robot Wake-Up & Control System

ContinuonBrain includes a complete wake-up orchestration system that automatically starts services, enables LAN discovery for iPhone/web control, and manages robot operational modes for training and autonomous operation.

## Overview

When the robot wakes from sleep, it automatically:
1. **Runs health checks** - Validates hardware and software
2. **Starts LAN discovery** - Broadcasts presence for iPhone/web browser
3. **Launches Robot API server** - Enables Flutter app control
4. **Initializes mode manager** - Sets operational mode (training/autonomous/idle)

## Quick Start

### Wake Robot with Services

```bash
# Wake robot and start all services
PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py

# Output:
# 🚀 ContinuonBrain Startup
# 🏥 Running health check...
# 📡 Starting LAN discovery...
# 🌐 Robot Available on LAN
# 📱 Open in browser: http://192.168.1.10:8080/ui
```

### Prepare for Sleep with Learning

```bash
# Sleep and enable self-training
PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py --prepare-sleep

# Output:
# 💤 Preparing for sleep...
# 🧠 Enabling sleep learning mode...
#    Robot will self-train on saved memories
#    Using Gemma-3 for knowledge extraction
```

## Robot Operational Modes

### Manual Training Mode
**Purpose**: Human teleop control for collecting training data

```bash
# Set mode via command
echo '{"method": "set_mode", "params": {"mode": "manual_training"}}' | nc localhost 8080
```

**Features**:
- ✅ Motion enabled (human control)
- ✅ Episode recording enabled
- ❌ VLA inference disabled
- ❌ Self-training disabled

**Use Case**: Demonstrate tasks to robot via Flutter app or web UI. All actions recorded for training.

### Autonomous Mode
**Purpose**: VLA policy control with continuous learning

```bash
# Set mode via command
echo '{"method": "set_mode", "params": {"mode": "autonomous"}}' | nc localhost 8080
```

**Features**:
- ✅ Motion enabled (VLA control)
- ✅ Episode recording enabled
- ✅ VLA inference enabled
- ❌ Self-training disabled

**Use Case**: Robot executes tasks autonomously while recording for continuous improvement.

### Sleep Learning Mode
**Purpose**: Self-training on saved memories during sleep

```bash
# Set mode via command
echo '{"method": "set_mode", "params": {"mode": "sleep_learning"}}' | nc localhost 8080
```

**Features**:
- ❌ Motion disabled (safety during learning)
- ❌ Episode recording disabled
- ❌ VLA inference disabled
- ✅ Self-training enabled

**Use Case**: Robot replays saved episodes, trains LoRA adapters, uses Gemma-3 for knowledge extraction.

### Idle Mode
**Purpose**: Awake but not active

**Features**:
- ❌ All features disabled
- Robot ready to transition to other modes

### Emergency Stop
**Purpose**: Immediate safety stop

**Features**:
- ❌ All motion disabled
- Can only transition to Idle

## LAN Discovery for iPhone/Web

### How It Works

The robot broadcasts its presence on the local network using:
1. **Zeroconf/mDNS (Bonjour)** - Proper discovery for iPhone apps
2. **UDP Broadcast (fallback)** - Simple discovery if Zeroconf unavailable

### Finding Robot on iPhone

**Method 1: Web Browser**
1. Connect iPhone to same Wi-Fi as robot
2. Open Safari or Chrome
3. Navigate to: `http://<robot-ip>:8080/ui`
4. Or scan network for `_continuonbrain._tcp` services

**Method 2: Flutter Companion App**
1. Open Flutter companion app
2. App automatically discovers robots via Bonjour
3. Select "ContinuonBot" from list
4. Connect and control

### Discovery Configuration

```python
from continuonbrain.network_discovery import LANDiscoveryService

discovery = LANDiscoveryService(
    robot_name="MyRobot",  # Custom robot name
    service_port=8080       # Robot API port
)

discovery.start()
```

## Robot API Endpoints

### Control Commands

**Send Arm Command**:
```json
{
  "method": "send_command",
  "params": {
    "client_id": "flutter_app",
    "control_mode": "armJointAngles",
    "arm_joint_angles": {
      "normalized_angles": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
  }
}
```

**Response**:
```json
{
  "success": true,
  "latency_ms": 5,
  "message": "Executed arm command from flutter_app"
}
```

### Mode Management

**Set Robot Mode**:
```json
{
  "method": "set_mode",
  "params": {
    "mode": "manual_training"  // or "autonomous", "sleep_learning", "idle"
  }
}
```

**Get Robot Status**:
```json
{
  "method": "get_status",
  "params": {}
}
```

**Response**:
```json
{
  "success": true,
  "status": {
    "robot_name": "ContinuonBot",
    "mode": "manual_training",
    "mode_duration": 45.2,
    "allow_motion": true,
    "recording_enabled": true,
    "is_recording": false,
    "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }
}
```

### Episode Recording

**Start Recording**:
```json
{
  "method": "start_recording",
  "params": {
    "instruction": "Pick up the red cube"
  }
}
```

**Stop Recording**:
```json
{
  "method": "stop_recording",
  "params": {
    "success": true
  }
}
```

## Complete Wake-Sleep Cycle

### 1. Robot Wakes Up

```bash
PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py --robot-name "KitchenBot"
```

**What Happens**:
1. Detects wake from sleep
2. Runs quick health check (~2s)
3. Starts LAN discovery (Zeroconf + UDP)
4. Launches Robot API server on port 8080
5. Sets mode to Idle
6. Prints connection info for iPhone/web

### 2. User Finds Robot on iPhone

- Open browser: `http://192.168.1.10:8080/ui`
- Or use Flutter app (auto-discovers via Bonjour)

### 3. Start Training Session

**Via API**:
```bash
# Set to manual training mode
echo '{"method": "set_mode", "params": {"mode": "manual_training"}}' | nc localhost 8080

# Start recording episode
echo '{"method": "start_recording", "params": {"instruction": "Pick cube"}}' | nc localhost 8080

# Control robot (repeated)
echo '{"method": "send_command", "params": {"client_id": "web", "control_mode": "armJointAngles", "arm_joint_angles": {"normalized_angles": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]}}}' | nc localhost 8080

# Stop recording
echo '{"method": "stop_recording", "params": {"success": true}}' | nc localhost 8080
```

### 4. Switch to Autonomous

```bash
echo '{"method": "set_mode", "params": {"mode": "autonomous"}}' | nc localhost 8080
```

Robot now executes tasks using VLA policy while continuing to record for improvement.

### 5. Put Robot to Sleep

```bash
# Prepare for sleep with learning enabled
PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py --prepare-sleep
```

**What Happens**:
1. Shuts down LAN discovery
2. Stops Robot API server
3. Sets mode to Sleep Learning
4. Robot self-trains on saved episodes
5. Uses Gemma-3 for knowledge extraction

## Sleep Learning Details

### What Happens During Sleep

1. **Episode Replay**: Robot reviews recorded episodes
2. **LoRA Training**: Trains adapter on new demonstrations
3. **Gemma Knowledge**: Uses Gemma-3 to extract semantic understanding
4. **Memory Consolidation**: Updates skill library

### Configuration

```python
manager.prepare_sleep(enable_learning=True)

# Or disable learning
manager.prepare_sleep(enable_learning=False)
```

### Monitoring

Sleep learning logs saved to:
- `/opt/continuonos/brain/trainer/logs/`
- Health reports in `/opt/continuonos/brain/logs/health_*.json`

## Flutter App Integration

### Connection Flow

1. **Discovery**: App scans for `_continuonbrain._tcp` services
2. **Selection**: User selects robot from list
3. **Connection**: App connects to Robot API (port 8080)
4. **Mode Check**: App queries status to see current mode
5. **Control**: App sends commands based on mode

### Flutter Code Example

```dart
import 'package:continuon_companion/services/robot_api_service.dart';

// Initialize service
final api = RobotApiService();

// Connect to robot
await api.connect('192.168.1.10', 8080);

// Set to manual training
await api.setMode('manual_training');

// Start recording
await api.startRecording('Pick up cup');

// Send arm command
await api.sendArmCommand([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);

// Stop recording
await api.stopRecording(success: true);
```

## Command Line Tools

### Start Robot

```bash
# Normal startup
python3 continuonbrain/startup_manager.py

# Custom robot name
python3 continuonbrain/startup_manager.py --robot-name "LabBot"

# Skip services (manual mode)
python3 continuonbrain/startup_manager.py --no-services

# Force health check on cold boot
python3 continuonbrain/startup_manager.py --force-health-check
```

### Prepare Sleep

```bash
# Sleep with learning
python3 continuonbrain/startup_manager.py --prepare-sleep

# Sleep without learning
python3 continuonbrain/startup_manager.py --prepare-sleep --no-learning
```

### Clean Shutdown

```bash
python3 continuonbrain/startup_manager.py --prepare-shutdown
```

## Systemd Integration

Create `/etc/systemd/system/continuonbrain.service`:

```ini
[Unit]
Description=ContinuonBrain Robot Control
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ContinuonXR
Environment="PYTHONPATH=/home/pi/ContinuonXR"
ExecStart=/usr/bin/python3 continuonbrain/startup_manager.py --robot-name "HomeBot"
ExecStop=/usr/bin/python3 continuonbrain/startup_manager.py --prepare-sleep
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable continuonbrain
sudo systemctl start continuonbrain
```

## Troubleshooting

### Robot Not Discoverable on iPhone

**Check**:
1. Same Wi-Fi network?
2. Firewall blocking port 8080?
3. mDNS/Bonjour enabled on router?

**Solutions**:
```bash
# Check robot IP
hostname -I

# Test API directly
curl http://<robot-ip>:8080/status

# Check if discovery is running
ps aux | grep network_discovery
```

### Mode Changes Rejected

**Error**: "Invalid transition: autonomous → sleep_learning"

**Solution**: Must go through Idle first:
```bash
echo '{"method": "set_mode", "params": {"mode": "idle"}}' | nc localhost 8080
echo '{"method": "set_mode", "params": {"mode": "sleep_learning"}}' | nc localhost 8080
```

### Motion Commands Ignored

**Check current mode**:
```bash
echo '{"method": "get_status", "params": {}}' | nc localhost 8080
```

**Solution**: Ensure in Manual Training or Autonomous mode:
```bash
echo '{"method": "set_mode", "params": {"mode": "manual_training"}}' | nc localhost 8080
```

## See Also

- [System Health](system-health.md) - Health check system
- [Hardware Detection](hardware-detection.md) - Auto-detect cameras/servos
- [Flutter Companion](../apps/flutter-companion/README.md) - Mobile app
- [Robot API](../apps/mock-continuonbrain/README.md) - API reference
