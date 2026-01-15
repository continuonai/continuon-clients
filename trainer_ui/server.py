#!/usr/bin/env python3
"""
Brain A Trainer - Simple Web Server

A minimal web interface for training your robot:
- Drive controls (WASD)
- Robot arm controls
- Camera feed with face recognition
- Microphone/speaker
- Claude Code integration
- Records everything for RLDS training

Usage:
    python server.py
    # Open http://localhost:8000
"""

import asyncio
import json
import base64
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
import subprocess

# Add parent directory to path for brain_b imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
except ImportError:
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "websockets"], check=True)
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("OpenCV not available - camera features disabled")

try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False
    print("face_recognition not available - face features disabled")

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8000
    face_db_path: Path = Path("face_db")
    brain_b_path: Path = Path("../brain_b_data")
    rlds_path: Path = Path("../continuonbrain/rlds/episodes")
    camera_index: int = 0
    face_recognition_tolerance: float = 0.6

CONFIG = Config()

# ============================================================================
# Face Recognition
# ============================================================================

class FaceDB:
    """Simple face database for recognition."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.known_faces: dict[str, list] = {}  # name -> encodings
        self._load()

    def _load(self):
        """Load known faces from disk."""
        for face_file in self.db_path.glob("*.json"):
            name = face_file.stem
            with open(face_file) as f:
                data = json.load(f)
                self.known_faces[name] = [np.array(e) for e in data["encodings"]]
        print(f"Loaded {len(self.known_faces)} known faces")

    def add_face(self, name: str, encoding: np.ndarray):
        """Add a face encoding to the database."""
        if name not in self.known_faces:
            self.known_faces[name] = []

        self.known_faces[name].append(encoding)

        # Save to disk
        face_file = self.db_path / f"{name}.json"
        with open(face_file, "w") as f:
            json.dump({
                "name": name,
                "encodings": [e.tolist() for e in self.known_faces[name]],
                "updated": datetime.now().isoformat(),
            }, f)

        print(f"Added face for {name} ({len(self.known_faces[name])} encodings)")

    def recognize(self, encoding: np.ndarray, tolerance: float = 0.6) -> Optional[str]:
        """Recognize a face encoding against known faces."""
        if not HAS_FACE_RECOGNITION:
            return None

        for name, known_encodings in self.known_faces.items():
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance)
            if any(matches):
                return name

        return None

# ============================================================================
# Robot State
# ============================================================================

@dataclass
class RobotState:
    """Current robot state."""

    # Drive
    drive_left: float = 0.0
    drive_right: float = 0.0

    # Arm (6 joints)
    arm_joints: list = field(default_factory=lambda: [0.0] * 6)
    gripper: float = 0.0

    # Status
    battery: float = 1.0
    connected: bool = False
    last_command: str = ""
    last_command_time: float = 0.0

    # Recognition
    recognized_user: Optional[str] = None
    face_confidence: float = 0.0

# ============================================================================
# Training Recorder
# ============================================================================

class TrainingRecorder:
    """Records interactions for RLDS training."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.session_id = f"trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.steps: list[dict] = []
        self.recording = False

    def start(self):
        self.recording = True
        self.steps = []
        print(f"Recording started: {self.session_id}")

    def stop(self) -> Path:
        self.recording = False
        if not self.steps:
            return None

        # Write episode
        episode_dir = self.output_path / self.session_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Metadata
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump({
                "episode_id": self.session_id,
                "environment_id": "trainer_ui",
                "num_steps": len(self.steps),
                "created": datetime.now().isoformat(),
            }, f, indent=2)

        # Steps
        steps_dir = episode_dir / "steps"
        steps_dir.mkdir(exist_ok=True)
        with open(steps_dir / "000000.jsonl", "w") as f:
            for step in self.steps:
                f.write(json.dumps(step) + "\n")

        print(f"Saved {len(self.steps)} steps to {episode_dir}")
        return episode_dir

    def record(self, action: dict, state: RobotState, user: Optional[str] = None):
        if not self.recording:
            return

        self.steps.append({
            "timestamp": time.time(),
            "action": action,
            "state": asdict(state),
            "user": user,
            "step_idx": len(self.steps),
        })

# ============================================================================
# Claude Code Integration
# ============================================================================

async def ask_claude(prompt: str) -> str:
    """Send a prompt to Claude Code and get response."""
    try:
        # Use Claude Code CLI
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"
    except FileNotFoundError:
        return "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
    except subprocess.TimeoutExpired:
        return "Request timed out"
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# WebSocket Handler
# ============================================================================

class TrainerServer:
    """Main server handling all WebSocket connections."""

    def __init__(self):
        self.state = RobotState()
        self.face_db = FaceDB(CONFIG.face_db_path)
        self.recorder = TrainingRecorder(CONFIG.rlds_path)
        self.connections: list[WebSocket] = []
        self.camera = None
        self.camera_task = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        print(f"Client connected ({len(self.connections)} total)")

        # Send initial state
        await websocket.send_json({
            "type": "state",
            "data": asdict(self.state),
            "known_faces": list(self.face_db.known_faces.keys()),
        })

    def disconnect(self, websocket: WebSocket):
        self.connections.remove(websocket)
        print(f"Client disconnected ({len(self.connections)} total)")

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except:
                pass

    async def handle_message(self, websocket: WebSocket, data: dict):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type", "")

        if msg_type == "drive":
            await self.handle_drive(data)

        elif msg_type == "arm":
            await self.handle_arm(data)

        elif msg_type == "gripper":
            await self.handle_gripper(data)

        elif msg_type == "speak":
            await self.handle_speak(data, websocket)

        elif msg_type == "ask_claude":
            await self.handle_claude(data, websocket)

        elif msg_type == "camera_frame":
            await self.handle_camera_frame(data, websocket)

        elif msg_type == "register_face":
            await self.handle_register_face(data, websocket)

        elif msg_type == "start_recording":
            self.recorder.start()
            await websocket.send_json({"type": "recording", "status": "started"})

        elif msg_type == "stop_recording":
            path = self.recorder.stop()
            await websocket.send_json({
                "type": "recording",
                "status": "stopped",
                "path": str(path) if path else None,
            })

        elif msg_type == "emergency_stop":
            await self.emergency_stop()

    async def handle_drive(self, data: dict):
        """Handle drive command."""
        direction = data.get("direction", "stop")
        speed = data.get("speed", 0.5)

        if direction == "forward":
            self.state.drive_left = speed
            self.state.drive_right = speed
        elif direction == "backward":
            self.state.drive_left = -speed
            self.state.drive_right = -speed
        elif direction == "left":
            self.state.drive_left = -speed * 0.5
            self.state.drive_right = speed * 0.5
        elif direction == "right":
            self.state.drive_left = speed * 0.5
            self.state.drive_right = -speed * 0.5
        elif direction == "stop":
            self.state.drive_left = 0
            self.state.drive_right = 0

        self.state.last_command = f"drive:{direction}"
        self.state.last_command_time = time.time()

        # Record for training
        self.recorder.record(
            {"type": "drive", "direction": direction, "speed": speed},
            self.state,
            self.state.recognized_user,
        )

        # TODO: Send to actual motors
        # motor_controller.set_speed(self.state.drive_left, self.state.drive_right)

        await self.broadcast({"type": "state", "data": asdict(self.state)})

    async def handle_arm(self, data: dict):
        """Handle arm joint command."""
        joint = data.get("joint", 0)
        value = data.get("value", 0.0)

        if 0 <= joint < 6:
            self.state.arm_joints[joint] = max(-1.0, min(1.0, value))

        self.state.last_command = f"arm:joint{joint}={value:.2f}"
        self.state.last_command_time = time.time()

        # Record for training
        self.recorder.record(
            {"type": "arm", "joint": joint, "value": value},
            self.state,
            self.state.recognized_user,
        )

        # TODO: Send to actual arm
        # arm_controller.set_joint(joint, value)

        await self.broadcast({"type": "state", "data": asdict(self.state)})

    async def handle_gripper(self, data: dict):
        """Handle gripper command."""
        value = data.get("value", 0.0)
        self.state.gripper = max(0.0, min(1.0, value))

        self.state.last_command = f"gripper:{value:.2f}"
        self.state.last_command_time = time.time()

        # Record for training
        self.recorder.record(
            {"type": "gripper", "value": value},
            self.state,
            self.state.recognized_user,
        )

        await self.broadcast({"type": "state", "data": asdict(self.state)})

    async def handle_speak(self, data: dict, websocket: WebSocket):
        """Handle text-to-speech request."""
        text = data.get("text", "")

        try:
            # Use system TTS
            if sys.platform == "darwin":
                subprocess.run(["say", text], check=True)
            else:
                subprocess.run(["espeak", text], check=True)

            await websocket.send_json({"type": "speak_done", "text": text})
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})

    async def handle_claude(self, data: dict, websocket: WebSocket):
        """Handle Claude Code request."""
        prompt = data.get("prompt", "")

        await websocket.send_json({"type": "claude_thinking"})

        response = await ask_claude(prompt)

        # Record for training
        self.recorder.record(
            {"type": "claude", "prompt": prompt, "response": response[:500]},
            self.state,
            self.state.recognized_user,
        )

        await websocket.send_json({
            "type": "claude_response",
            "prompt": prompt,
            "response": response,
        })

    async def handle_camera_frame(self, data: dict, websocket: WebSocket):
        """Handle camera frame from browser (for face recognition)."""
        if not HAS_FACE_RECOGNITION:
            return

        try:
            # Decode base64 image
            image_data = data.get("image", "").split(",")[-1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return

            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            faces = []
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                name = self.face_db.recognize(encoding, CONFIG.face_recognition_tolerance)
                faces.append({
                    "box": [left, top, right, bottom],
                    "name": name or "Unknown",
                })

                # Update recognized user
                if name:
                    self.state.recognized_user = name
                    self.state.face_confidence = 1.0

            await websocket.send_json({
                "type": "faces_detected",
                "faces": faces,
            })

        except Exception as e:
            print(f"Face recognition error: {e}")

    async def handle_register_face(self, data: dict, websocket: WebSocket):
        """Register a new face."""
        if not HAS_FACE_RECOGNITION:
            await websocket.send_json({
                "type": "error",
                "message": "Face recognition not available",
            })
            return

        name = data.get("name", "").strip()
        image_data = data.get("image", "").split(",")[-1]

        if not name:
            await websocket.send_json({
                "type": "error",
                "message": "Name is required",
            })
            return

        try:
            # Decode image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_frame)

            if not face_encodings:
                await websocket.send_json({
                    "type": "error",
                    "message": "No face detected in image",
                })
                return

            # Add to database
            self.face_db.add_face(name, face_encodings[0])

            await websocket.send_json({
                "type": "face_registered",
                "name": name,
                "known_faces": list(self.face_db.known_faces.keys()),
            })

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })

    async def emergency_stop(self):
        """Emergency stop all motors."""
        self.state.drive_left = 0
        self.state.drive_right = 0
        self.state.arm_joints = [0.0] * 6
        self.state.last_command = "EMERGENCY_STOP"
        self.state.last_command_time = time.time()

        # TODO: Send to actual hardware
        # motor_controller.stop()
        # arm_controller.stop()

        await self.broadcast({
            "type": "emergency_stop",
            "state": asdict(self.state),
        })

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Brain A Trainer")
server = TrainerServer()

@app.get("/")
async def index():
    """Serve the main UI."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await server.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await server.handle_message(websocket, data)
    except WebSocketDisconnect:
        server.disconnect(websocket)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "has_opencv": HAS_OPENCV,
        "has_face_recognition": HAS_FACE_RECOGNITION,
        "connections": len(server.connections),
        "known_faces": list(server.face_db.known_faces.keys()),
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║               BRAIN A TRAINER - Web Interface                  ║
╠═══════════════════════════════════════════════════════════════╣
║  Open http://localhost:{CONFIG.port} in your browser              ║
║                                                                 ║
║  Features:                                                      ║
║  • Drive: WASD keys or on-screen controls                      ║
║  • Arm: 6-axis joint control with sliders                      ║
║  • Camera: Live feed with face recognition                     ║
║  • Voice: Speak to robot, TTS output                           ║
║  • Claude: Ask Claude Code for help                            ║
║  • Recording: Save sessions as RLDS for training               ║
║                                                                 ║
║  OpenCV: {'✓' if HAS_OPENCV else '✗'}  Face Recognition: {'✓' if HAS_FACE_RECOGNITION else '✗'}              ║
╚═══════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port)
