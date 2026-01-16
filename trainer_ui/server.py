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
from typing import Optional, Dict, List
import subprocess

# Add parent directory to path for brain_b imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Hardware management
from hardware import TrainerHardwareDetector, HardwareConfig, DualArmManager, AudioManager, PoseManager, TeachingMode
from hardware.arm_manager import ArmState
from hardware.audio_manager import AudioConfig

# Brain B integration
try:
    from brain_b_integration import BrainBIntegration
    HAS_BRAIN_B = True
except ImportError:
    HAS_BRAIN_B = False
    print("Brain B integration not available")

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

    # Arms (support for dual arms - arm_0 and arm_1)
    arms: Dict[str, dict] = field(default_factory=dict)

    # Legacy single arm (for backwards compatibility)
    arm_joints: list = field(default_factory=lambda: [0.0] * 6)
    gripper: float = 0.0

    # Hardware status
    hailo_available: bool = False
    hailo_tops: float = 0.0
    hailo_model: str = ""
    audio_available: bool = False
    audio_backend: str = ""
    cameras: List[str] = field(default_factory=list)
    is_mock: bool = False

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

        # Hardware management
        self.hw_config: Optional[HardwareConfig] = None
        self.arm_manager: Optional[DualArmManager] = None
        self.audio_manager: Optional[AudioManager] = None
        self.pose_manager: Optional[PoseManager] = None
        self.teaching_mode: Optional[TeachingMode] = None
        self.brain_b: Optional["BrainBIntegration"] = None
        self._hardware_initialized = False

    async def initialize_hardware(self):
        """Initialize hardware detection and controllers."""
        if self._hardware_initialized:
            return

        print("\n" + "=" * 50)
        print("Initializing Hardware...")
        print("=" * 50)

        # Detect hardware
        detector = TrainerHardwareDetector()
        self.hw_config = detector.detect_all()

        # Initialize arm manager
        self.arm_manager = DualArmManager(self.hw_config)
        arm_states = self.arm_manager.initialize()

        # Update robot state with arm info
        self.state.arms = {
            arm_id: asdict(arm_state)
            for arm_id, arm_state in arm_states.items()
        }

        # Initialize audio manager
        audio_config = AudioConfig(
            available=self.hw_config.audio_available,
            backend=self.hw_config.audio_backend,
        )
        self.audio_manager = AudioManager(audio_config)

        # Initialize pose manager and teaching mode
        self.pose_manager = PoseManager(Path("poses"))
        self.teaching_mode = TeachingMode(Path("teachings"))

        # Initialize Brain B integration
        if HAS_BRAIN_B:
            self.brain_b = BrainBIntegration(str(Path("../brain_b_data")))
            if self.brain_b.is_available:
                print("Brain B integration enabled")

        # Update robot state with hardware status
        self.state.hailo_available = self.hw_config.hailo_available
        self.state.hailo_tops = self.hw_config.hailo_tops
        self.state.hailo_model = self.hw_config.hailo_model
        self.state.audio_available = self.hw_config.audio_available
        self.state.audio_backend = self.hw_config.audio_backend
        self.state.cameras = self.hw_config.cameras
        self.state.is_mock = self.hw_config.is_mock

        self._hardware_initialized = True

        print("\nHardware Summary:")
        print(f"  Arms: {list(self.state.arms.keys())}")
        print(f"  Hailo: {self.hw_config.hailo_model or 'N/A'} ({self.hw_config.hailo_tops} TOPS)")
        print(f"  Audio: {self.hw_config.audio_backend or 'N/A'}")
        print(f"  Cameras: {self.hw_config.cameras}")
        print(f"  Mock Mode: {self.hw_config.is_mock}")
        print("=" * 50 + "\n")

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

        elif msg_type == "get_hardware_status":
            await self.handle_get_hardware_status(websocket)

        # Pose management
        elif msg_type == "list_poses":
            await self.handle_list_poses(websocket)

        elif msg_type == "save_pose":
            await self.handle_save_pose(data, websocket)

        elif msg_type == "load_pose":
            await self.handle_load_pose(data, websocket)

        elif msg_type == "delete_pose":
            await self.handle_delete_pose(data, websocket)

        # Teaching mode
        elif msg_type == "start_teaching":
            await self.handle_start_teaching(data, websocket)

        elif msg_type == "stop_teaching":
            await self.handle_stop_teaching(websocket)

        elif msg_type == "list_teachings":
            await self.handle_list_teachings(websocket)

        elif msg_type == "playback_teaching":
            await self.handle_playback_teaching(data, websocket)

        elif msg_type == "delete_teaching":
            await self.handle_delete_teaching(data, websocket)

        # Dual arm coordination
        elif msg_type == "mirror_arms":
            await self.handle_mirror_arms(data)

        elif msg_type == "sync_arms":
            await self.handle_sync_arms(data)

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
        """Handle arm joint command with multi-arm support and Brain B validation."""
        arm_id = data.get("arm_id", "arm_0")  # Default to arm_0 for backwards compatibility
        joint = data.get("joint", 0)
        value = data.get("value", 0.0)

        # Validate through Brain B if available
        if self.brain_b and self.brain_b.is_available:
            is_valid, message, adjusted_value = self.brain_b.validate_arm_action(
                arm_id, joint, value, self.state.arms.get(arm_id)
            )
            if adjusted_value != value:
                value = adjusted_value  # Use rate-limited value

            # Record action in Brain B for teaching
            self.brain_b.record_arm_action("arm", arm_id, joint=joint, value=value)

        # Update legacy single arm state for backwards compatibility
        if 0 <= joint < 6:
            self.state.arm_joints[joint] = max(-1.0, min(1.0, value))

        # Send to actual arm via arm manager
        if self.arm_manager:
            success = self.arm_manager.set_joint(arm_id, joint, value)
            if success:
                # Update arms state
                arm_states = self.arm_manager.get_all_states()
                self.state.arms = {
                    aid: asdict(astate)
                    for aid, astate in arm_states.items()
                }

                # Record frame if teaching mode is active
                if self.teaching_mode and self.teaching_mode.is_recording:
                    arm = self.arm_manager.get_arm(arm_id)
                    if arm and self.teaching_mode.current_recording and self.teaching_mode.current_recording.arm_id == arm_id:
                        self.teaching_mode.record_frame(arm)

        self.state.last_command = f"arm:{arm_id}:joint{joint}={value:.2f}"
        self.state.last_command_time = time.time()

        # Record for RLDS training
        self.recorder.record(
            {"type": "arm", "arm_id": arm_id, "joint": joint, "value": value},
            self.state,
            self.state.recognized_user,
        )

        await self.broadcast({"type": "state", "data": asdict(self.state)})

    async def handle_gripper(self, data: dict):
        """Handle gripper command with multi-arm support and Brain B validation."""
        arm_id = data.get("arm_id", "arm_0")  # Default to arm_0 for backwards compatibility
        value = data.get("value", 0.0)

        # Validate through Brain B if available
        if self.brain_b and self.brain_b.is_available:
            is_valid, message, adjusted_value = self.brain_b.validate_gripper_action(
                arm_id, value, self.state.arms.get(arm_id)
            )
            if adjusted_value != value:
                value = adjusted_value

            # Record action in Brain B for teaching
            self.brain_b.record_arm_action("gripper", arm_id, value=value)

        # Update legacy single gripper state for backwards compatibility
        self.state.gripper = max(0.0, min(1.0, value))

        # Send to actual gripper via arm manager
        if self.arm_manager:
            success = self.arm_manager.set_gripper(arm_id, value)
            if success:
                # Update arms state
                arm_states = self.arm_manager.get_all_states()
                self.state.arms = {
                    aid: asdict(astate)
                    for aid, astate in arm_states.items()
                }

                # Record frame if teaching mode is active
                if self.teaching_mode and self.teaching_mode.is_recording:
                    arm = self.arm_manager.get_arm(arm_id)
                    if arm and self.teaching_mode.current_recording and self.teaching_mode.current_recording.arm_id == arm_id:
                        self.teaching_mode.record_frame(arm)

        self.state.last_command = f"gripper:{arm_id}:{value:.2f}"
        self.state.last_command_time = time.time()

        # Record for RLDS training
        self.recorder.record(
            {"type": "gripper", "arm_id": arm_id, "value": value},
            self.state,
            self.state.recognized_user,
        )

        await self.broadcast({"type": "state", "data": asdict(self.state)})

    async def handle_speak(self, data: dict, websocket: WebSocket):
        """Handle text-to-speech request using audio manager."""
        text = data.get("text", "")

        if self.audio_manager:
            result = self.audio_manager.speak(text)
            if result.get("status") == "ok":
                await websocket.send_json({"type": "speak_done", "text": text, "backend": result.get("backend")})
            else:
                await websocket.send_json({"type": "error", "message": result.get("message", "TTS failed")})
        else:
            # Fallback to direct system calls
            try:
                if sys.platform == "darwin":
                    subprocess.run(["say", text], check=True)
                else:
                    subprocess.run(["espeak-ng", text], check=True)
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
        """Emergency stop all motors and arms."""
        self.state.drive_left = 0
        self.state.drive_right = 0
        self.state.arm_joints = [0.0] * 6
        self.state.last_command = "EMERGENCY_STOP"
        self.state.last_command_time = time.time()

        # Stop all arms
        if self.arm_manager:
            self.arm_manager.emergency_stop_all()
            # Update arm states
            arm_states = self.arm_manager.get_all_states()
            self.state.arms = {
                aid: asdict(astate)
                for aid, astate in arm_states.items()
            }

        await self.broadcast({
            "type": "emergency_stop",
            "state": asdict(self.state),
        })

    async def handle_get_hardware_status(self, websocket: WebSocket):
        """Send current hardware status to client."""
        status = {
            "type": "hardware_status",
            "arms": self.state.arms,
            "hailo": {
                "available": self.state.hailo_available,
                "tops": self.state.hailo_tops,
                "model": self.state.hailo_model,
            },
            "audio": {
                "available": self.state.audio_available,
                "backend": self.state.audio_backend,
            },
            "cameras": self.state.cameras,
            "is_mock": self.state.is_mock,
        }
        await websocket.send_json(status)

    # ========================================================================
    # Pose Management Handlers
    # ========================================================================

    async def handle_list_poses(self, websocket: WebSocket):
        """List all available poses."""
        if self.pose_manager:
            poses = self.pose_manager.list_poses()
            await websocket.send_json({"type": "poses_list", "poses": poses})

    async def handle_save_pose(self, data: dict, websocket: WebSocket):
        """Save current arm position as a named pose."""
        name = data.get("name", "").strip()
        arm_id = data.get("arm_id", "arm_0")

        if not name:
            await websocket.send_json({"type": "error", "message": "Pose name required"})
            return

        if self.arm_manager and self.pose_manager:
            arm = self.arm_manager.get_arm(arm_id)
            if arm:
                pose = self.pose_manager.save_pose(name, arm)
                await websocket.send_json({
                    "type": "pose_saved",
                    "name": name,
                    "poses": self.pose_manager.list_poses(),
                })
            else:
                await websocket.send_json({"type": "error", "message": f"Arm not found: {arm_id}"})

    async def handle_load_pose(self, data: dict, websocket: WebSocket):
        """Load a named pose to an arm."""
        name = data.get("name", "")
        arm_id = data.get("arm_id", "arm_0")

        if self.arm_manager and self.pose_manager:
            arm = self.arm_manager.get_arm(arm_id)
            if arm:
                success = self.pose_manager.load_pose(name, arm)
                if success:
                    # Update state
                    arm_states = self.arm_manager.get_all_states()
                    self.state.arms = {
                        aid: asdict(astate)
                        for aid, astate in arm_states.items()
                    }
                    await self.broadcast({"type": "state", "data": asdict(self.state)})
                    await websocket.send_json({"type": "pose_loaded", "name": name, "arm_id": arm_id})
                else:
                    await websocket.send_json({"type": "error", "message": f"Pose not found: {name}"})

    async def handle_delete_pose(self, data: dict, websocket: WebSocket):
        """Delete a saved pose."""
        name = data.get("name", "")

        if self.pose_manager:
            success = self.pose_manager.delete_pose(name)
            if success:
                await websocket.send_json({
                    "type": "pose_deleted",
                    "name": name,
                    "poses": self.pose_manager.list_poses(),
                })
            else:
                await websocket.send_json({"type": "error", "message": f"Cannot delete pose: {name}"})

    # ========================================================================
    # Teaching Mode Handlers
    # ========================================================================

    async def handle_start_teaching(self, data: dict, websocket: WebSocket):
        """Start recording arm movements for teaching."""
        name = data.get("name", "").strip()
        arm_id = data.get("arm_id", "arm_0")

        if not name:
            await websocket.send_json({"type": "error", "message": "Teaching name required"})
            return

        if self.teaching_mode:
            success = self.teaching_mode.start_recording(name, arm_id)
            if success:
                await websocket.send_json({"type": "teaching_started", "name": name, "arm_id": arm_id})
            else:
                await websocket.send_json({"type": "error", "message": "Already recording"})

    async def handle_stop_teaching(self, websocket: WebSocket):
        """Stop teaching recording and save."""
        if self.teaching_mode:
            rec = self.teaching_mode.stop_recording()
            if rec:
                await websocket.send_json({
                    "type": "teaching_stopped",
                    "name": rec.name,
                    "duration_s": rec.duration_s,
                    "frame_count": len(rec.frames),
                    "teachings": self.teaching_mode.list_recordings(),
                })
            else:
                await websocket.send_json({"type": "error", "message": "No recording in progress"})

    async def handle_list_teachings(self, websocket: WebSocket):
        """List all teaching recordings."""
        if self.teaching_mode:
            teachings = self.teaching_mode.list_recordings()
            await websocket.send_json({"type": "teachings_list", "teachings": teachings})

    async def handle_playback_teaching(self, data: dict, websocket: WebSocket):
        """Play back a teaching recording."""
        name = data.get("name", "")
        arm_id = data.get("arm_id", "arm_0")
        speed = data.get("speed", 1.0)

        if self.arm_manager and self.teaching_mode:
            arm = self.arm_manager.get_arm(arm_id)
            if arm:
                await websocket.send_json({"type": "playback_started", "name": name})
                await self.teaching_mode.playback(name, arm, speed)
                # Update state after playback
                arm_states = self.arm_manager.get_all_states()
                self.state.arms = {
                    aid: asdict(astate)
                    for aid, astate in arm_states.items()
                }
                await self.broadcast({"type": "state", "data": asdict(self.state)})
                await websocket.send_json({"type": "playback_complete", "name": name})

    async def handle_delete_teaching(self, data: dict, websocket: WebSocket):
        """Delete a teaching recording."""
        name = data.get("name", "")

        if self.teaching_mode:
            success = self.teaching_mode.delete_recording(name)
            if success:
                await websocket.send_json({
                    "type": "teaching_deleted",
                    "name": name,
                    "teachings": self.teaching_mode.list_recordings(),
                })
            else:
                await websocket.send_json({"type": "error", "message": f"Recording not found: {name}"})

    # ========================================================================
    # Dual Arm Coordination Handlers
    # ========================================================================

    async def handle_mirror_arms(self, data: dict):
        """Mirror arm_0 movements to arm_1 (or vice versa)."""
        source_id = data.get("source", "arm_0")
        target_id = "arm_1" if source_id == "arm_0" else "arm_0"
        invert_x = data.get("invert_x", True)  # Invert base rotation for mirroring

        if self.arm_manager:
            source = self.arm_manager.get_arm(source_id)
            target = self.arm_manager.get_arm(target_id)

            if source and target:
                # Copy joints with optional X inversion
                for i in range(5):
                    value = source.joint_values[i]
                    if i == 0 and invert_x:  # Base joint
                        value = -value
                    target.set_joint(i, value)
                target.set_gripper(source.gripper_value)

                # Update state
                arm_states = self.arm_manager.get_all_states()
                self.state.arms = {
                    aid: asdict(astate)
                    for aid, astate in arm_states.items()
                }
                await self.broadcast({"type": "state", "data": asdict(self.state)})

    async def handle_sync_arms(self, data: dict):
        """Synchronize both arms to the same position."""
        joints = data.get("joints", [0.0] * 5)
        gripper = data.get("gripper", 0.0)

        if self.arm_manager:
            for arm_id in self.arm_manager.arm_ids:
                arm = self.arm_manager.get_arm(arm_id)
                if arm:
                    for i, value in enumerate(joints[:5]):
                        arm.set_joint(i, value)
                    arm.set_gripper(gripper)

            # Update state
            arm_states = self.arm_manager.get_all_states()
            self.state.arms = {
                aid: asdict(astate)
                for aid, astate in arm_states.items()
            }
            await self.broadcast({"type": "state", "data": asdict(self.state)})


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Brain A Trainer")
server = TrainerServer()


@app.on_event("startup")
async def startup_event():
    """Initialize hardware on startup."""
    await server.initialize_hardware()


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
        "hardware": {
            "arms": list(server.state.arms.keys()) if server.state.arms else [],
            "arm_count": len(server.state.arms) if server.state.arms else 0,
            "hailo": {
                "available": server.state.hailo_available,
                "model": server.state.hailo_model,
                "tops": server.state.hailo_tops,
            },
            "audio": {
                "available": server.state.audio_available,
                "backend": server.state.audio_backend,
            },
            "cameras": server.state.cameras,
            "is_mock": server.state.is_mock,
        },
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
║  • Arms: Dual SO-ARM101 support (6-axis + gripper)             ║
║  • Camera: Live feed with face recognition                     ║
║  • Voice: Speak to robot, TTS output                           ║
║  • Claude: Ask Claude Code for help                            ║
║  • Recording: Save sessions as RLDS for training               ║
║  • Hardware: Auto-detect Hailo, arms, audio                    ║
║                                                                 ║
║  OpenCV: {'✓' if HAS_OPENCV else '✗'}  Face Recognition: {'✓' if HAS_FACE_RECOGNITION else '✗'}              ║
║                                                                 ║
║  Set TRAINER_MOCK_HARDWARE=1 to test without hardware          ║
╚═══════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port)
