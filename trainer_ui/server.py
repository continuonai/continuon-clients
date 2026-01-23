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

# Simulator utilities
try:
    from simulator_utils import SimulatorSession, get_simulator_session
    HAS_SIMULATOR = True
except ImportError:
    HAS_SIMULATOR = False
    print("Simulator utilities not available")

# Simulator training
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "brain_b" / "simulator"))
    from simulator_training import (
        SimulatorTrainer,
        SimulatorDataset,
        SimulatorActionPredictor,
        run_simulator_training,
        get_simulator_predictor,
    )
    HAS_SIM_TRAINING = True
except ImportError:
    HAS_SIM_TRAINING = False
    print("Simulator training not available")

# Auto-trainer for retraining models
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "brain_b"))
    from trainer.auto_trainer import AutoTrainer, TrainingConfig, get_auto_trainer
    HAS_AUTO_TRAINER = True
except ImportError:
    HAS_AUTO_TRAINER = False
    print("Auto-trainer not available")

# HomeScan 3D home simulator integration
try:
    from home_scan import HomeScanIntegration, HomeScanConfig, create_home_scan_handlers
    HAS_HOME_SCAN = True
except ImportError:
    HAS_HOME_SCAN = False
    print("HomeScan 3D simulator not available")

# Room Scanner - Image to 3D asset pipeline
try:
    from room_scanner import RoomScanner, process_room_images, get_room_scanner
    HAS_ROOM_SCANNER = True
except ImportError:
    HAS_ROOM_SCANNER = False
    print("Room scanner not available")

# Hailo-8 NPU Scanner - Fast neural segmentation
try:
    from hailo_scanner import (
        HailoScanner, scan_with_hailo, get_hailo_scanner,
        scan_with_hailo_sam3, get_sam3_model,
        scan_with_fastsam, get_fastsam_scanner
    )
    HAS_HAILO_SCANNER = True
except ImportError:
    HAS_HAILO_SCANNER = False
    print("Hailo scanner not available")

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

try:
    from oakd_camera import get_oakd_camera, OakDCamera, HAS_DEPTHAI
except ImportError:
    HAS_DEPTHAI = False
    print("OAK-D camera module not available")

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

        # Simulator session for HomeScan (legacy 2D)
        self.simulator_session: Optional[SimulatorSession] = None
        if HAS_SIMULATOR:
            self.simulator_session = get_simulator_session(CONFIG.rlds_path)

        # HomeScan 3D home simulator
        self.home_scan: Optional["HomeScanIntegration"] = None
        self._home_scan_handlers: Dict = {}
        if HAS_HOME_SCAN:
            home_scan_config = HomeScanConfig(
                rlds_output_path=Path("../brain_b/brain_b_data/home_rlds"),
                auto_record=True,
            )
            self.home_scan = HomeScanIntegration(home_scan_config)
            self._home_scan_handlers = create_home_scan_handlers(self.home_scan)

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
        disconnected = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                # Client likely disconnected, mark for removal
                disconnected.append(ws)
        # Clean up disconnected clients
        for ws in disconnected:
            if ws in self.connections:
                self.connections.remove(ws)

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

        # Simulator handlers (legacy 2D)
        elif msg_type == "sim_step":
            await self.handle_sim_step(data, websocket)

        elif msg_type == "sim_reset":
            await self.handle_sim_reset(websocket)

        elif msg_type == "sim_status":
            await self.handle_sim_status(websocket)

        # HomeScan 3D home simulator handlers
        elif msg_type.startswith("home_"):
            await self.handle_home_scan(msg_type, data, websocket)

    async def handle_drive(self, data: dict):
        """Handle drive command with Brain B motor control."""
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

        # Send to actual motors via Brain B
        if self.brain_b and self.brain_b.is_available:
            if direction == "stop":
                self.brain_b.stop_drive()
            else:
                self.brain_b.set_drive(self.state.drive_left, self.state.drive_right)
            # Record for Brain B teaching
            self.brain_b.record_drive_action(direction, speed)

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

    # ========================================================================
    # Simulator Handlers (HomeScan)
    # ========================================================================

    async def handle_sim_step(self, data: dict, websocket: WebSocket):
        """Handle a simulator step from HomeScan client."""
        if not self.simulator_session:
            await websocket.send_json({"type": "error", "message": "Simulator not available"})
            return

        # Start session if not already started
        session_id = data.get("session_id", f"sim_{int(time.time())}")
        if not self.simulator_session.session_id:
            self.simulator_session.start(session_id)

        # Process the step
        result = self.simulator_session.process_step(data)

        # Also record to the main RLDS recorder if recording is active
        if self.recorder.recording:
            action_data = data.get("action", {})
            self.recorder.record(
                {"type": "sim_action", **action_data},
                self.state,
                self.state.recognized_user,
            )

        await websocket.send_json({
            "type": "sim_step_result",
            **result,
        })

    async def handle_sim_reset(self, websocket: WebSocket):
        """Reset the simulator environment."""
        if self.simulator_session:
            self.simulator_session.reset()
            await websocket.send_json({
                "type": "sim_reset",
                "status": "ok",
            })

    async def handle_sim_status(self, websocket: WebSocket):
        """Get simulator session status."""
        if self.simulator_session:
            status = self.simulator_session.get_status()
            await websocket.send_json({
                "type": "sim_status",
                **status,
            })
        else:
            await websocket.send_json({
                "type": "sim_status",
                "available": False,
            })

    # ========================================================================
    # HomeScan 3D Home Simulator Handlers
    # ========================================================================

    async def handle_home_scan(self, msg_type: str, data: dict, websocket: WebSocket):
        """Handle HomeScan 3D simulator messages."""
        print(f"[HomeScan] Received: {msg_type}")

        if not self.home_scan or not HAS_HOME_SCAN:
            print(f"[HomeScan] Not available: home_scan={self.home_scan}, HAS_HOME_SCAN={HAS_HOME_SCAN}")
            await websocket.send_json({
                "type": "error",
                "message": "HomeScan 3D simulator not available",
            })
            return

        # Get the handler for this message type
        handler = self._home_scan_handlers.get(msg_type)
        if handler:
            print(f"[HomeScan] Calling handler for: {msg_type}")
            result = await handler(data)
            print(f"[HomeScan] Handler result type: {result.get('type')}")
            await websocket.send_json(result)

            # Broadcast state updates to all clients
            if result.get("type") in ("home_scan_state", "home_scan_result"):
                await self.broadcast({
                    "type": "home_scan_update",
                    "state": result.get("state"),
                })
        else:
            print(f"[HomeScan] Unknown message type: {msg_type}, available handlers: {list(self._home_scan_handlers.keys())}")
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown HomeScan message type: {msg_type}",
            })


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Brain A Trainer")
server = TrainerServer()

# Mount static files directory
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize hardware on startup."""
    await server.initialize_hardware()


@app.get("/")
async def index():
    """Serve the main UI."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/homescan")
async def homescan():
    """Serve the HomeScan simulator UI (3D Three.js version)."""
    return FileResponse(Path(__file__).parent / "static" / "homescan.html")


@app.get("/home-explore")
async def home_explore():
    """Serve the Home Explore 3D navigation trainer UI."""
    return FileResponse(Path(__file__).parent / "static" / "home_explore.html")


@app.get("/room-scanner")
async def room_scanner_ui():
    """Serve the Room Scanner UI for image-to-3D processing."""
    return FileResponse(Path(__file__).parent / "static" / "room_scanner.html")


@app.get("/home-scan/status")
async def home_scan_status():
    """Get HomeScan 3D simulator status."""
    if server.home_scan and HAS_HOME_SCAN:
        return {
            "status": "ok",
            "home_scan": server.home_scan.get_state_dict(),
            "levels": server.home_scan.get_available_levels(),
            "curriculum": server.home_scan.get_curriculum(),
        }
    return {"status": "unavailable", "home_scan": None}


@app.get("/simulator/status")
async def simulator_status():
    """Get simulator session status."""
    if server.simulator_session:
        return {
            "status": "ok",
            "simulator": server.simulator_session.get_status(),
        }
    return {"status": "unavailable", "simulator": None}

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
        "brain_b": {
            "available": server.brain_b is not None and server.brain_b.is_available,
            "predictor": server.brain_b.handler.predictor.is_ready if (
                server.brain_b and server.brain_b.is_available and
                hasattr(server.brain_b, 'handler') and server.brain_b.handler and
                hasattr(server.brain_b.handler, 'predictor') and server.brain_b.handler.predictor
            ) else False,
        },
        "simulator": {
            "available": HAS_SIMULATOR and server.simulator_session is not None,
            "recording": server.simulator_session.recorder.is_recording() if server.simulator_session else False,
            "session_id": server.simulator_session.session_id if server.simulator_session else None,
        },
        "home_scan": {
            "available": HAS_HOME_SCAN and server.home_scan is not None,
            "active": server.home_scan.state.active if server.home_scan else False,
            "level": server.home_scan.state.level_name if server.home_scan else None,
            "recording": server.home_scan.state.recording if server.home_scan else False,
            "predictor_ready": server.home_scan.state.predictor_ready if server.home_scan else False,
        },
        "oakd": {
            "available": HAS_DEPTHAI if 'HAS_DEPTHAI' in dir() else False,
        },
    }


# ============================================================================
# OAK-D Camera Endpoints
# ============================================================================

@app.get("/oakd/devices")
async def oakd_list_devices():
    """List available OAK-D devices."""
    if not HAS_DEPTHAI:
        return {"error": "DepthAI not available", "devices": []}

    camera = get_oakd_camera()
    devices = camera.get_available_devices()
    return {
        "status": "ok",
        "devices": devices,
        "count": len(devices),
    }


@app.post("/oakd/start")
async def oakd_start():
    """Start the OAK-D camera."""
    if not HAS_DEPTHAI:
        return {"error": "DepthAI not available"}

    camera = get_oakd_camera()
    success = camera.start()
    return {
        "status": "ok" if success else "error",
        "running": camera.is_running,
    }


@app.post("/oakd/stop")
async def oakd_stop():
    """Stop the OAK-D camera."""
    if not HAS_DEPTHAI:
        return {"error": "DepthAI not available"}

    camera = get_oakd_camera()
    camera.stop()
    return {"status": "ok", "running": False}


@app.get("/oakd/status")
async def oakd_status():
    """Get OAK-D camera status."""
    if not HAS_DEPTHAI:
        return {"available": False, "running": False}

    camera = get_oakd_camera()
    devices = camera.get_available_devices()
    return {
        "available": True,
        "running": camera.is_running,
        "devices": devices,
    }


@app.get("/oakd/capture")
async def oakd_capture(include_depth: bool = True):
    """Capture a frame from OAK-D camera."""
    if not HAS_DEPTHAI:
        return {"error": "DepthAI not available"}

    camera = get_oakd_camera()
    if not camera.is_running:
        # Auto-start if not running
        if not camera.start():
            return {"error": "Failed to start camera"}

    result = camera.capture_base64(include_depth=include_depth)
    return result


@app.get("/predict")
async def predict_tool(task: str = ""):
    """
    Predict the next tool based on task description.

    Args:
        task: Optional task description (e.g., "run the tests")

    Returns:
        Tool prediction with confidence and alternatives
    """
    if not server.brain_b or not server.brain_b.is_available:
        return {"error": "Brain B not available", "prediction": None}

    handler = server.brain_b.handler
    if not handler or not hasattr(handler, 'predict_tool'):
        return {"error": "Predictor not available", "prediction": None}

    prediction = handler.predict_tool(task)
    if prediction:
        return {
            "status": "ok",
            "task": task,
            "prediction": prediction,
        }
    return {"error": "No prediction available", "prediction": None}


@app.get("/suggestions")
async def get_suggestions(count: int = 3):
    """
    Get top tool suggestions based on current context.

    Args:
        count: Number of suggestions to return (default 3)

    Returns:
        List of tool suggestions with confidence
    """
    if not server.brain_b or not server.brain_b.is_available:
        return {"error": "Brain B not available", "suggestions": []}

    handler = server.brain_b.handler
    if not handler or not hasattr(handler, 'get_tool_suggestions'):
        return {"error": "Predictor not available", "suggestions": []}

    suggestions = handler.get_tool_suggestions(count)
    return {
        "status": "ok",
        "suggestions": suggestions,
    }


# ============================================================================
# Training Dashboard API
# ============================================================================

@app.get("/training/status")
async def training_status():
    """Get current training status and metrics."""
    if not HAS_AUTO_TRAINER:
        return {"error": "Auto-trainer not available", "status": None}

    trainer = get_auto_trainer(TrainingConfig(
        episodes_dir=str(Path(__file__).parent.parent / "continuonbrain" / "rlds" / "episodes"),
        models_dir=str(Path(__file__).parent.parent / "brain_b" / "brain_b_data" / "models"),
    ))
    return {
        "status": "ok",
        "training": trainer.get_status(),
    }


@app.get("/training/history")
async def training_history():
    """Get training history (loss/accuracy over epochs)."""
    if not HAS_AUTO_TRAINER:
        return {"error": "Auto-trainer not available", "history": []}

    trainer = get_auto_trainer()
    return {
        "status": "ok",
        "history": trainer.get_training_history(),
    }


@app.get("/training/model")
async def training_model():
    """Get information about the current model."""
    if not HAS_AUTO_TRAINER:
        return {"error": "Auto-trainer not available", "model": None}

    trainer = get_auto_trainer()
    return {
        "status": "ok",
        "model": trainer.get_model_info(),
    }


@app.post("/training/retrain")
async def trigger_retrain(force: bool = False):
    """
    Trigger model retraining.

    Args:
        force: If True, train even if not enough new episodes

    Returns:
        Status of the retrain request
    """
    if not HAS_AUTO_TRAINER:
        return {"error": "Auto-trainer not available", "result": None}

    trainer = get_auto_trainer()
    result = trainer.trigger_retrain(force=force)
    return {
        "status": "ok",
        "result": result,
    }


# ============================================================================
# Simulator Training API
# ============================================================================

@app.get("/simulator/training/status")
async def simulator_training_status():
    """Get simulator training status and available data."""
    if not HAS_SIM_TRAINING:
        return {"error": "Simulator training not available", "status": None}

    episodes_dir = Path(__file__).parent.parent / "continuonbrain" / "rlds" / "episodes"
    models_dir = Path(__file__).parent.parent / "brain_b" / "brain_b_data" / "simulator_models"

    # Count trainer episodes
    trainer_episodes = 0
    if episodes_dir.exists():
        for ep_dir in episodes_dir.iterdir():
            if ep_dir.is_dir() and ep_dir.name.startswith("trainer_"):
                trainer_episodes += 1

    # Check for existing model
    model_path = models_dir / "simulator_action_model.json"
    model_exists = model_path.exists()

    model_info = {}
    if model_exists:
        try:
            with open(model_path) as f:
                data = json.load(f)
            model_info = {
                "version": data.get("version", "unknown"),
                "num_actions": data.get("num_actions", 0),
                "action_vocab": data.get("action_vocab", []),
                "saved_at": data.get("saved_at", "unknown"),
            }
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"[Server] Could not load model info: {e}")

    return {
        "status": "ok",
        "training_available": True,
        "trainer_episodes": trainer_episodes,
        "model_exists": model_exists,
        "model_info": model_info,
        "episodes_dir": str(episodes_dir),
        "models_dir": str(models_dir),
    }


@app.post("/simulator/training/train")
async def run_simulator_train(epochs: int = 10, batch_size: int = 16):
    """
    Train Brain B on simulator RLDS episodes.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Training results with metrics
    """
    if not HAS_SIM_TRAINING:
        return {"error": "Simulator training not available", "result": None}

    episodes_dir = str(Path(__file__).parent.parent / "continuonbrain" / "rlds" / "episodes")
    output_dir = str(Path(__file__).parent.parent / "brain_b" / "brain_b_data" / "simulator_models")

    try:
        metrics = run_simulator_training(
            episodes_dir=episodes_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            filter_prefix="trainer_",
        )

        return {
            "status": "ok",
            "result": {
                "loss": metrics.loss,
                "accuracy": metrics.accuracy,
                "samples_seen": metrics.samples_seen,
                "episodes_processed": metrics.episodes_processed,
                "action_distribution": metrics.action_distribution,
            },
        }
    except Exception as e:
        return {"error": str(e), "result": None}


@app.get("/simulator/training/predict")
async def simulator_predict_action(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    rotation: float = 0.0,
    obstacle_count: int = 0,
):
    """
    Predict next simulator action based on state.

    Args:
        x, y, z: Robot position
        rotation: Robot rotation in radians
        obstacle_count: Number of obstacles

    Returns:
        Predicted action with confidence
    """
    if not HAS_SIM_TRAINING:
        return {"error": "Simulator training not available", "prediction": None}

    model_path = str(Path(__file__).parent.parent / "brain_b" / "brain_b_data" / "simulator_models" / "simulator_action_model.json")
    predictor = get_simulator_predictor(model_path)

    if not predictor.is_ready:
        return {"error": "Model not trained yet", "prediction": None}

    # Build state vector
    import math
    state_vector = [
        (x + 50) / 100.0,
        (y + 50) / 100.0,
        (z + 50) / 100.0,
        (rotation + math.pi) / (2 * math.pi),
        min(obstacle_count / 10.0, 1.0),
    ]
    # Pad to 32 dims
    state_vector.extend([0.0] * (32 - len(state_vector)))

    action, confidence = predictor.predict_action(state_vector)
    top_k = predictor.predict_top_k(state_vector, k=3)

    return {
        "status": "ok",
        "prediction": {
            "action": action,
            "confidence": confidence,
            "alternatives": [{"action": a, "confidence": c} for a, c in top_k],
        },
        "input_state": {
            "x": x, "y": y, "z": z,
            "rotation": rotation,
            "obstacle_count": obstacle_count,
        },
    }


# ============================================================================
# Unified Training Integration API
# ============================================================================

# Import training integration
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "brain_b"))
    from trainer.simulator_integration import (
        SimulatorTrainingIntegration,
        SimulatorTrainingConfig,
        get_training_integration,
    )
    HAS_TRAINING_INTEGRATION = True
except ImportError as e:
    HAS_TRAINING_INTEGRATION = False
    print(f"[Server] Training integration not available: {e}")


@app.get("/brain-b/training/status")
async def brain_b_training_status():
    """
    Get unified Brain B training status.

    Returns status for both navigation and tool prediction models.
    """
    if not HAS_TRAINING_INTEGRATION:
        return {"error": "Training integration not available", "status": None}

    try:
        integration = get_training_integration()
        status = integration.get_status()
        summary = integration.get_episode_summary()

        return {
            "status": "ok",
            "training_status": status,
            "episode_summary": summary,
            "should_train_navigation": integration.should_train_navigation(),
            "should_train_tools": integration.should_train_tools(),
        }
    except Exception as e:
        return {"error": str(e), "status": None}


@app.post("/brain-b/training/navigation")
async def train_brain_b_navigation(force: bool = False, epochs: int = 10, batch_size: int = 32):
    """
    Train Brain B navigation model on Home3D RLDS episodes.

    Args:
        force: Train even if threshold not reached
        epochs: Training epochs
        batch_size: Batch size

    Returns:
        Training result with metrics
    """
    if not HAS_TRAINING_INTEGRATION:
        return {"error": "Training integration not available", "result": None}

    try:
        integration = get_training_integration()

        # Update config if custom values provided
        integration.config.navigation_epochs = epochs
        integration.config.batch_size = batch_size

        result = integration.train_navigation(force=force)

        return {
            "status": "ok" if result.success else "error",
            "result": {
                "success": result.success,
                "model_type": result.model_type,
                "accuracy": result.accuracy,
                "loss": result.loss,
                "episodes_processed": result.episodes_processed,
                "samples_seen": result.samples_seen,
                "duration_s": result.duration_s,
                "model_path": result.model_path,
                "error": result.error,
            },
        }
    except Exception as e:
        return {"error": str(e), "result": None}


@app.post("/brain-b/training/tools")
async def train_brain_b_tools(force: bool = False, epochs: int = 10, batch_size: int = 32):
    """
    Train Brain B tool predictor on Claude Code RLDS episodes.

    Args:
        force: Train even if threshold not reached
        epochs: Training epochs
        batch_size: Batch size

    Returns:
        Training result with metrics
    """
    if not HAS_TRAINING_INTEGRATION:
        return {"error": "Training integration not available", "result": None}

    try:
        integration = get_training_integration()

        # Update config if custom values provided
        integration.config.tool_epochs = epochs
        integration.config.batch_size = batch_size

        result = integration.train_tools(force=force)

        return {
            "status": "ok" if result.success else "error",
            "result": {
                "success": result.success,
                "model_type": result.model_type,
                "accuracy": result.accuracy,
                "loss": result.loss,
                "episodes_processed": result.episodes_processed,
                "samples_seen": result.samples_seen,
                "duration_s": result.duration_s,
                "model_path": result.model_path,
                "error": result.error,
            },
        }
    except Exception as e:
        return {"error": str(e), "result": None}


@app.post("/brain-b/training/all")
async def train_brain_b_all(force: bool = False):
    """
    Train all Brain B models if needed.

    Args:
        force: Train even if thresholds not reached

    Returns:
        Results for each model type
    """
    if not HAS_TRAINING_INTEGRATION:
        return {"error": "Training integration not available", "results": None}

    try:
        integration = get_training_integration()
        results = integration.train_all(force=force)

        return {
            "status": "ok",
            "results": {
                model_type: {
                    "success": r.success,
                    "accuracy": r.accuracy,
                    "episodes_processed": r.episodes_processed,
                    "error": r.error,
                }
                for model_type, r in results.items()
            },
        }
    except Exception as e:
        return {"error": str(e), "results": None}


@app.post("/brain-b/training/convert-episodes")
async def convert_nav_to_tool_episodes():
    """
    Convert navigation episodes to tool prediction format.

    This allows the tool predictor to learn from navigation patterns.

    Returns:
        Number of episodes converted
    """
    if not HAS_TRAINING_INTEGRATION:
        return {"error": "Training integration not available", "converted": 0}

    try:
        integration = get_training_integration()
        converted = integration.convert_navigation_to_tool_episodes()

        return {
            "status": "ok",
            "converted": converted,
            "message": f"Converted {converted} navigation episodes to tool format",
        }
    except Exception as e:
        return {"error": str(e), "converted": 0}


# ============================================================================
# Room Scanner API - Image to 3D Asset Pipeline
# ============================================================================

# Store scan results
_scan_results: Dict[str, dict] = {}


@app.post("/room/scan")
async def scan_room_images(request_data: dict):
    """
    Process room images and generate 3D assets.

    Request body:
        {
            "images": ["base64_image_data", ...],
            "room_scale": 10.0,       # Optional, world scale factor
            "use_fastsam": true,      # BEST: FastSAM on Hailo-8 (48 FPS zero-shot segmentation)
            "use_hailo_sam3": true,   # Hailo-8 detection + SAM3 segmentation (slow)
            "use_hailo": true,        # Optional, use Hailo-8 NPU only (fast YOLOv5)
            "use_sam3": true,         # Optional, use SAM3 only (very slow on CPU)
        }

    Priority: use_fastsam > use_hailo_sam3 > use_hailo > use_sam3 > OpenCV

    Returns:
        Scan result with detected objects and generated assets
    """
    if not HAS_ROOM_SCANNER:
        return {"error": "Room scanner not available", "result": None}

    images = request_data.get("images", [])
    room_scale = request_data.get("room_scale", 10.0)
    use_fastsam = request_data.get("use_fastsam", False)
    use_hailo_sam3 = request_data.get("use_hailo_sam3", False)
    use_hailo = request_data.get("use_hailo", False)
    use_sam3 = request_data.get("use_sam3", False)

    if not images:
        return {"error": "No images provided", "result": None}

    try:
        # Priority: FastSAM > Hailo+SAM3 > Hailo > SAM3 > OpenCV
        if use_fastsam and HAS_HAILO_SCANNER:
            try:
                print("[RoomScanner] Using FastSAM on Hailo-8 (48 FPS)")
                result = scan_with_fastsam(images[0], room_scale)
            except Exception as e:
                print(f"[RoomScanner] FastSAM failed, trying YOLOv5: {e}")
                try:
                    result = scan_with_hailo(images[0], room_scale)
                except Exception as e2:
                    print(f"[RoomScanner] Hailo also failed, using OpenCV: {e2}")
                    result = process_room_images(images, room_scale)
        elif use_hailo_sam3 and HAS_HAILO_SCANNER:
            try:
                print("[RoomScanner] Using Hailo-8 + SAM3 (hybrid)")
                result = await scan_with_hailo_sam3(images[0], room_scale)
            except Exception as e:
                print(f"[RoomScanner] Hailo+SAM3 failed, trying Hailo only: {e}")
                try:
                    result = scan_with_hailo(images[0], room_scale)
                except Exception as e2:
                    print(f"[RoomScanner] Hailo also failed, using OpenCV: {e2}")
                    result = process_room_images(images, room_scale)
        elif use_hailo and HAS_HAILO_SCANNER:
            try:
                print("[RoomScanner] Using Hailo-8 NPU")
                result = scan_with_hailo(images[0], room_scale)
            except Exception as e:
                print(f"[RoomScanner] Hailo failed, falling back to OpenCV: {e}")
                result = process_room_images(images, room_scale)
        elif use_sam3:
            try:
                scanner = get_room_scanner()
                result = await scan_with_sam3(images, room_scale, scanner)
            except Exception as e:
                print(f"[RoomScanner] SAM3 failed, falling back to OpenCV: {e}")
                result = process_room_images(images, room_scale)
        else:
            result = process_room_images(images, room_scale)

        # Store result for later retrieval
        _scan_results[result["scan_id"]] = result

        return {
            "status": "ok",
            "result": result,
        }
    except Exception as e:
        return {"error": str(e), "result": None}


@app.post("/room/analyze-frame")
async def analyze_frame(request_data: dict):
    """
    Analyze a single frame for guided room scanning.

    Detects floor, walls, ceiling boundaries and provides
    real-time guidance for capturing optimal room images.

    Request body:
        {
            "image": "base64_image_data"
        }

    Returns:
        Detected boundaries, coverage info, and user guidance
    """
    if not HAS_ROOM_SCANNER:
        return {"error": "Room scanner not available"}

    image_data = request_data.get("image", "")
    if not image_data:
        return {"error": "No image provided"}

    try:
        from room_scanner import analyze_frame_for_guided_scan
        result = analyze_frame_for_guided_scan(image_data)
        return {"status": "ok", **result}
    except Exception as e:
        return {"error": str(e)}


@app.get("/room/scan/{scan_id}")
async def get_scan_result(scan_id: str):
    """
    Retrieve a previous scan result.

    Args:
        scan_id: The scan ID from a previous scan

    Returns:
        The scan result or error if not found
    """
    if scan_id in _scan_results:
        return {"status": "ok", "result": _scan_results[scan_id]}

    # Try to load from disk
    try:
        scanner = get_room_scanner()
        result_file = scanner.output_dir / f"{scan_id}.json"
        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
            return {"status": "ok", "result": result}
    except Exception as e:
        print(f"[RoomScanner] Error loading scan {scan_id}: {e}")

    return {"error": f"Scan {scan_id} not found", "result": None}


@app.get("/room/scans")
async def list_scans():
    """
    List all available scan results.

    Returns:
        List of scan IDs with metadata
    """
    scans = []

    # In-memory scans
    for scan_id, result in _scan_results.items():
        scans.append({
            "scan_id": scan_id,
            "timestamp": result.get("timestamp", ""),
            "assets_count": result.get("assets_generated", 0),
            "source": "memory",
        })

    # Disk scans
    if HAS_ROOM_SCANNER:
        scanner = get_room_scanner()
        for scan_file in scanner.output_dir.glob("scan_*.json"):
            scan_id = scan_file.stem
            if scan_id not in _scan_results:
                try:
                    with open(scan_file) as f:
                        data = json.load(f)
                    scans.append({
                        "scan_id": scan_id,
                        "timestamp": data.get("timestamp", ""),
                        "assets_count": len(data.get("generated_assets", [])),
                        "source": "disk",
                    })
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[RoomScanner] Could not load {scan_file}: {e}")

    return {"status": "ok", "scans": scans}


async def scan_with_sam3(images: list, room_scale: float, scanner) -> dict:
    """
    Use SAM3 for advanced image segmentation.

    This function uses Meta's SAM3 model for open-vocabulary
    object detection and segmentation.
    """
    start_time = time.perf_counter()

    try:
        from transformers import Sam3Processor, Sam3Model
        import torch
        from PIL import Image
        from io import BytesIO
        import numpy as np
    except ImportError:
        raise ImportError("SAM3 requires: pip install transformers torch pillow")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (cached after first load)
    if not hasattr(scan_with_sam3, "_model"):
        print("[RoomScanner] Loading SAM3 model...")
        scan_with_sam3._processor = Sam3Processor.from_pretrained("facebook/sam3")
        scan_with_sam3._model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        print("[RoomScanner] SAM3 model loaded")

    processor = scan_with_sam3._processor
    model = scan_with_sam3._model

    # Object categories to detect
    categories = ["furniture", "table", "chair", "couch", "shelf", "lamp", "plant", "wall", "floor"]

    all_assets = []
    all_detected = []
    scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for img_idx, image_data in enumerate(images):
        # Decode image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = pil_image.size

        # Detect objects for each category
        for category in categories:
            try:
                inputs = processor(images=pil_image, text=category, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.3,
                    mask_threshold=0.5,
                    target_sizes=[[height, width]]
                )[0]

                # Process detected objects
                for i, (mask, box, score) in enumerate(zip(
                    results.get("masks", []),
                    results.get("boxes", []),
                    results.get("scores", [])
                )):
                    if score < 0.3:
                        continue

                    # Get bounding box
                    x1, y1, x2, y2 = box.tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    # Convert to world coordinates
                    norm_x = cx / width
                    norm_y = cy / height
                    world_x = (norm_x - 0.5) * room_scale
                    world_z = (norm_y - 0.5) * room_scale

                    # Estimate size from bounding box
                    obj_width = (x2 - x1) / width * room_scale * 0.5
                    obj_depth = (y2 - y1) / height * room_scale * 0.5

                    from room_scanner import ASSET_TYPES
                    type_info = ASSET_TYPES.get(category, ASSET_TYPES["decoration"])

                    asset = {
                        "asset_id": f"sam3_{img_idx}_{category}_{i}",
                        "asset_type": category,
                        "position": {"x": world_x, "y": type_info["default_size"][2] / 2, "z": world_z},
                        "size": {
                            "width": max(0.3, obj_width),
                            "height": type_info["default_size"][2],
                            "depth": max(0.3, obj_depth)
                        },
                        "rotation": 0.0,
                        "color": type_info["color"],
                        "geometry": type_info["geometry"],
                        "metadata": {
                            "source": "sam3",
                            "confidence": float(score),
                            "category": category,
                        }
                    }
                    all_assets.append(asset)

                    all_detected.append({
                        "object_type": category,
                        "confidence": float(score),
                        "bounding_box": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        "center": [int(cx), int(cy)],
                        "area": int((x2-x1) * (y2-y1)),
                    })

            except Exception as e:
                print(f"[RoomScanner] SAM3 error for {category}: {e}")
                continue

    # Add floor and walls
    from room_scanner import RoomScanner as RS
    temp_scanner = RS()
    room_dims = {"width": room_scale, "height": 3.0, "depth": room_scale}
    room_structure = temp_scanner._generate_room_structure(room_dims)
    all_assets = [
        {
            "asset_id": a.asset_id,
            "asset_type": a.asset_type,
            "position": a.position,
            "size": a.size,
            "rotation": a.rotation,
            "color": a.color,
            "geometry": a.geometry,
            "metadata": getattr(a, "metadata", {}),
        }
        for a in room_structure
    ] + all_assets

    return {
        "scan_id": scan_id,
        "timestamp": datetime.now().isoformat(),
        "images_processed": len(images),
        "objects_detected": len(all_detected),
        "assets_generated": len(all_assets),
        "room_dimensions": room_dims,
        "processing_time_ms": int((time.perf_counter() - start_time) * 1000),
        "assets": all_assets,
        "metadata": {
            "segmentation_model": "sam3",
            "room_scale": room_scale,
        }
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
