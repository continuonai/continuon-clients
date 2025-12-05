"""
ContinuonBrain Robot API server for Pi5 robot arm.
Runs against real hardware by default with optional mock fallback for dev.
"""
import asyncio
import os
import time
from typing import AsyncIterator, Optional
import json
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure repo root on path when launched as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.actuators.pca9685_arm import PCA9685ArmController, ArmConfig
from continuonbrain.actuators.drivetrain_controller import DrivetrainController
from continuonbrain.sensors.oak_depth import OAKDepthCapture, CameraConfig
from continuonbrain.recording.arm_episode_recorder import ArmEpisodeRecorder
from continuonbrain.sensors.hardware_detector import HardwareDetector
from continuonbrain.robot_modes import RobotModeManager, RobotMode
from continuonbrain.gemma_chat import create_gemma_chat
from continuonbrain.system_context import SystemContext
from continuonbrain.system_health import SystemHealthChecker
from continuonbrain.system_instructions import SystemInstructions


class RobotService:
    """
    Robot API server for arm control.
    Prefers real hardware when available with optional mock fallback.
    """
    
    def __init__(
        self,
        config_dir: str = "/tmp/continuonbrain_demo",
        prefer_real_hardware: bool = True,
        auto_detect: bool = True,
        allow_mock_fallback: bool = True,
        system_instructions: Optional[SystemInstructions] = None,
    ):
        self.config_dir = config_dir
        self.prefer_real_hardware = prefer_real_hardware
        self.auto_detect = auto_detect
        self.allow_mock_fallback = allow_mock_fallback
        self.use_real_hardware = False
        self.arm: Optional[PCA9685ArmController] = None
        self.camera: Optional[OAKDepthCapture] = None
        self.recorder: Optional[ArmEpisodeRecorder] = None
        self.drivetrain: Optional[DrivetrainController] = None
        self.mode_manager: Optional[RobotModeManager] = None
        self.is_recording = False
        self.current_episode_id: Optional[str] = None
        self.detected_config: dict = {}
        self.last_drive_result: Optional[dict] = None
        self.system_instructions: Optional[SystemInstructions] = system_instructions or SystemContext.get_instructions()
        self.health_checker = SystemHealthChecker(config_dir=config_dir)

        # Initialize Gemma chat (will use mock if transformers not available)
        self.gemma_chat = create_gemma_chat(use_mock=False)

    def _ensure_system_instructions(self) -> None:
        """Guarantee that merged system instructions are available."""

        if self.system_instructions:
            if SystemContext.get_instructions() is None:
                SystemContext.register_instructions(self.system_instructions)
            return

        env_path = os.environ.get("CONTINUON_SYSTEM_INSTRUCTIONS_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                self.system_instructions = SystemContext.load_and_register(path)
                return

        # Fall back to loading from the configured directory
        self.system_instructions = SystemInstructions.load(Path(self.config_dir))
        SystemContext.register_instructions(self.system_instructions)
        
    async def initialize(self):
        """Initialize hardware components with auto-detection."""
        mode_label = "REAL HARDWARE" if self.prefer_real_hardware else "MOCK"
        print(f"Initializing Robot Service ({mode_label} MODE)...")
        print()

        self._ensure_system_instructions()
        
        # Auto-detect hardware (used for status reporting)
        if self.auto_detect:
            print("🔍 Auto-detecting hardware...")
            detector = HardwareDetector()
            devices = detector.detect_all()
            if devices:
                self.detected_config = detector.generate_config()
                detector.print_summary()
                print()
            else:
                print("⚠️  No hardware detected!")
                print()
        
        # Initialize recorder and hardware (prefers real, falls back to mock if allowed)
        print("📼 Initializing episode recorder...")
        self.recorder = ArmEpisodeRecorder(
            episodes_dir=f"{self.config_dir}/episodes",
            max_steps=500,
        )
        
        hardware_ready = False
        if self.prefer_real_hardware:
            print("🦾 Initializing hardware via ContinuonBrain...")
            hardware_ready = self.recorder.initialize_hardware(
                use_mock=False,
                auto_detect=self.auto_detect,
            )
            self.arm = self.recorder.arm
            self.camera = self.recorder.camera
            
            if not hardware_ready:
                print("⚠️  Real hardware initialization incomplete")
                if not self.allow_mock_fallback:
                    raise RuntimeError("Failed to initialize arm or camera in real mode")
                print("↩️  Falling back to mock mode")
        
        if not hardware_ready:
            # Ensure clean mock state
            self.recorder.initialize_hardware(use_mock=True, auto_detect=self.auto_detect)
            self.arm = None
            self.camera = None
            self.recorder.arm = None
            self.recorder.camera = None
            self.use_real_hardware = False
        else:
            self.use_real_hardware = True
        
        print("✅ Episode recorder ready")

        # Initialize drivetrain controller for steering/throttle
        print("🛞 Initializing drivetrain controller...")
        self.drivetrain = DrivetrainController()
        drivetrain_ready = self.drivetrain.initialize()
        if drivetrain_ready:
            print(f"✅ Drivetrain ready ({self.drivetrain.mode.upper()} MODE)")
        else:
            print("⚠️  Drivetrain controller unavailable")

        # Initialize mode manager
        print("🎮 Initializing mode manager...")
        self.mode_manager = RobotModeManager(
            config_dir=self.config_dir,
            system_instructions=self.system_instructions,
        )
        self.mode_manager.return_to_idle()  # Start in idle mode
        print("✅ Mode manager ready")
        
        print()
        print("=" * 60)
        print(f"✅ Robot Service Ready ({'REAL' if self.use_real_hardware else 'MOCK'} MODE)")
        print("=" * 60)
        if self.use_real_hardware and self.detected_config.get("primary"):
            print("\n🎯 Using detected hardware:")
            for key, value in self.detected_config["primary"].items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        print()
    
    async def StreamRobotState(self, client_id: str) -> AsyncIterator[dict]:
        """
        Stream robot state at ~20Hz.
        
        Yields dict with:
        - timestamp_nanos
        - joint_positions (6 floats, normalized [-1, 1])
        - gripper_open (bool)
        - frame_id (str)
        """
        print(f"Client {client_id} subscribed to robot state stream")
        
        while True:
            try:
                if not self.system_instructions:
                    yield {"success": False, "message": "System instructions unavailable"}
                    return

                # Get current arm state
                normalized_state = self.arm.get_normalized_state() if self.arm else [0.0] * 6
                gripper_open = normalized_state[5] < 0.0  # Gripper: -1.0 = open, 1.0 = closed
                
                state = {
                    "timestamp_nanos": time.time_ns(),
                    "joint_positions": normalized_state,
                    "gripper_open": gripper_open,
                    "frame_id": f"state_{int(time.time())}",
                    "wall_time_millis": int(time.time() * 1000),
                }
                
                yield state
                await asyncio.sleep(0.05)  # 20Hz
                
            except Exception as e:
                print(f"Error streaming state: {e}")
                break
    
    async def SendCommand(self, command: dict) -> dict:
        """
        Accept control command from Flutter.
        
        command dict:
        - client_id (str)
        - control_mode (str): "armJointAngles"
        - arm_joint_angles: {"normalized_angles": [6 floats]}
        """
        try:
            client_id = command.get("client_id", "unknown")
            control_mode = command.get("control_mode")

            if not self.system_instructions:
                return {"success": False, "message": "System instructions unavailable"}

            # Check if motion is allowed in current mode
            if self.mode_manager:
                mode_config = self.mode_manager.get_mode_config(self.mode_manager.current_mode)
                if not mode_config.allow_motion:
                    return {
                        "success": False,
                        "message": f"Motion not allowed in {self.mode_manager.current_mode.value} mode"
                    }
            
            if control_mode == "armJointAngles":
                arm_cmd = command.get("arm_joint_angles", {})
                action = arm_cmd.get("normalized_angles", [0.0] * 6)
                ball_reached = command.get("ball_reached", False)
                safety_violations = command.get("safety_violations")
                step_metadata = command.get("step_metadata")

                # Execute on arm
                if self.arm:
                    self.arm.set_normalized_action(action)

                # Record step if recording and in training mode
                if self.is_recording and self.recorder:
                    action_source = "human_teleop_flutter"
                    if self.mode_manager and self.mode_manager.current_mode == RobotMode.AUTONOMOUS:
                        action_source = "vla_policy"

                    self.recorder.record_step(
                        action=action,
                        action_source=action_source,
                        ball_reached=ball_reached,
                        safety_violations=safety_violations if isinstance(safety_violations, list) else None,
                        step_metadata=step_metadata if isinstance(step_metadata, dict) else None,
                    )
                
                return {
                    "success": True,
                    "latency_ms": 0,
                    "message": f"Executed arm command from {client_id}"
                }
            
            return {
                "success": False,
                "message": f"Unknown control mode: {control_mode}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    async def Drive(self, steering: float, throttle: float) -> dict:
        """Apply drivetrain command with safety checks."""

        def _record_result(result: dict) -> dict:
            self.last_drive_result = result
            return result

        try:
            if not self.system_instructions:
                return _record_result({"success": False, "message": "System instructions unavailable"})

            if not self.mode_manager:
                return _record_result({"success": False, "message": "Mode manager not initialized"})

            current_mode = self.mode_manager.current_mode
            if current_mode not in {RobotMode.MANUAL_CONTROL, RobotMode.MANUAL_TRAINING}:
                return _record_result(
                    {
                        "success": False,
                        "message": "Driving allowed only in manual control or manual training modes",
                        "mode": current_mode.value,
                    }
                )

            mode_config = self.mode_manager.get_mode_config(current_mode)
            if not mode_config.allow_motion:
                return _record_result(
                    {
                        "success": False,
                        "message": f"Motion not allowed in {current_mode.value} mode",
                        "mode": current_mode.value,
                    }
                )

            try:
                steering_value = float(steering)
                throttle_value = float(throttle)
            except (TypeError, ValueError):
                return _record_result(
                    {
                        "success": False,
                        "message": "Steering and throttle must be numeric",
                    }
                )

            if not self.drivetrain:
                return _record_result(
                    {
                        "success": False,
                        "message": "Drivetrain controller not available",
                        "steering": steering_value,
                        "throttle": throttle_value,
                    }
                )

            drive_result = self.drivetrain.apply_drive(steering_value, throttle_value)
            if "mode" not in drive_result:
                drive_result["mode"] = self.drivetrain.mode

            return _record_result(drive_result)

        except Exception as e:
            return _record_result({"success": False, "message": f"Error: {str(e)}"})
    
    async def SetRobotMode(self, mode: str) -> dict:
        """Change robot operational mode."""
        try:
            if not self.system_instructions:
                return {"success": False, "message": "System instructions unavailable"}

            if not self.mode_manager:
                return {"success": False, "message": "Mode manager not initialized"}
            
            # Map string to enum
            mode_map = {
                "manual_control": RobotMode.MANUAL_CONTROL,
                "manual_training": RobotMode.MANUAL_TRAINING,
                "autonomous": RobotMode.AUTONOMOUS,
                "sleep_learning": RobotMode.SLEEP_LEARNING,
                "idle": RobotMode.IDLE,
                "emergency_stop": RobotMode.EMERGENCY_STOP,
            }
            
            robot_mode = mode_map.get(mode)
            if not robot_mode:
                return {"success": False, "message": f"Unknown mode: {mode}"}
            
            success = self.mode_manager.set_mode(robot_mode)
            
            return {
                "success": success,
                "mode": robot_mode.value,
                "message": f"Mode changed to {robot_mode.value}" if success else "Mode change failed"
            }
        
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def GetRobotStatus(self) -> dict:
        """Get robot status including mode and capabilities."""
        try:
            status = {
                "robot_name": "ContinuonBot",
                "is_recording": self.is_recording,
                "current_episode": self.current_episode_id,
                "hardware_mode": "real" if self.use_real_hardware else "mock",
                "audio_recording_active": bool(self.recorder and self.recorder.audio_enabled),
            }

            instructions = self.system_instructions or SystemContext.get_instructions()
            if instructions:
                status["system_instructions"] = instructions.as_dict()

            if self.mode_manager:
                mode_status = self.mode_manager.get_status()
                status.update({
                    "mode": mode_status["mode"],
                    "mode_duration": mode_status["duration_seconds"],
                    "allow_motion": mode_status["config"]["allow_motion"],
                    "recording_enabled": mode_status["config"]["record_episodes"],
                })
                status["gate_snapshot"] = self.mode_manager.get_gate_snapshot()
                status["loop_metrics"] = self.mode_manager.get_loop_metrics()

            if self.detected_config:
                status["detected_hardware"] = self.detected_config.get("primary")

            if self.arm:
                status["joint_positions"] = self.arm.get_normalized_state()

            if self.drivetrain:
                drivetrain_hardware_available = not self.drivetrain.is_mock
                status["drivetrain"] = {
                    "connected": self.drivetrain.initialized and drivetrain_hardware_available,
                    "hardware_available": drivetrain_hardware_available,
                    "mode": self.drivetrain.mode,
                    "message": "PCA9685 output inactive (mock mode)" if self.drivetrain.is_mock else "Drivetrain ready",
                    "last_command": self.last_drive_result or self.drivetrain.last_command,
                }

            if self.health_checker:
                status["safety_head"] = self.health_checker.get_safety_head_status()

            return {"success": True, "status": status}
        
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    async def GetLoopHealth(self) -> dict:
        """Expose HOPE/CMS loop metrics, safety head status, and gates."""
        try:
            if not self.mode_manager:
                return {"success": False, "message": "Mode manager not initialized"}

            metrics = self.mode_manager.get_loop_metrics()
            gates = self.mode_manager.get_gate_snapshot()
            safety_head = self.health_checker.get_safety_head_status() if self.health_checker else None

            return {
                "success": True,
                "metrics": metrics,
                "gates": gates,
                "safety_head": safety_head,
            }
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def StartEpisodeRecording(self, language_instruction: str) -> dict:
        """Start RLDS episode recording."""
        try:
            if self.is_recording:
                return {
                    "success": False,
                    "message": "Already recording"
                }
            
            episode_id = self.recorder.start_episode(
                language_instruction=language_instruction,
                action_source="human_teleop_flutter",
            )
            
            self.is_recording = True
            self.current_episode_id = episode_id
            
            return {
                "success": True,
                "episode_id": episode_id,
                "message": f"Started episode: {episode_id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    async def StopEpisodeRecording(self, success: bool = True) -> dict:
        """Stop RLDS episode recording."""
        try:
            if not self.is_recording:
                return {
                    "success": False,
                    "message": "Not recording"
                }
            
            episode_path = self.recorder.end_episode(success=success)
            
            self.is_recording = False
            episode_id = self.current_episode_id
            self.current_episode_id = None
            
            return {
                "success": True,
                "episode_id": episode_id,
                "episode_path": str(episode_path) if episode_path else None,
                "message": f"Saved episode: {episode_id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    async def GetDepthFrame(self) -> Optional[dict]:
        """Get latest depth camera frame."""
        if not self.camera:
            return None
        
        frame = self.camera.capture_frame()
        if not frame:
            return None
        
        return {
            "timestamp_nanos": frame["timestamp_ns"],
            "rgb_shape": frame["rgb"].shape,
            "depth_shape": frame["depth"].shape,
            # Note: In real implementation, would encode images as bytes
            "has_data": True,
        }
    
    async def GetCameraFrameJPEG(self) -> Optional[bytes]:
        """Get latest RGB camera frame as JPEG bytes."""
        if not self.camera:
            return None
        
        try:
            frame = self.camera.capture_frame()
            if not frame or 'rgb' not in frame:
                return None
            
            # Convert BGR to RGB and encode as JPEG
            rgb_frame = frame['rgb']
            # OAK-D outputs BGR, need to convert to RGB for proper display
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            
            # Encode as JPEG with quality 85
            success, jpeg_bytes = cv2.imencode('.jpg', rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                return jpeg_bytes.tobytes()
            return None
        except Exception as e:
            print(f"Error encoding camera frame: {e}")
            return None
    
    async def ChatWithGemma(self, message: str, history: list) -> dict:
        """
        Chat with Gemma 3n model about robot control and status.
        
        Args:
            message: User's message
            history: Chat history for context
            
        Returns:
            dict with 'response' or 'error'
        """
        try:
            # Get current robot status for context
            status_data = await self.GetRobotStatus()
            
            # Build context from robot status
            context_parts = []
            if status_data.get('success') and status_data.get('status'):
                status = status_data['status']
                context_parts.append(f"Current mode: {status.get('mode', 'unknown')}")
                context_parts.append(f"Motion allowed: {status.get('allow_motion', False)}")
                context_parts.append(f"Recording: {status.get('is_recording', False)}")
                context_parts.append(f"Hardware: {status.get('hardware_mode', 'unknown')}")
                
                if status.get('joint_positions'):
                    joints_str = ', '.join([f"J{i}:{v:.2f}" for i, v in enumerate(status['joint_positions'])])
                    context_parts.append(f"Joint positions: {joints_str}")
            
            robot_context = " | ".join(context_parts)
            
            # Try to use real Gemma model, fall back to simple responses
            try:
                # Use the Gemma chat instance for real AI responses
                response = self.gemma_chat.chat(message, system_context=robot_context)
            except Exception as gemma_error:
                print(f"Gemma chat error: {gemma_error}, using fallback")
                # Fallback to simple keyword-based responses
                response = self._generate_gemma_response(message, robot_context)
            
            return {"response": response}
            
        except Exception as e:
            print(f"Error in ChatWithGemma: {e}")
            return {"error": str(e)}
    
    def _generate_gemma_response(self, message: str, context: str) -> str:
        """Generate a helpful response based on the message and context."""
        msg_lower = message.lower()
        
        # Status queries
        if any(word in msg_lower for word in ['status', 'state', 'how', 'what']):
            return f"Robot status: {context}. The robot is ready for your commands. Use the arrow controls or keyboard to move the arm joints and drive the car."
        
        # Control help
        if any(word in msg_lower for word in ['control', 'move', 'drive', 'steer']):
            return "Control the arm with joint sliders or arrow buttons. For the car, use the driving controls - default speed is set to SLOW (0.3) for safety. Hold Ctrl+Arrow keys for keyboard driving, or use WASD for arm control."
        
        # Joint control
        if any(word in msg_lower for word in ['joint', 'arm', 'gripper']):
            return "The arm has 6 joints: J0 (base rotation), J1 (shoulder), J2 (elbow), J3 (wrist roll), J4 (wrist pitch), and J5 (gripper). Use the sliders or arrow buttons to control each joint. Values range from -1.0 to 1.0."
        
        # Car driving
        if any(word in msg_lower for word in ['car', 'speed', 'throttle']):
            return "The car is based on a DonkeyCar RC platform. Speed is preset to SLOW (0.3) for safety - you can adjust using the speed buttons (Crawl, Slow, Med, Fast). Use arrow buttons or keyboard to steer and control throttle."
        
        # Recording
        if any(word in msg_lower for word in ['record', 'episode', 'training']):
            return "Episode recording captures your manual control demonstrations for training. Make sure you're in manual_training mode and motion is enabled. Your actions will be recorded as RLDS episodes."
        
        # Safety
        if any(word in msg_lower for word in ['safe', 'stop', 'emergency']):
            return "For safety, the speed is preset to SLOW. Use the Emergency Stop button if needed - it will halt all motion immediately. Always start with slow movements to test the robot's response."
        
        # Default helpful response
        return f"I'm here to help with robot control! Current status: {context}. Ask me about controls, status, movement, or safety."
    
    def shutdown(self):
        """Graceful shutdown."""
        print("Shutting down Robot Service...")
        
        if self.is_recording and self.recorder:
            self.recorder.end_episode(success=False)
        
        if self.recorder:
            self.recorder.shutdown()
        
        if self.camera:
            self.camera.stop()
        
        if self.arm:
            self.arm.shutdown()
        
        print("✅ Shutdown complete")


class SimpleJSONServer:
    """
    HTTP/JSON server for robot control and web UI.
    Supports both HTTP endpoints and raw JSON protocol.
    """
    
    def __init__(self, service: RobotService):
        self.service = service
        self.server = None
    
    async def handle_http_request(self, request_line: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle HTTP request and return HTML/JSON response."""
        # Parse request line
        parts = request_line.split()
        method = parts[0] if len(parts) > 0 else "GET"
        full_path = parts[1] if len(parts) > 1 else "/"
        
        # Strip query string for routing
        path = full_path.split('?')[0]
        
        print(f"[HTTP] {method} {path}")
        
        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if not line or line == b'\r\n' or line == b'\n':
                break
            header_line = line.decode().strip()
            if ':' in header_line:
                key, value = header_line.split(':', 1)
                headers[key.strip().lower()] = value.strip()
        
        # Route the request
        if path == "/" or path == "/ui":
            response_body = self.get_web_ui_html()
            response_bytes = response_body.encode('utf-8')
            response = f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/control":
            # Set mode to manual_control and show live control interface
            await self.service.SetRobotMode("manual_control")
            response_body = self.get_control_interface_html()
            response_bytes = response_body.encode('utf-8')
            response = f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/status":
            status = await self.service.GetRobotStatus()
            response_body = json.dumps(status, indent=2)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/status":
            status = await self.service.GetRobotStatus()
            response_body = json.dumps(status)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/loops":
            status = await self.service.GetLoopHealth()
            response_body = json.dumps(status)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path.startswith("/api/mode/"):
            mode = path.split("/")[-1]
            print(f"[MODE] Changing to: {mode}")
            result = await self.service.SetRobotMode(mode)
            print(f"[MODE] Result: {result}")
            response_body = json.dumps(result)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/camera/frame":
            # Get latest camera frame as JPEG
            frame_data = await self.service.GetCameraFrameJPEG()
            if frame_data:
                response = f"HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(frame_data)}\r\n\r\n".encode('utf-8') + frame_data
            else:
                response_body = "No camera frame available"
                response = f"HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/plain\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/command" and method == "POST":
            # Read POST body
            content_length = int(headers.get('content-length', 0))
            body = await reader.read(content_length) if content_length > 0 else b''
            try:
                command = json.loads(body.decode())
                result = await self.service.SendCommand(command)
                response_body = json.dumps(result)
                response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as e:
                response_body = json.dumps({"success": False, "message": str(e)})
                response = f"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/drive" and method == "POST":
            # Read POST body for car driving
            content_length = int(headers.get('content-length', 0))
            body = await reader.read(content_length) if content_length > 0 else b''
            try:
                drive_cmd = json.loads(body.decode())
                steering = drive_cmd.get('steering', 0.0)
                throttle = drive_cmd.get('throttle', 0.0)

                result = await self.service.Drive(steering, throttle)
                response_body = json.dumps(result)
                response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as e:
                response_body = json.dumps({"success": False, "message": str(e)})
                response = f"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/chat" and method == "POST":
            # Read POST body for chat message
            content_length = int(headers.get('content-length', 0))
            body = await reader.read(content_length) if content_length > 0 else b''
            try:
                chat_data = json.loads(body.decode())
                message = chat_data.get('message', '')
                history = chat_data.get('history', [])
                
                # Get chat response from Gemma
                result = await self.service.ChatWithGemma(message, history)
                response_body = json.dumps(result)
                response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as e:
                response_body = json.dumps({"error": str(e)})
                response = f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        else:
            response_body = "404 Not Found"
            response = f"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        
        # Write response (handle both bytes and string)
        if isinstance(response, bytes):
            writer.write(response)
        else:
            writer.write(response.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    
    def get_web_ui_html(self):
        """Generate simple web UI for robot control."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CraigBot Control</title>
    <style>
        :root {
            --bg: #0c111b;
            --panel: #0f1729;
            --panel-glow: rgba(0, 170, 255, 0.15);
            --border: #1f2a3d;
            --text: #e8f0ff;
            --muted: #7f8ba7;
            --accent: #7ad7ff;
            --accent-strong: #4f9dff;
            --danger: #ff4d6d;
            --success: #38d996;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 0;
            background: radial-gradient(circle at 20% 20%, rgba(74, 217, 255, 0.08), transparent 25%),
                        radial-gradient(circle at 80% 10%, rgba(137, 90, 255, 0.08), transparent 22%),
                        radial-gradient(circle at 50% 70%, rgba(56, 217, 150, 0.06), transparent 30%),
                        var(--bg);
            color: var(--text);
        }

        .ide-shell {
            max-width: 1100px;
            margin: 0 auto;
            padding: 28px 22px 36px 22px;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .ide-topbar {
            background: linear-gradient(135deg, rgba(12, 17, 27, 0.9), rgba(19, 27, 43, 0.9));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.02);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-mark {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            background: radial-gradient(circle at 30% 30%, rgba(122, 215, 255, 0.4), rgba(79, 157, 255, 0.15));
            display: grid;
            place-items: center;
            font-size: 22px;
            box-shadow: 0 0 0 1px var(--border);
        }

        .brand-title {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 0.2px;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: 12px;
            margin-top: 2px;
        }

        .top-status {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .chip {
            padding: 10px 14px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.02);
            color: var(--text);
            font-weight: 600;
            font-size: 13px;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
        }

        .ide-workspace {
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 18px;
        }

        .ide-sidebar {
            background: linear-gradient(180deg, rgba(18, 28, 44, 0.9), rgba(15, 23, 41, 0.95));
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .sidebar-title {
            font-size: 13px;
            letter-spacing: 0.4px;
            text-transform: uppercase;
            color: var(--muted);
        }

        .command-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
        }

        .command-btn {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 14px;
            text-align: left;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.08s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .command-btn:hover {
            border-color: var(--accent);
            box-shadow: 0 10px 28px var(--panel-glow);
            transform: translateY(-1px);
        }

        .command-btn.primary { background: linear-gradient(135deg, #0aa4ff, #4f9dff); border-color: #1c80ff; color: #0b1020; }
        .command-btn.subtle { opacity: 0.8; }
        .command-btn.danger { background: linear-gradient(135deg, #ff4d6d, #ff7b7b); border-color: #ff4d6d; color: #0b1020; }

        .sidebar-footnote {
            color: var(--muted);
            font-size: 12px;
            line-height: 1.5;
        }

        .ide-main {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .panel {
            background: rgba(14, 21, 35, 0.92);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.35);
        }

        .panel-header h2 {
            margin: 4px 0;
            font-size: 22px;
        }

        .panel-eyebrow {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.4px;
            font-size: 11px;
        }

        .panel-subtitle {
            color: var(--muted);
            margin: 4px 0 0 0;
            font-size: 13px;
        }

        .status-deck {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin-top: 12px;
        }

        .status-card {
            padding: 14px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.02), rgba(122, 215, 255, 0.04));
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
        }

        .status-label { color: var(--muted); font-size: 12px; letter-spacing: 0.3px; }
        .status-value { font-size: 18px; font-weight: 700; margin-top: 6px; }

        .status-item {
            padding: 12px;
            border-radius: 10px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.02);
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
        }

        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .canvas-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .canvas-card {
            padding: 14px;
            border-radius: 12px;
            border: 1px dashed var(--border);
            background: linear-gradient(180deg, rgba(15, 23, 41, 0.9), rgba(18, 26, 46, 0.9));
        }

        .canvas-title { font-size: 14px; font-weight: 700; margin-bottom: 6px; }
        .canvas-text { color: var(--muted); font-size: 13px; line-height: 1.5; margin: 0; }

        .inline-status {
            margin-top: 12px;
            padding: 12px;
            border-radius: 10px;
            background: linear-gradient(90deg, rgba(79, 157, 255, 0.12), rgba(122, 215, 255, 0.08));
            border: 1px solid var(--border);
            text-align: center;
        }

        .badge { 
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            background: #34c759;
            color: #0b1020;
        }
        .badge.idle { background: #86868b; color: #0b1020; }
        .badge.training { background: #007aff; color: #0b1020; }
        .badge.autonomous { background: #af52de; color: #0b1020; }
        .badge.sleeping { background: #ff9500; color: #0b1020; }

        .chip.success { background: rgba(56, 217, 150, 0.12); color: #8df5c7; border-color: rgba(56, 217, 150, 0.4); }
        .chip.danger { background: rgba(255, 77, 109, 0.12); color: #ff99ae; border-color: rgba(255, 77, 109, 0.4); }

        .loop-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
            margin-top: 12px;
        }

        .loop-card {
            padding: 14px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: linear-gradient(160deg, rgba(122, 215, 255, 0.06), rgba(79, 157, 255, 0.04));
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
        }

        .loop-title { font-size: 14px; font-weight: 700; margin-bottom: 6px; }
        .loop-meta { color: var(--muted); font-size: 12px; }

        .gauge-bar {
            position: relative;
            height: 12px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.06);
            overflow: hidden;
            border: 1px solid var(--border);
            margin: 8px 0;
        }

        .gauge-fill {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #4f9dff, #7ad7ff);
            transition: width 0.25s ease;
        }

        .safety-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .badge-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
            background: #38d996;
            box-shadow: 0 0 0 4px rgba(56, 217, 150, 0.12);
        }
    </style>
</head>
<body class="ide-body">
    <div class="ide-shell">
        <header class="ide-topbar">
            <div class="brand">
                <div class="brand-mark">🤖</div>
                <div>
                    <div class="brand-title">Robot Editor</div>
                    <div class="brand-subtitle">ContinuonBrain live console</div>
                </div>
            </div>
            <div class="top-status">
                <div class="chip" id="mode">Loading...</div>
                <div class="chip" id="recording">No</div>
                <div class="chip" id="motion">No</div>
            </div>
        </header>

        <div class="ide-workspace">
            <aside class="ide-sidebar">
                <div class="sidebar-title">Command Deck</div>
                <div class="command-grid">
                    <button class="command-btn primary" onclick="window.location.href='/control'">🎮 Manual Control</button>
                    <button class="command-btn" onclick="setMode('manual_training')">📝 Manual Training</button>
                    <button class="command-btn" onclick="setMode('autonomous')">🚀 Autonomous</button>
                    <button class="command-btn" onclick="setMode('sleep_learning')">💤 Sleep Learning</button>
                    <button class="command-btn subtle" onclick="setMode('idle')">⏸️ Idle</button>
                    <button class="command-btn danger" onclick="setMode('emergency_stop')">🛑 Emergency Stop</button>
                    <button class="command-btn" onclick="window.triggerSafetyHold()">🛡️ Safety Hold</button>
                    <button class="command-btn subtle" onclick="window.resetSafetyGates()">♻️ Reset Gates</button>
                </div>
                <div class="sidebar-footnote">Use the deck like an IDE command palette to swap modes quickly.</div>
            </aside>

            <main class="ide-main">
                <section class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-eyebrow">Live State</div>
                            <h2>Robot Health Overview</h2>
                            <p class="panel-subtitle">Visualize safety status, recording posture, and motion gates.</p>
                        </div>
                    </div>
                    <div class="status-deck">
                        <div class="status-card">
                            <div class="status-label">Robot Mode</div>
                            <div class="status-value" id="mode-card">mirrors mode badge</div>
                        </div>
                        <div class="status-card">
                            <div class="status-label">Recording</div>
                            <div class="status-value" id="recording-card">No</div>
                        </div>
                        <div class="status-card">
                            <div class="status-label">Motion Allowed</div>
                            <div class="status-value" id="motion-card">No</div>
                        </div>
                    </div>
                </section>

                <section class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-eyebrow">HOPE / CMS</div>
                            <h2>Loop Telemetry & Safety</h2>
                            <p class="panel-subtitle">Wave/particle balance, safety envelopes, and gate heartbeats.</p>
                        </div>
                    </div>
                    <div class="loop-grid">
                        <div class="loop-card">
                            <div class="loop-title">Wave / Particle</div>
                            <div class="gauge-bar"><div class="gauge-fill" id="wave-meter"></div></div>
                            <div class="loop-meta" id="wave-value">--</div>
                        </div>
                        <div class="loop-card">
                            <div class="loop-title">HOPE Loops</div>
                            <div class="loop-meta" id="hope-fast">Fast: --</div>
                            <div class="loop-meta" id="hope-mid">Mid: --</div>
                            <div class="loop-meta" id="hope-slow">Slow: --</div>
                        </div>
                        <div class="loop-card">
                            <div class="loop-title">CMS Balance</div>
                            <div class="loop-meta" id="cms-ratio">--</div>
                            <div class="loop-meta" id="cms-buffer">Buffer: --</div>
                            <div class="chip" id="heartbeat-badge">Heartbeat...</div>
                        </div>
                    </div>
                    <div class="safety-grid">
                        <div class="status-item">
                            <span class="status-label">Safety Head</span>
                            <span class="status-value" id="safety-head-path">loading...</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Envelope</span>
                            <span class="status-value" id="safety-envelope">--</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Motion Gate</span>
                            <span class="status-value" id="gate-allow">--</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Recording Gate</span>
                            <span class="status-value" id="gate-record">--</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Safety Heartbeat</span>
                            <span class="status-value" id="safety-heartbeat">--</span>
                        </div>
                    </div>
                </section>

                <section class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-eyebrow">Sensors</div>
                            <h2>Hardware Canvas</h2>
                            <p class="panel-subtitle">Auto-discovered sensors render into a visual rack.</p>
                        </div>
                    </div>
                    <div class="sensor-grid" id="hardware-status">
                        <div class="status-item">
                            <span class="status-label">Loading sensors...</span>
                        </div>
                    </div>
                </section>

                <section class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-eyebrow">Workspace</div>
                            <h2>Editor Canvas</h2>
                            <p class="panel-subtitle">A visual staging area that mirrors robot readiness.</p>
                        </div>
                    </div>
                    <div class="canvas-grid">
                        <div class="canvas-card">
                            <div class="canvas-title">Mode Timeline</div>
                            <p class="canvas-text">Snapshot of the current behavior lane; switch to Manual Control for a live scene.</p>
                        </div>
                        <div class="canvas-card">
                            <div class="canvas-title">Safety Boundaries</div>
                            <p class="canvas-text">Motion gates, emergency stops, and recording toggles stay front and center in this editor skin.</p>
                        </div>
                        <div class="canvas-card">
                            <div class="canvas-title">Hardware Dock</div>
                            <p class="canvas-text">Detected cameras and controllers render as modules so you can reason about availability before deploying changes.</p>
                        </div>
                    </div>
                </section>

                <div id="status-message" class="inline-status" style="display: none;"></div>
            </main>
        </div>
    </div>
    
    <script type="text/javascript">
        // Global functions for onclick handlers
        window.showMessage = function(message, isError) {
            if (typeof isError === 'undefined') { isError = false; }
            var msgDiv = document.getElementById('status-message');
            msgDiv.textContent = message;
            msgDiv.style.display = 'block';
            msgDiv.style.background = isError ? '#ff3b30' : '#34c759';
            msgDiv.style.color = 'white';
            msgDiv.style.textAlign = 'center';
            setTimeout(function() {
                msgDiv.style.display = 'none';
            }, 3000);
        };

        window.triggerSafetyHold = function() {
            window.showMessage('Engaging safety hold...');
            setMode('emergency_stop');
        };

        window.resetSafetyGates = function() {
            window.showMessage('Resetting gates to idle baseline...');
            setMode('idle');
        };

        function renderLoopTelemetry(status) {
            var loops = status.loop_metrics || {};
            var gates = status.gate_snapshot || {};
            var safety = status.safety_head || {};

            var wave = (typeof loops.wave_particle_balance === 'number') ? Math.min(1, Math.max(0, loops.wave_particle_balance)) : 0;
            var waveFill = document.getElementById('wave-meter');
            if (waveFill) {
                waveFill.style.width = Math.round(wave * 100) + '%';
            }
            var waveLabel = document.getElementById('wave-value');
            if (waveLabel) {
                var wavePercent = Math.round(wave * 100);
                waveLabel.textContent = wavePercent + '% wave / ' + (100 - wavePercent) + '% particle';
            }

            var hope = loops.hope_loops || {};
            var fast = hope.fast || {};
            var mid = hope.mid || {};
            var slow = hope.slow || {};
            var hopeFast = document.getElementById('hope-fast');
            if (hopeFast) { hopeFast.textContent = 'Fast: ' + (fast.hz ? fast.hz + ' Hz (' + fast.latency_ms + ' ms)' : '--'); }
            var hopeMid = document.getElementById('hope-mid');
            if (hopeMid) { hopeMid.textContent = 'Mid: ' + (mid.hz ? mid.hz + ' Hz (' + mid.latency_ms + ' ms)' : '--'); }
            var hopeSlow = document.getElementById('hope-slow');
            if (hopeSlow) { hopeSlow.textContent = 'Slow: ' + (slow.hz ? slow.hz + ' Hz (' + slow.latency_ms + ' ms)' : '--'); }

            var cms = loops.cms || {};
            var cmsRatio = document.getElementById('cms-ratio');
            if (cmsRatio) {
                cmsRatio.textContent = cms.policy_ratio ? 'Policy ' + cms.policy_ratio + ' | Maintenance ' + cms.maintenance_ratio : '--';
            }
            var cmsBuffer = document.getElementById('cms-buffer');
            if (cmsBuffer) {
                cmsBuffer.textContent = cms.buffer_fill ? 'Buffer fill: ' + Math.round(cms.buffer_fill * 100) + '%' : 'Buffer fill: --';
            }

            var heartbeat = loops.heartbeat || {};
            var heartbeatBadge = document.getElementById('heartbeat-badge');
            if (heartbeatBadge) {
                heartbeatBadge.textContent = heartbeat.ok ? 'Heartbeat stable' : 'Heartbeat delayed';
                heartbeatBadge.className = 'chip ' + (heartbeat.ok ? 'success' : 'danger');
            }

            var gateAllow = document.getElementById('gate-allow');
            if (gateAllow) { gateAllow.textContent = gates.allow_motion ? 'Open' : 'Locked'; }
            var gateRecord = document.getElementById('gate-record');
            if (gateRecord) { gateRecord.textContent = gates.recording_gate ? 'Armed' : 'Off'; }

            var safetyHead = document.getElementById('safety-head-path');
            if (safetyHead) { safetyHead.textContent = safety.head_path || 'stub'; }
            var safetyEnvelope = document.getElementById('safety-envelope');
            if (safetyEnvelope) {
                var env = safety.envelope || {};
                safetyEnvelope.textContent = (env.status || 'simulated') + ' • ' + (env.radius_m || '?') + 'm radius';
            }
            var safetyHeartbeat = document.getElementById('safety-heartbeat');
            if (safetyHeartbeat) {
                safetyHeartbeat.textContent = safety.heartbeat && safety.heartbeat.ok ? 'Online (' + safety.heartbeat.source + ')' : 'Simulated';
            }
        }
        
        window.updateStatus = function() {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/api/status', true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    try {
                        var data = JSON.parse(xhr.responseText);
                        if (data.status) {
                            var mode = data.status.mode || 'unknown';
                            var modeText = mode.replace(/_/g, ' ').toUpperCase();
                            document.getElementById('mode').innerHTML = '<span class="badge ' + mode + '">' + modeText + '</span>';
                            document.getElementById('recording').textContent = data.status.is_recording ? 'Recording' : 'Idle';
                            document.getElementById('motion').textContent = data.status.allow_motion ? 'Motion Enabled' : 'Motion Locked';

                            var modeCard = document.getElementById('mode-card');
                            if (modeCard) { modeCard.textContent = modeText; }
                            var recordingCard = document.getElementById('recording-card');
                            if (recordingCard) { recordingCard.textContent = data.status.is_recording ? 'Recording' : 'Idle'; }
                            var motionCard = document.getElementById('motion-card');
                            if (motionCard) { motionCard.textContent = data.status.allow_motion ? 'Allowed' : 'Prevented'; }

                            renderLoopTelemetry(data.status);

                            // Update hardware sensors
                            var hardwareDiv = document.getElementById('hardware-status');
                            if (data.status.detected_hardware) {
                                var hw = data.status.detected_hardware;
                                var hwHtml = '';
                                
                                if (hw.depth_camera) {
                                    hwHtml += '<div class="status-item"><span class="status-label">📷 Depth Camera</span><span class="status-value">' + hw.depth_camera + '</span></div>';
                                }
                                if (hw.depth_camera_driver) {
                                    hwHtml += '<div class="status-item"><span class="status-label">Camera Driver</span><span class="status-value">' + hw.depth_camera_driver + '</span></div>';
                                }
                                if (hw.servo_controller) {
                                    hwHtml += '<div class="status-item"><span class="status-label">🦾 Servo Controller</span><span class="status-value">' + hw.servo_controller + '</span></div>';
                                }
                                if (hw.servo_controller_address) {
                                    hwHtml += '<div class="status-item"><span class="status-label">I2C Address</span><span class="status-value">' + hw.servo_controller_address + '</span></div>';
                                }
                                
                                if (hwHtml) {
                                    hardwareDiv.innerHTML = hwHtml;
                                } else {
                                    hardwareDiv.innerHTML = '<div class="status-item"><span class="status-label">No hardware detected</span></div>';
                                }
                            } else {
                                hardwareDiv.innerHTML = '<div class="status-item"><span class="status-label">Hardware info not available</span></div>';
                            }
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                }
            };
            xhr.onerror = function() {
                console.error('Connection failed');
                window.showMessage('Failed to connect to robot', true);
            };
            xhr.send();
        };
        
        window.setMode = function(mode) {
            console.log('Setting mode to:', mode);
            window.showMessage('Changing mode to ' + mode.replace(/_/g, ' ') + '...');
            
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/api/mode/' + mode, true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    try {
                        var data = JSON.parse(xhr.responseText);
                        if (data.success) {
                            window.showMessage('Mode changed to ' + mode.replace(/_/g, ' ').toUpperCase());
                            setTimeout(window.updateStatus, 500);
                        } else {
                            window.showMessage('Failed: ' + (data.message || 'Unknown error'), true);
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                        window.showMessage('Error parsing response', true);
                    }
                } else {
                    window.showMessage('Server error: ' + xhr.status, true);
                }
            };
            xhr.onerror = function() {
                console.error('Connection failed');
                window.showMessage('Connection failed', true);
            };
            xhr.send();
        };
        
        // Update status every 2 seconds
        window.updateStatus();
        setInterval(window.updateStatus, 2000);
    </script>
</body>
</html>"""
    
    def get_control_interface_html(self):
        """Generate live control interface with camera feed and system status."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manual Control - CraigBot</title>
    <style>
        :root {
            --bg: #0b1020;
            --panel: #0f1729;
            --border: #1f2a3d;
            --muted: #8b95b5;
            --accent: #7ad7ff;
            --accent-strong: #4f9dff;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: radial-gradient(circle at 25% 20%, rgba(122, 215, 255, 0.08), transparent 30%),
                        radial-gradient(circle at 80% 0%, rgba(79, 157, 255, 0.08), transparent 25%),
                        var(--bg);
            color: #fff;
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, rgba(15, 23, 41, 0.95), rgba(16, 22, 38, 0.9));
            padding: 14px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            box-shadow: 0 10px 32px rgba(0, 0, 0, 0.4);
        }
        .header h1 {
            font-size: 18px;
            color: #fff;
            letter-spacing: 0.3px;
        }
        .back-btn {
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: border-color 0.15s ease, transform 0.1s ease;
        }
        .back-btn:hover { border-color: var(--accent); transform: translateY(-1px); }
        .main-container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            height: calc(100vh - 60px);
            gap: 12px;
            background: transparent;
            padding: 12px;
        }
        .video-panel {
            background: linear-gradient(180deg, rgba(10, 16, 32, 0.9), rgba(15, 23, 41, 0.94));
            border: 1px solid var(--border);
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
        }
        .video-feed {
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 50% 20%, rgba(74, 217, 255, 0.05), rgba(15, 23, 41, 0.9));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: #666;
            position: relative;
        }
        .video-feed img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .video-placeholder {
            text-align: center;
            position: absolute;
        }
        .video-info {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.55);
            padding: 12px;
            border-radius: 10px;
            font-size: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.45);
        }
        .status-panel {
            background: linear-gradient(180deg, rgba(14, 21, 35, 0.94), rgba(12, 18, 32, 0.94));
            padding: 20px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
        }
        .status-section {
            margin-bottom: 20px;
        }
        .status-section h3 {
            font-size: 14px;
            color: var(--muted);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-item {
            background: rgba(255, 255, 255, 0.02);
            padding: 14px;
            border-radius: 10px;
            margin-bottom: 8px;
            border: 1px solid var(--border);
        }
        .status-label {
            font-size: 11px;
            color: var(--muted);
            margin-bottom: 4px;
        }
        .status-value {
            font-size: 16px;
            font-weight: 600;
            color: #fff;
        }
        .status-good { color: #34c759; }
        .status-warning { color: #ff9500; }
        .status-critical { color: #ff3b30; }
        .joint-controls {
            display: grid;
            gap: 8px;
        }
        .joint-slider {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .joint-slider label {
            font-size: 11px;
            color: #86868b;
        }
        .joint-slider input {
            width: 100%;
        }
        .emergency-btn {
            width: 100%;
            background: #ff3b30;
            color: white;
            border: none;
            padding: 16px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
        }
        .arrow-controls {
            display: grid;
            gap: 12px;
            margin-top: 12px;
        }
        .arrow-group {
            background: #2a2a2c;
            padding: 12px;
            border-radius: 8px;
        }
        .arrow-group-title {
            font-size: 11px;
            color: #86868b;
            margin-bottom: 8px;
            text-align: center;
        }
        .arrow-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 4px;
            max-width: 200px;
            margin: 0 auto;
        }
        .arrow-btn {
            background: #007aff;
            color: white;
            border: none;
            padding: 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 18px;
            transition: background 0.1s;
            user-select: none;
        }
        .arrow-btn:active {
            background: #0051d5;
        }
        .arrow-btn:disabled {
            background: #333;
            cursor: not-allowed;
            opacity: 0.5;
        }
        .arrow-btn.center {
            background: #333;
        }
        .keyboard-hint {
            font-size: 10px;
            color: #555;
            text-align: center;
            margin-top: 4px;
        }
        .chat-overlay {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 400px;
            max-height: 600px;
            background: rgba(29, 29, 31, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid #333;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            z-index: 1000;
        }
        .chat-overlay.minimized {
            max-height: 50px;
        }
        .chat-header {
            padding: 12px 16px;
            background: #2a2a2c;
            border-radius: 12px 12px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            border-bottom: 1px solid #333;
        }
        .chat-header h3 {
            font-size: 14px;
            color: #fff;
            margin: 0;
        }
        .chat-toggle {
            background: none;
            border: none;
            color: #86868b;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            min-height: 200px;
            max-height: 400px;
        }
        .chat-message {
            margin-bottom: 12px;
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 13px;
            line-height: 1.5;
        }
        .chat-message.user {
            background: #007aff;
            color: white;
            margin-left: 40px;
        }
        .chat-message.assistant {
            background: #2a2a2c;
            color: #fff;
            margin-right: 40px;
        }
        .chat-message.system {
            background: #333;
            color: #86868b;
            font-size: 11px;
            text-align: center;
            margin: 8px 20px;
        }
        .chat-input-area {
            padding: 12px;
            border-top: 1px solid #333;
            display: flex;
            gap: 8px;
        }
        .chat-input {
            flex: 1;
            background: #2a2a2c;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 10px;
            color: #fff;
            font-size: 13px;
            outline: none;
        }
        .chat-input:focus {
            border-color: #007aff;
        }
        .chat-send-btn {
            background: #007aff;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
        }
        .chat-send-btn:disabled {
            background: #333;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎮 Manual Control - CraigBot</h1>
        <button class="back-btn" onclick="window.location.href='/ui'">← Back to Menu</button>
    </div>
    
    <div class="main-container">
        <div class="video-panel">
            <div class="video-info">
                <div>Mode: <strong id="current-mode">Manual Control</strong></div>
                <div>FPS: <span id="fps">0</span></div>
                <div>Latency: <span id="latency">0</span>ms</div>
            </div>
            <div class="video-feed" id="video-container">
                <img id="camera-stream" style="display:none;" alt="Camera Feed">
                <div class="video-placeholder" id="video-placeholder">
                    <div>📹</div>
                    <div>Camera Feed</div>
                    <div style="font-size: 14px; color: #444; margin-top: 10px;">
                        Connecting to OAK-D Lite...
                    </div>
                </div>
            </div>
        </div>
        
        <div class="status-panel">
            <div class="status-section">
                <h3>System Status</h3>
                <div class="status-item">
                    <div class="status-label">Hardware Mode</div>
                    <div class="status-value status-good" id="hardware-mode">REAL</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Robot Mode</div>
                    <div class="status-value status-good" id="robot-mode">MANUAL</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Motion Enabled</div>
                    <div class="status-value status-good" id="motion-enabled">Yes</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Recording</div>
                    <div class="status-value" id="recording-status">No</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Drivetrain</div>
                    <div class="status-value" id="drivetrain-connection">Checking...</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Last Drive</div>
                    <div class="status-value" id="drive-message">Awaiting command</div>
                </div>
            </div>
            
            <div class="status-section">
                <h3>Hardware Status</h3>
                <div id="hardware-details">
                    <div class="status-item">
                        <div class="status-label">Loading...</div>
                    </div>
                </div>
            </div>
            
            <div class="status-section">
                <h3>Joint Control</h3>
                <div class="joint-controls">
                    <div class="joint-slider">
                        <label>J0 (Base): <span id="j0-display">0.0</span></label>
                        <input type="range" id="j0" min="-100" max="100" value="0" oninput="updateJointDisplay(0, this.value); sendJointCommand()">
                    </div>
                    <div class="joint-slider">
                        <label>J1 (Shoulder): <span id="j1-display">0.0</span></label>
                        <input type="range" id="j1" min="-100" max="100" value="0" oninput="updateJointDisplay(1, this.value); sendJointCommand()">
                    </div>
                    <div class="joint-slider">
                        <label>J2 (Elbow): <span id="j2-display">0.0</span></label>
                        <input type="range" id="j2" min="-100" max="100" value="0" oninput="updateJointDisplay(2, this.value); sendJointCommand()">
                    </div>
                    <div class="joint-slider">
                        <label>J3 (Wrist Roll): <span id="j3-display">0.0</span></label>
                        <input type="range" id="j3" min="-100" max="100" value="0" oninput="updateJointDisplay(3, this.value); sendJointCommand()">
                    </div>
                    <div class="joint-slider">
                        <label>J4 (Wrist Pitch): <span id="j4-display">0.0</span></label>
                        <input type="range" id="j4" min="-100" max="100" value="0" oninput="updateJointDisplay(4, this.value); sendJointCommand()">
                    </div>
                    <div class="joint-slider">
                        <label>Gripper: <span id="j5-display">-1.0</span></label>
                        <input type="range" id="j5" min="-100" max="100" value="-100" oninput="updateJointDisplay(5, this.value); sendJointCommand()">
                    </div>
                </div>
            </div>
            
            <div class="status-section">
                <h3>Arrow Controls</h3>
                <div class="arrow-controls">
                    <div class="arrow-group">
                        <div class="arrow-group-title">Base Rotation (J0)</div>
                        <div class="arrow-grid">
                            <div></div>
                            <button class="arrow-btn" onmousedown="startMove(0, 0.1)" onmouseup="stopMove()" ontouchstart="startMove(0, 0.1)" ontouchend="stopMove()">⬆</button>
                            <div></div>
                            <button class="arrow-btn" onmousedown="startMove(0, -0.1)" onmouseup="stopMove()" ontouchstart="startMove(0, -0.1)" ontouchend="stopMove()">⬅</button>
                            <button class="arrow-btn center" disabled>J0</button>
                            <button class="arrow-btn" onmousedown="startMove(0, 0.1)" onmouseup="stopMove()" ontouchstart="startMove(0, 0.1)" ontouchend="stopMove()">➡</button>
                            <div></div>
                            <button class="arrow-btn" onmousedown="startMove(0, -0.1)" onmouseup="stopMove()" ontouchstart="startMove(0, -0.1)" ontouchend="stopMove()">⬇</button>
                            <div></div>
                        </div>
                    </div>
                    <div class="arrow-group">
                        <div class="arrow-group-title">Shoulder/Elbow (J1/J2)</div>
                        <div class="arrow-grid">
                            <button class="arrow-btn" onmousedown="startMove(1, 0.1)" onmouseup="stopMove()" ontouchstart="startMove(1, 0.1)" ontouchend="stopMove()">J1⬆</button>
                            <button class="arrow-btn" onmousedown="startMove(2, 0.1)" onmouseup="stopMove()" ontouchstart="startMove(2, 0.1)" ontouchend="stopMove()">J2⬆</button>
                            <div></div>
                            <button class="arrow-btn" onmousedown="startMove(1, -0.1)" onmouseup="stopMove()" ontouchstart="startMove(1, -0.1)" ontouchend="stopMove()">J1⬇</button>
                            <button class="arrow-btn" onmousedown="startMove(2, -0.1)" onmouseup="stopMove()" ontouchstart="startMove(2, -0.1)" ontouchend="stopMove()">J2⬇</button>
                            <div></div>
                        </div>
                    </div>
                    <div class="arrow-group">
                        <div class="arrow-group-title">Wrist (J3/J4)</div>
                        <div class="arrow-grid">
                            <button class="arrow-btn" onmousedown="startMove(3, 0.1)" onmouseup="stopMove()" ontouchstart="startMove(3, 0.1)" ontouchend="stopMove()">J3⬆</button>
                            <button class="arrow-btn" onmousedown="startMove(4, 0.1)" onmouseup="stopMove()" ontouchstart="startMove(4, 0.1)" ontouchend="stopMove()">J4⬆</button>
                            <div></div>
                            <button class="arrow-btn" onmousedown="startMove(3, -0.1)" onmouseup="stopMove()" ontouchstart="startMove(3, -0.1)" ontouchend="stopMove()">J3⬇</button>
                            <button class="arrow-btn" onmousedown="startMove(4, -0.1)" onmouseup="stopMove()" ontouchstart="startMove(4, -0.1)" ontouchend="stopMove()">J4⬇</button>
                            <div></div>
                        </div>
                        <div class="keyboard-hint">Use arrow keys + WASD for keyboard control</div>
                    </div>
                </div>
            </div>
            
            <div class="status-section">
                <h3>🏎️ Car Driving Controls</h3>
                <div class="arrow-controls">
                    <div class="arrow-group">
                        <div class="arrow-group-title">Steering & Throttle</div>
                        <div class="arrow-grid">
                            <div></div>
                            <button class="arrow-btn" onmousedown="startDrive('forward')" onmouseup="stopDrive()" ontouchstart="startDrive('forward')" ontouchend="stopDrive()">⬆️</button>
                            <div></div>
                            <button class="arrow-btn" onmousedown="startDrive('left')" onmouseup="stopDrive()" ontouchstart="startDrive('left')" ontouchend="stopDrive()">⬅️</button>
                            <button class="arrow-btn center" disabled>🏎️</button>
                            <button class="arrow-btn" onmousedown="startDrive('right')" onmouseup="stopDrive()" ontouchstart="startDrive('right')" ontouchend="stopDrive()">➡️</button>
                            <div></div>
                            <button class="arrow-btn" onmousedown="startDrive('backward')" onmouseup="stopDrive()" ontouchstart="startDrive('backward')" ontouchend="stopDrive()">⬇️</button>
                            <div></div>
                        </div>
                        <div class="keyboard-hint">Arrow keys or WASD to drive</div>
                    </div>
                    <div class="arrow-group">
                        <div class="arrow-group-title">Speed: <span id="speed-level">SLOW</span> (<span id="speed-value">0.3</span>)</div>
                        <div style="display: flex; gap: 4px;">
                            <button style="flex: 1; background: #34c759; color: white; border: none; padding: 8px; border-radius: 6px; cursor: pointer; font-size: 11px;" onclick="setSpeed(0.2, 'CRAWL')">🐌 Crawl</button>
                            <button style="flex: 1; background: #007aff; color: white; border: none; padding: 8px; border-radius: 6px; cursor: pointer; font-size: 11px;" onclick="setSpeed(0.3, 'SLOW')">🚪 Slow</button>
                            <button style="flex: 1; background: #ff9500; color: white; border: none; padding: 8px; border-radius: 6px; cursor: pointer; font-size: 11px;" onclick="setSpeed(0.5, 'MED')">🚶 Med</button>
                            <button style="flex: 1; background: #ff3b30; color: white; border: none; padding: 8px; border-radius: 6px; cursor: pointer; font-size: 11px;" onclick="setSpeed(0.7, 'FAST')">🏃 Fast</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="status-section">
                <h3>🦾 Arm Preset Positions</h3>
                <button style="width: 100%; margin-bottom: 8px; background: #007aff; color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer;" onclick="gotoHome()">🏠 Home Position</button>
                <button style="width: 100%; margin-bottom: 8px; background: #007aff; color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer;" onclick="gotoZero()">0️⃣ Zero Position</button>
                <button style="width: 100%; margin-bottom: 8px; background: #34c759; color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer;" onclick="openGripper()">✋ Open Gripper</button>
                <button style="width: 100%; margin-bottom: 8px; background: #ff9500; color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer;" onclick="closeGripper()">✊ Close Gripper</button>
            </div>
            
            <button class="emergency-btn" onclick="emergencyStop()">🛑 EMERGENCY STOP</button>
        </div>
    </div>
    
    <!-- Chat Interface -->
    <div class="chat-overlay" id="chat-panel">
        <div class="chat-header" onclick="toggleChat()">
            <h3>🤖 Gemma 3n Assistant</h3>
            <button class="chat-toggle" id="chat-toggle">−</button>
        </div>
        <div class="chat-messages" id="chat-messages">
            <div class="chat-message system">Chat with Gemma 3n about robot control</div>
        </div>
        <div class="chat-input-area">
            <input type="text" class="chat-input" id="chat-input" placeholder="Ask about robot status, control tips..." onkeypress="if(event.key==='Enter') sendChatMessage()">
            <button class="chat-send-btn" id="chat-send" onclick="sendChatMessage()">➤</button>
        </div>
    </div>
    
    <script type="text/javascript">
        var frameCount = 0;
        var lastFrameTime = Date.now();
        var cameraActive = false;
        
        window.updateCameraFrame = function() {
            var img = document.getElementById('camera-stream');
            var placeholder = document.getElementById('video-placeholder');
            var currentTime = Date.now();
            
            // Add timestamp to prevent caching
            img.src = '/api/camera/frame?t=' + currentTime;
            
            img.onload = function() {
                if (!cameraActive) {
                    // First successful frame - hide placeholder
                    placeholder.style.display = 'none';
                    img.style.display = 'block';
                    cameraActive = true;
                }
                
                // Calculate FPS
                frameCount++;
                var now = Date.now();
                var elapsed = now - lastFrameTime;
                if (elapsed >= 1000) {
                    var fps = Math.round(frameCount * 1000 / elapsed);
                    document.getElementById('fps').textContent = fps;
                    frameCount = 0;
                    lastFrameTime = now;
                }
                
                // Calculate latency
                var latency = Date.now() - currentTime;
                document.getElementById('latency').textContent = latency;
            };
            
            img.onerror = function() {
                if (cameraActive) {
                    // Lost camera - show placeholder
                    img.style.display = 'none';
                    placeholder.style.display = 'block';
                    placeholder.innerHTML = '<div>📹</div><div>Camera Disconnected</div><div style="font-size: 14px; color: #444; margin-top: 10px;">Reconnecting...</div>';
                    cameraActive = false;
                }
            };
        };
        
        var currentJointPositions = [0, 0, 0, 0, 0, -1];
        var moveInterval = null;
        var activeJoint = -1;
        var activeDelta = 0;
        
        // Car driving state
        var carSteering = 0.0;  // -1.0 (left) to 1.0 (right)
        var carThrottle = 0.0;  // -1.0 (reverse) to 1.0 (forward)
        var maxSpeed = 0.3;     // Default to slow (30% throttle)
        var driveInterval = null;
        var activeDriveDirection = null;
        
        window.updateJointDisplay = function(index, value) {
            var normalized = value / 100.0;
            currentJointPositions[index] = normalized;
            document.getElementById('j' + index + '-display').textContent = normalized.toFixed(2);
        };
        
        window.startMove = function(jointIndex, delta) {
            activeJoint = jointIndex;
            activeDelta = delta;
            
            // Immediate first move
            moveJoint(jointIndex, delta);
            
            // Continue moving while held
            moveInterval = setInterval(function() {
                moveJoint(jointIndex, delta);
            }, 50); // 20Hz updates
        };
        
        window.stopMove = function() {
            if (moveInterval) {
                clearInterval(moveInterval);
                moveInterval = null;
            }
            activeJoint = -1;
            activeDelta = 0;
        };
        
        window.moveJoint = function(jointIndex, delta) {
            var newValue = currentJointPositions[jointIndex] + delta;
            // Clamp to [-1, 1]
            newValue = Math.max(-1.0, Math.min(1.0, newValue));
            
            currentJointPositions[jointIndex] = newValue;
            document.getElementById('j' + jointIndex).value = newValue * 100;
            document.getElementById('j' + jointIndex + '-display').textContent = newValue.toFixed(2);
            
            sendJointCommand();
        };
        
        // Keyboard controls
        document.addEventListener('keydown', function(e) {
            // Prevent if input is focused
            if (document.activeElement.tagName === 'INPUT') return;
            
            var handled = false;
            
            // Check for modifier keys - hold Ctrl for car driving
            var isDriving = e.ctrlKey || e.metaKey;
            
            switch(e.key) {
                // Arrow keys - Car driving with Ctrl, or arm control without
                case 'ArrowUp':
                    if (isDriving) {
                        if (!activeDriveDirection) startDrive('forward');
                    }
                    handled = isDriving;
                    break;
                case 'ArrowDown':
                    if (isDriving) {
                        if (!activeDriveDirection) startDrive('backward');
                    }
                    handled = isDriving;
                    break;
                case 'ArrowLeft':
                    if (isDriving) {
                        if (!activeDriveDirection) startDrive('left');
                    } else {
                        if (activeJoint !== 0) startMove(0, -0.1);
                    }
                    handled = true;
                    break;
                case 'ArrowRight':
                    if (isDriving) {
                        if (!activeDriveDirection) startDrive('right');
                    } else {
                        if (activeJoint !== 0) startMove(0, 0.1);
                    }
                    handled = true;
                    break;
                // W/S - Shoulder (J1)
                case 'w':
                case 'W':
                    if (activeJoint !== 1) startMove(1, 0.1);
                    handled = true;
                    break;
                case 's':
                case 'S':
                    if (activeJoint !== 1) startMove(1, -0.1);
                    handled = true;
                    break;
                // A/D - Elbow (J2)
                case 'a':
                case 'A':
                    if (activeJoint !== 2) startMove(2, -0.1);
                    handled = true;
                    break;
                case 'd':
                case 'D':
                    if (activeJoint !== 2) startMove(2, 0.1);
                    handled = true;
                    break;
                // Q/E - Wrist Roll (J3)
                case 'q':
                case 'Q':
                    if (activeJoint !== 3) startMove(3, -0.1);
                    handled = true;
                    break;
                case 'e':
                case 'E':
                    if (activeJoint !== 3) startMove(3, 0.1);
                    handled = true;
                    break;
                // R/F - Wrist Pitch (J4)
                case 'r':
                case 'R':
                    if (activeJoint !== 4) startMove(4, 0.1);
                    handled = true;
                    break;
                case 'f':
                case 'F':
                    if (activeJoint !== 4) startMove(4, -0.1);
                    handled = true;
                    break;
                // Space/Shift - Gripper
                case ' ':
                    openGripper();
                    handled = true;
                    break;
                case 'Shift':
                    closeGripper();
                    handled = true;
                    break;
            }
            
            if (handled) {
                e.preventDefault();
            }
        });
        
        document.addEventListener('keyup', function(e) {
            // Stop car driving on any arrow key release
            if (activeDriveDirection && (e.key === 'ArrowUp' || e.key === 'ArrowDown' || e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
                stopDrive();
                e.preventDefault();
                return;
            }
            
            var shouldStop = false;
            switch(e.key) {
                case 'ArrowLeft':
                case 'ArrowRight':
                    if (activeJoint === 0) shouldStop = true;
                    break;
                case 'w':
                case 'W':
                case 's':
                case 'S':
                    if (activeJoint === 1) shouldStop = true;
                    break;
                case 'a':
                case 'A':
                case 'd':
                case 'D':
                    if (activeJoint === 2) shouldStop = true;
                    break;
                case 'q':
                case 'Q':
                case 'e':
                case 'E':
                    if (activeJoint === 3) shouldStop = true;
                    break;
                case 'r':
                case 'R':
                case 'f':
                case 'F':
                    if (activeJoint === 4) shouldStop = true;
                    break;
            }
            
            if (shouldStop) {
                stopMove();
                e.preventDefault();
            }
        });
        
        window.sendJointCommand = function() {
            // Throttle commands - only send if motion is enabled
            var motionEnabled = document.getElementById('motion-enabled').textContent === 'Yes';
            if (!motionEnabled) {
                console.warn('Motion not enabled in current mode');
                return;
            }
            
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/command', true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.send(JSON.stringify({
                client_id: 'web_control',
                control_mode: 'armJointAngles',
                arm_joint_angles: {
                    normalized_angles: currentJointPositions
                }
            }));
        };
        
        window.gotoHome = function() {
            setJointPositions([0, 0, 0, 0, 0, -1]);
        };
        
        window.gotoZero = function() {
            setJointPositions([0, 0, 0, 0, 0, 0]);
        };
        
        window.openGripper = function() {
            currentJointPositions[5] = -1.0;
            document.getElementById('j5').value = -100;
            document.getElementById('j5-display').textContent = '-1.00';
            sendJointCommand();
        };
        
        window.closeGripper = function() {
            currentJointPositions[5] = 1.0;
            document.getElementById('j5').value = 100;
            document.getElementById('j5-display').textContent = '1.00';
            sendJointCommand();
        };
        
        window.setJointPositions = function(positions) {
            for (var i = 0; i < 6; i++) {
                currentJointPositions[i] = positions[i];
                document.getElementById('j' + i).value = positions[i] * 100;
                document.getElementById('j' + i + '-display').textContent = positions[i].toFixed(2);
            }
            sendJointCommand();
        };
        
        // Car driving controls
        window.setSpeed = function(speed, label) {
            maxSpeed = speed;
            document.getElementById('speed-value').textContent = speed.toFixed(1);
            document.getElementById('speed-level').textContent = label;
        };
        
        window.startDrive = function(direction) {
            activeDriveDirection = direction;
            
            // Set initial drive values
            updateDriveValues(direction);
            sendDriveCommand();
            
            // Continue sending while held
            driveInterval = setInterval(function() {
                updateDriveValues(direction);
                sendDriveCommand();
            }, 50); // 20Hz updates
        };
        
        window.stopDrive = function() {
            if (driveInterval) {
                clearInterval(driveInterval);
                driveInterval = null;
            }
            activeDriveDirection = null;
            
            // Stop the car
            carSteering = 0.0;
            carThrottle = 0.0;
            sendDriveCommand();
        };
        
        window.updateDriveValues = function(direction) {
            switch(direction) {
                case 'forward':
                    carThrottle = maxSpeed;
                    carSteering = 0.0;
                    break;
                case 'backward':
                    carThrottle = -maxSpeed;
                    carSteering = 0.0;
                    break;
                case 'left':
                    carSteering = -1.0;
                    if (carThrottle === 0) carThrottle = maxSpeed * 0.5; // Gentle forward when steering
                    break;
                case 'right':
                    carSteering = 1.0;
                    if (carThrottle === 0) carThrottle = maxSpeed * 0.5;
                    break;
            }
        };
        
        window.sendDriveCommand = function() {
            var motionEnabled = document.getElementById('motion-enabled').textContent === 'Yes';
            if (!motionEnabled) {
                console.warn('Motion not enabled in current mode');
                renderDriveResult({ success: false, message: 'Motion disabled in current mode' });
                return;
            }
            
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/drive', true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.onload = function() {
                if (xhr.status === 200) {
                    try {
                        var result = JSON.parse(xhr.responseText);
                        renderDriveResult(result);
                    } catch (e) {
                        console.error('Failed to parse drive response', e);
                    }
                }
            };
            xhr.onerror = function() {
                renderDriveResult({ success: false, message: 'Drive command failed to reach server' });
            };
            xhr.send(JSON.stringify({
                steering: carSteering,
                throttle: carThrottle
            }));
        };

        window.renderDriveResult = function(result) {
            if (!result) return;
            var text = (result.success ? '✅ ' : '⚠️ ') + (result.message || 'Drive command sent');
            if (typeof result.steering === 'number' && typeof result.throttle === 'number') {
                text += ' (S:' + parseFloat(result.steering).toFixed(2) + ', T:' + parseFloat(result.throttle).toFixed(2) + ')';
            }

            var target = document.getElementById('drive-message');
            target.textContent = text;
            target.className = 'status-value ' + (result.success ? 'status-good' : 'status-warning');

            // Keep connection status in sync if mode returned
            if (result.mode) {
                var connectionText = (result.success ? 'Connected' : 'Not Connected') + ' (' + result.mode.toUpperCase() + ')';
                var connectionClass = result.success ? 'status-good' : 'status-critical';
                document.getElementById('drivetrain-connection').textContent = connectionText;
                document.getElementById('drivetrain-connection').className = 'status-value ' + connectionClass;
            }
        };
        
        window.updateControlStatus = function() {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/api/status', true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    try {
                        var data = JSON.parse(xhr.responseText);
                        if (data.status) {
                            // Update mode and status
                            document.getElementById('hardware-mode').textContent = data.status.hardware_mode.toUpperCase();
                            document.getElementById('hardware-mode').className = 'status-value ' + (data.status.hardware_mode === 'real' ? 'status-good' : 'status-warning');
                            
                            document.getElementById('robot-mode').textContent = data.status.mode.replace(/_/g, ' ').toUpperCase();
                            document.getElementById('current-mode').textContent = data.status.mode.replace(/_/g, ' ').toUpperCase();
                            
                            var motionAllowed = data.status.allow_motion;
                            document.getElementById('motion-enabled').textContent = motionAllowed ? 'Yes' : 'No';
                            document.getElementById('motion-enabled').className = 'status-value ' + (motionAllowed ? 'status-good' : 'status-critical');
                            
                            document.getElementById('recording-status').textContent = data.status.is_recording ? 'Yes' : 'No';
                            document.getElementById('recording-status').className = 'status-value ' + (data.status.is_recording ? 'status-good' : '');

                            var drivetrain = data.status.drivetrain || null;
                            if (drivetrain) {
                                var connectionClass = drivetrain.connected ? 'status-good' : 'status-critical';
                                var connectionText = drivetrain.connected ? 'Connected' : 'Not Connected';
                                if (drivetrain.mode) {
                                    connectionText += ' (' + drivetrain.mode.toUpperCase() + ')';
                                }
                                document.getElementById('drivetrain-connection').textContent = connectionText;
                                document.getElementById('drivetrain-connection').className = 'status-value ' + connectionClass;

                                if (drivetrain.last_command) {
                                    var lastCmd = drivetrain.last_command;
                                    var driveText = (lastCmd.success ? '✅ ' : '⚠️ ') + (lastCmd.message || 'Drive command sent');
                                    if (typeof lastCmd.steering === 'number' && typeof lastCmd.throttle === 'number') {
                                        driveText += ' (S:' + lastCmd.steering.toFixed(2) + ', T:' + lastCmd.throttle.toFixed(2) + ')';
                                    }
                                    document.getElementById('drive-message').textContent = driveText;
                                    document.getElementById('drive-message').className = 'status-value ' + (lastCmd.success ? 'status-good' : 'status-warning');
                                }
                            }

                            // Update hardware details
                            if (data.status.detected_hardware) {
                                var hw = data.status.detected_hardware;
                                var hwHtml = '';
                                
                                if (hw.depth_camera) {
                                    hwHtml += '<div class="status-item"><div class="status-label">📷 Camera</div><div class="status-value status-good">' + hw.depth_camera + '</div></div>';
                                }
                                if (hw.depth_camera_driver) {
                                    hwHtml += '<div class="status-item"><div class="status-label">Driver</div><div class="status-value">' + hw.depth_camera_driver + '</div></div>';
                                }
                                if (hw.servo_controller) {
                                    hwHtml += '<div class="status-item"><div class="status-label">🦾 Servo</div><div class="status-value status-good">' + hw.servo_controller + '</div></div>';
                                }
                                if (hw.servo_controller_address) {
                                    hwHtml += '<div class="status-item"><div class="status-label">I2C Address</div><div class="status-value">' + hw.servo_controller_address + '</div></div>';
                                }
                                
                                if (hwHtml) {
                                    document.getElementById('hardware-details').innerHTML = hwHtml;
                                }
                            }
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                }
            };
            xhr.send();
        };
        
        window.emergencyStop = function() {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/api/mode/emergency_stop', true);
            xhr.onload = function() {
                alert('EMERGENCY STOP ACTIVATED');
                window.location.href = '/ui';
            };
            xhr.send();
        };
        
        // Chat functionality
        var chatMinimized = false;
        var chatHistory = [];
        
        window.toggleChat = function() {
            chatMinimized = !chatMinimized;
            var panel = document.getElementById('chat-panel');
            var toggle = document.getElementById('chat-toggle');
            
            if (chatMinimized) {
                panel.classList.add('minimized');
                toggle.textContent = '+';
            } else {
                panel.classList.remove('minimized');
                toggle.textContent = '−';
            }
        };
        
        window.addChatMessage = function(text, role) {
            var messagesDiv = document.getElementById('chat-messages');
            var messageDiv = document.createElement('div');
            messageDiv.className = 'chat-message ' + role;
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            chatHistory.push({role: role, content: text});
        };
        
        window.sendChatMessage = function() {
            var input = document.getElementById('chat-input');
            var sendBtn = document.getElementById('chat-send');
            var message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addChatMessage(message, 'user');
            input.value = '';
            
            // Disable input while processing
            input.disabled = true;
            sendBtn.disabled = true;
            
            // Send to Gemma endpoint
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/chat', true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.onload = function() {
                input.disabled = false;
                sendBtn.disabled = false;
                
                if (xhr.status === 200) {
                    try {
                        var data = JSON.parse(xhr.responseText);
                        if (data.response) {
                            addChatMessage(data.response, 'assistant');
                        } else if (data.error) {
                            addChatMessage('Error: ' + data.error, 'system');
                        }
                    } catch (e) {
                        addChatMessage('Error parsing response', 'system');
                    }
                } else {
                    addChatMessage('Server error: ' + xhr.status, 'system');
                }
                
                input.focus();
            };
            xhr.onerror = function() {
                input.disabled = false;
                sendBtn.disabled = false;
                addChatMessage('Connection error', 'system');
                input.focus();
            };
            
            // Include chat history for context
            xhr.send(JSON.stringify({
                message: message,
                history: chatHistory.slice(-10) // Last 10 messages for context
            }));
        };
        
        // Update every 100ms for responsive control
        window.updateControlStatus();
        setInterval(window.updateControlStatus, 100);
        
        // Update camera feed at ~30 FPS
        window.updateCameraFrame();
        setInterval(window.updateCameraFrame, 33);
    </script>
</body>
</html>"""
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single client connection."""
        addr = writer.get_extra_info('peername')
        print(f"Client connected: {addr}")
        
        try:
            # Read first line to detect HTTP vs JSON
            first_line = await reader.readline()
            if not first_line:
                return
            
            first_line_str = first_line.decode().strip()
            
            # Check if it's an HTTP request
            if first_line_str.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ', 'OPTIONS ')):
                await self.handle_http_request(first_line_str, reader, writer)
                return
            
            # Otherwise handle as JSON command (legacy)
            data = first_line
            
            while True:
                if not data:
                    break
                
                try:
                    command = json.loads(data.decode().strip())
                    method = command.get("method")
                    
                    # Route to service method
                    if method == "send_command":
                        response = await self.service.SendCommand(command.get("params", {}))
                    elif method == "set_mode":
                        response = await self.service.SetRobotMode(
                            command.get("params", {}).get("mode", "idle")
                        )
                    elif method == "get_status":
                        response = await self.service.GetRobotStatus()
                    elif method == "start_recording":
                        response = await self.service.StartEpisodeRecording(
                            command.get("params", {}).get("instruction", "")
                        )
                    elif method == "stop_recording":
                        response = await self.service.StopEpisodeRecording(
                            command.get("params", {}).get("success", True)
                        )
                    elif method == "get_depth":
                        response = await self.service.GetDepthFrame()
                    elif method == "stream_state":
                        # Stream multiple states
                        async for state in self.service.StreamRobotState(
                            command.get("params", {}).get("client_id", "json_client")
                        ):
                            response_json = json.dumps(state) + "\n"
                            writer.write(response_json.encode())
                            await writer.drain()
                            
                            # Check for client disconnect
                            if reader.at_eof():
                                break
                        continue
                    else:
                        response = {"success": False, "message": f"Unknown method: {method}"}
                    
                    # Send response
                    response_json = json.dumps(response) + "\n"
                    writer.write(response_json.encode())
                    await writer.drain()
                    
                except json.JSONDecodeError as e:
                    error_response = json.dumps({
                        "success": False,
                        "message": f"Invalid JSON: {e}"
                    }) + "\n"
                    writer.write(error_response.encode())
                    await writer.drain()
                
                # Read next command
                data = await reader.readline()
        
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        
        finally:
            print(f"Client disconnected: {addr}")
            writer.close()
            await writer.wait_closed()
    
    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the server."""
        self.server = await asyncio.start_server(
            self.handle_client, host, port
        )
        
        addr = self.server.sockets[0].getsockname()
        print(f"\n{'='*60}")
        print(f"🚀 ContinuonBrain Robot API listening on {addr[0]}:{addr[1]}")
        print(f"{'='*60}\n")
        print("📱 Web UI: http://{0}:{1}/ui".format(addr[0] if addr[0] != '0.0.0.0' else 'localhost', addr[1]))
        print("🔌 API Endpoint: http://{0}:{1}/status".format(addr[0] if addr[0] != '0.0.0.0' else 'localhost', addr[1]))
        print()
        print("Example JSON commands (via netcat):")
        print(f'  # Control arm')
        print(f'  echo \'{{"method": "send_command", "params": {{"client_id": "test", "control_mode": "armJointAngles", "arm_joint_angles": {{"normalized_angles": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]}}}}}}\' | nc {addr[0]} {addr[1]}')
        print(f'  # Change mode to manual training')
        print(f'  echo \'{{"method": "set_mode", "params": {{"mode": "manual_training"}}}}\' | nc {addr[0]} {addr[1]}')
        print(f'  # Get robot status')
        print(f'  echo \'{{"method": "get_status", "params": {{}}}}\' | nc {addr[0]} {addr[1]}')
        print()
        
        async with self.server:
            await self.server.serve_forever()


async def main():
    """Run the mock service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ContinuonBrain Robot API Server (Production)")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="/tmp/continuonbrain_demo",
        help="Configuration directory (default: /tmp/continuonbrain_demo)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 for all interfaces)"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--real-hardware",
        action="store_true",
        help="Force real hardware mode (fail if controllers are missing)"
    )
    mode_group.add_argument(
        "--mock-hardware",
        action="store_true",
        help="Force mock mode (skip hardware initialization)"
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable hardware auto-detection"
    )
    
    args = parser.parse_args()
    prefer_real = not args.mock_hardware
    allow_mock_fallback = not args.real_hardware
    auto_detect = not args.no_auto_detect
    
    # Create service in PRODUCTION mode
    service = RobotService(
        config_dir=args.config_dir,
        prefer_real_hardware=prefer_real,
        auto_detect=auto_detect,
        allow_mock_fallback=allow_mock_fallback,
    )
    await service.initialize()
    
    # Create simple JSON server
    server = SimpleJSONServer(service)
    
    try:
        await server.start(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
