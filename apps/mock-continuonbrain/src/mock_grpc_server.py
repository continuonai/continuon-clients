"""
Mock ContinuonBrain gRPC server for Pi5 robot arm development.
Simulates Robot API endpoints for Flutter companion testing.
"""
import asyncio
import time
from typing import AsyncIterator, Optional
import json
import sys
from pathlib import Path

# Add proto generated code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "proto" / "generated"))

# Import arm components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from continuonbrain.actuators.pca9685_arm import PCA9685ArmController, ArmConfig
from continuonbrain.sensors.oak_depth import OAKDepthCapture, CameraConfig
from continuonbrain.recording.arm_episode_recorder import ArmEpisodeRecorder
from continuonbrain.sensors.hardware_detector import HardwareDetector
from continuonbrain.robot_modes import RobotModeManager, RobotMode


class MockRobotService:
    """
    Mock implementation of Robot API for arm control.
    Integrates with real/mock arm controller and depth camera.
    """
    
    def __init__(self, use_real_hardware: bool = False, auto_detect: bool = True):
        self.use_real_hardware = use_real_hardware
        self.auto_detect = auto_detect
        self.arm: Optional[PCA9685ArmController] = None
        self.camera: Optional[OAKDepthCapture] = None
        self.recorder: Optional[ArmEpisodeRecorder] = None
        self.mode_manager: Optional[RobotModeManager] = None
        self.is_recording = False
        self.current_episode_id: Optional[str] = None
        self.detected_config: dict = {}
        
    async def initialize(self):
        """Initialize hardware components with auto-detection."""
        print(f"Initializing Mock Robot Service...")
        print(f"  Real hardware: {self.use_real_hardware}")
        print(f"  Auto-detect: {self.auto_detect}")
        print()
        
        # Auto-detect hardware if enabled
        if self.auto_detect and not self.use_real_hardware:
            # In mock mode, just report what would be detected
            print("🔍 Hardware detection (mock mode - not initializing)")
            detector = HardwareDetector()
            devices = detector.detect_all()
            if devices:
                self.detected_config = detector.generate_config()
                detector.print_summary()
        elif self.auto_detect and self.use_real_hardware:
            print("🔍 Auto-detecting hardware...")
            detector = HardwareDetector()
            devices = detector.detect_all()
            if devices:
                self.detected_config = detector.generate_config()
                print()
        
        # Initialize arm controller
        self.arm = PCA9685ArmController(ArmConfig())
        if not self.arm.initialize():
            print("Warning: Arm initialization failed, using mock")
        
        # Initialize camera (only in real hardware mode)
        if self.use_real_hardware:
            camera_driver = self.detected_config.get("primary", {}).get("depth_camera_driver")
            if camera_driver == "depthai" or camera_driver is None:
                self.camera = OAKDepthCapture(CameraConfig())
                if self.camera.initialize():
                    self.camera.start()
                    print("✅ OAK-D camera initialized")
                else:
                    print("Warning: Camera initialization failed")
                    self.camera = None
        
        # Initialize recorder
        self.recorder = ArmEpisodeRecorder(
            episodes_dir="/tmp/flutter_episodes",
            max_steps=500,
        )
        self.recorder.arm = self.arm
        self.recorder.camera = self.camera
        
        # Initialize mode manager
        self.mode_manager = RobotModeManager(config_dir="/tmp/robot_mode")
        self.mode_manager.return_to_idle()  # Start in idle mode
        
        print("✅ Mock Robot Service ready")
        if self.detected_config.get("primary"):
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
    
    async def SetRobotMode(self, mode: str) -> dict:
        """Change robot operational mode."""
        try:
            if not self.mode_manager:
                return {"success": False, "message": "Mode manager not initialized"}
            
            # Map string to enum
            mode_map = {
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
            }
            
            if self.mode_manager:
                mode_status = self.mode_manager.get_status()
                status.update({
                    "mode": mode_status["mode"],
                    "mode_duration": mode_status["duration_seconds"],
                    "allow_motion": mode_status["config"]["allow_motion"],
                    "recording_enabled": mode_status["config"]["record_episodes"],
                })
            
            if self.arm:
                status["joint_positions"] = self.arm.get_normalized_state()
            
            return {"success": True, "status": status}
        
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
    
    def shutdown(self):
        """Graceful shutdown."""
        print("Shutting down Mock Robot Service...")
        
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
    Simple HTTP-like server for testing without full gRPC setup.
    Listens on localhost:8080 and accepts JSON commands.
    """
    
    def __init__(self, service: MockRobotService):
        self.service = service
        self.server = None
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single client connection."""
        addr = writer.get_extra_info('peername')
        print(f"Client connected: {addr}")
        
        try:
            while True:
                # Read JSON command
                data = await reader.readline()
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
        print(f"Mock ContinuonBrain Service listening on {addr[0]}:{addr[1]}")
        print(f"{'='*60}\n")
        print("Example commands:")
        print(f'  # Control arm')
        print(f'  echo \'{{"method": "send_command", "params": {{"client_id": "test", "control_mode": "armJointAngles", "arm_joint_angles": {{"normalized_angles": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]}}}}}}\' | nc {addr[0]} {addr[1]}')
        print(f'  # Change mode to manual training')
        print(f'  echo \'{{"method": "set_mode", "params": {{"mode": "manual_training"}}}}\' | nc {addr[0]} {addr[1]}')
        print(f'  # Get robot status')
        print(f'  echo \'{{"method": "get_status", "params": {{}}}}\' | nc {addr[0]} {addr[1]}')
        print(f'  # Start recording')
        print(f'  echo \'{{"method": "start_recording", "params": {{"instruction": "Pick cube"}}}}\' | nc {addr[0]} {addr[1]}')
        print()
        
        async with self.server:
            await self.server.serve_forever()


async def main():
    """Run the mock service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock ContinuonBrain Robot API server")
    parser.add_argument(
        "--real-hardware",
        action="store_true",
        help="Use real OAK-D and PCA9685 hardware"
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable hardware auto-detection"
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
    
    args = parser.parse_args()
    
    # Create service
    service = MockRobotService(
        use_real_hardware=args.real_hardware,
        auto_detect=not args.no_auto_detect
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
