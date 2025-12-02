"""
ContinuonBrain Robot API server for Pi5 robot arm.
Production server with real hardware control.
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


class RobotService:
    """
    Production Robot API server for arm control.
    Uses real hardware detected on the system.
    """
    
    def __init__(self, config_dir: str = "/tmp/continuonbrain_demo"):
        self.config_dir = config_dir
        self.arm: Optional[PCA9685ArmController] = None
        self.camera: Optional[OAKDepthCapture] = None
        self.recorder: Optional[ArmEpisodeRecorder] = None
        self.mode_manager: Optional[RobotModeManager] = None
        self.is_recording = False
        self.current_episode_id: Optional[str] = None
        self.detected_config: dict = {}
        
    async def initialize(self):
        """Initialize hardware components with auto-detection."""
        print(f"Initializing Robot Service (PRODUCTION MODE)...")
        print()
        
        # Auto-detect hardware
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
        
        # Initialize arm controller with REAL hardware
        print("🦾 Initializing arm controller...")
        self.arm = PCA9685ArmController(ArmConfig())
        if self.arm.initialize():
            print("✅ Arm controller initialized (REAL HARDWARE)")
        else:
            print("❌ Arm initialization failed - check I2C connections")
            raise RuntimeError("Failed to initialize arm controller")
        
        # Initialize camera with REAL hardware
        print("📷 Initializing depth camera...")
        camera_driver = self.detected_config.get("primary", {}).get("depth_camera_driver")
        if camera_driver == "depthai" or camera_driver is None:
            self.camera = OAKDepthCapture(CameraConfig())
            if self.camera.initialize():
                self.camera.start()
                print("✅ OAK-D camera initialized (REAL HARDWARE)")
            else:
                print("⚠️  Camera initialization failed - continuing without camera")
                self.camera = None
        else:
            print(f"⚠️  Unsupported camera driver: {camera_driver}")
            self.camera = None
        
        # Initialize recorder
        print("📼 Initializing episode recorder...")
        self.recorder = ArmEpisodeRecorder(
            episodes_dir=f"{self.config_dir}/episodes",
            max_steps=500,
        )
        self.recorder.arm = self.arm
        self.recorder.camera = self.camera
        print("✅ Episode recorder ready")
        
        # Initialize mode manager
        print("🎮 Initializing mode manager...")
        self.mode_manager = RobotModeManager(config_dir=self.config_dir)
        self.mode_manager.return_to_idle()  # Start in idle mode
        print("✅ Mode manager ready")
        
        print()
        print("=" * 60)
        print("✅ Robot Service Ready (PRODUCTION MODE)")
        print("=" * 60)
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
        path = parts[1] if len(parts) > 1 else "/"
        
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
            response = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/status":
            status = await self.service.GetRobotStatus()
            response_body = json.dumps(status, indent=2)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/status":
            status = await self.service.GetRobotStatus()
            response_body = json.dumps(status)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path.startswith("/api/mode/"):
            mode = path.split("/")[-1]
            print(f"[MODE] Changing to: {mode}")
            result = await self.service.SetRobotMode(mode)
            print(f"[MODE] Result: {result}")
            response_body = json.dumps(result)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        else:
            response_body = "404 Not Found"
            response = f"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        
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
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f7;
        }
        .container {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1d1d1f;
            margin-top: 0;
            font-size: 28px;
        }
        .status {
            background: #f5f5f7;
            padding: 16px;
            border-radius: 8px;
            margin: 16px 0;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 14px;
        }
        .status-label {
            color: #86868b;
        }
        .status-value {
            font-weight: 600;
            color: #1d1d1f;
        }
        .mode-buttons {
            display: grid;
            gap: 12px;
            margin: 20px 0;
        }
        button {
            background: #007aff;
            color: white;
            border: none;
            padding: 14px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:active {
            background: #0051d5;
        }
        button.secondary {
            background: #86868b;
        }
        button.danger {
            background: #ff3b30;
        }
        .recording {
            background: #34c759;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            background: #34c759;
            color: white;
        }
        .badge.idle { background: #86868b; }
        .badge.training { background: #007aff; }
        .badge.autonomous { background: #af52de; }
        .badge.sleeping { background: #ff9500; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 CraigBot</h1>
        
        <div class="status">
            <div class="status-item">
                <span class="status-label">Mode</span>
                <span class="status-value" id="mode">Loading...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Recording</span>
                <span class="status-value" id="recording">No</span>
            </div>
            <div class="status-item">
                <span class="status-label">Motion Allowed</span>
                <span class="status-value" id="motion">No</span>
            </div>
        </div>
        
        <div class="mode-buttons">
            <button onclick="setMode('manual_training')">🎮 Manual Control</button>
            <button onclick="setMode('autonomous')" class="secondary">🚀 Autonomous</button>
            <button onclick="setMode('sleep_learning')" class="secondary">💤 Sleep Learning</button>
            <button onclick="setMode('idle')" class="secondary">⏸️ Idle</button>
            <button onclick="setMode('emergency_stop')" class="danger">🛑 Emergency Stop</button>
        </div>
        
        <div id="status-message" style="margin-top: 16px; padding: 12px; border-radius: 8px; display: none;"></div>
        
        <p style="text-align: center; color: #86868b; font-size: 12px; margin-top: 20px;">
            ContinuonXR Robot Control Interface
        </p>
    </div>
    
    <script>
        function showMessage(message, isError) {
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
        }
        
        function updateStatus() {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/api/status', true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    try {
                        var data = JSON.parse(xhr.responseText);
                        if (data.status) {
                            var mode = data.status.mode || 'unknown';
                            document.getElementById('mode').innerHTML = '<span class="badge ' + mode + '">' + mode.replace(/_/g, ' ').toUpperCase() + '</span>';
                            document.getElementById('recording').textContent = data.status.is_recording ? 'Yes' : 'No';
                            document.getElementById('motion').textContent = data.status.allow_motion ? 'Yes' : 'No';
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                }
            };
            xhr.onerror = function() {
                console.error('Connection failed');
                showMessage('Failed to connect to robot', true);
            };
            xhr.send();
        }
        
        function setMode(mode) {
            console.log('Setting mode to:', mode);
            showMessage('Changing mode to ' + mode.replace(/_/g, ' ') + '...');
            
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/api/mode/' + mode, true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    try {
                        var data = JSON.parse(xhr.responseText);
                        if (data.success) {
                            showMessage('Mode changed to ' + mode.replace(/_/g, ' ').toUpperCase());
                            setTimeout(updateStatus, 500);
                        } else {
                            showMessage('Failed: ' + (data.message || 'Unknown error'), true);
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                        showMessage('Error parsing response', true);
                    }
                } else {
                    showMessage('Server error: ' + xhr.status, true);
                }
            };
            xhr.onerror = function() {
                console.error('Connection failed');
                showMessage('Connection failed', true);
            };
            xhr.send();
        }
        
        // Update status every 2 seconds
        updateStatus();
        setInterval(updateStatus, 2000);
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
    
    args = parser.parse_args()
    
    # Create service in PRODUCTION mode
    service = RobotService(config_dir=args.config_dir)
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
