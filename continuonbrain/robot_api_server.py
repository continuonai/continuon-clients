"""
ContinuonBrain Robot API server for Pi5 robot arm.
Runs against real hardware by default with optional mock fallback for dev.
"""
import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs
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


@dataclass
class TaskDefinition:
    """Static task library entry used to seed Studio panels."""

    id: str
    title: str
    description: str
    group: str
    tags: List[str] = field(default_factory=list)
    requires_motion: bool = False
    requires_recording: bool = False
    required_modalities: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    estimated_duration: str = ""
    recommended_mode: str = "autonomous"
    telemetry_topic: str = "loop/tasks"


@dataclass
class TaskEligibilityMarker:
    code: str
    label: str
    severity: str = "info"
    blocking: bool = False
    source: str = "runtime"
    remediation: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "label": self.label,
            "severity": self.severity,
            "blocking": self.blocking,
            "source": self.source,
            "remediation": self.remediation,
        }


@dataclass
class TaskEligibility:
    eligible: bool
    markers: List[TaskEligibilityMarker] = field(default_factory=list)
    next_poll_after_ms: float = 250.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "eligible": self.eligible,
            "markers": [marker.to_dict() for marker in self.markers],
            "next_poll_after_ms": self.next_poll_after_ms,
        }


@dataclass
class TaskLibraryEntry:
    id: str
    title: str
    description: str
    group: str
    tags: List[str]
    eligibility: TaskEligibility
    estimated_duration: str = ""
    recommended_mode: str = "autonomous"

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "group": self.group,
            "tags": self.tags,
            "eligibility": self.eligibility.to_dict(),
            "estimated_duration": self.estimated_duration,
            "recommended_mode": self.recommended_mode,
        }


@dataclass
class TaskSummary:
    entry: TaskLibraryEntry
    required_modalities: List[str]
    steps: List[str]
    owner: str
    updated_at: str
    telemetry_topic: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry": self.entry.to_dict(),
            "required_modalities": self.required_modalities,
            "steps": self.steps,
            "owner": self.owner,
            "updated_at": self.updated_at,
            "telemetry_topic": self.telemetry_topic,
        }


class TaskLibrary:
    """In-memory Task Library with lightweight eligibility markers."""

    def __init__(self) -> None:
        self._entries: Dict[str, TaskDefinition] = {
            "workspace-inspection": TaskDefinition(
                id="workspace-inspection",
                title="Inspect Workspace",
                description="Sweep sensors for loose cables or obstacles before enabling autonomy.",
                group="Safety",
                tags=["safety", "vision", "preflight"],
                requires_motion=False,
                requires_recording=False,
                required_modalities=["vision"],
                steps=[
                    "Spin the camera and depth sensors across the workspace",
                    "Flag blocked envelopes for operator acknowledgement",
                    "Cache a short clip for offline review",
                ],
                estimated_duration="45s",
                recommended_mode="autonomous",
                telemetry_topic="telemetry/preflight",
            ),
            "pick-and-place": TaskDefinition(
                id="pick-and-place",
                title="Pick & Place Demo",
                description="Run the default manipulation loop for demos and regressions.",
                group="Autonomy",
                tags=["manipulation", "demo", "recordable"],
                requires_motion=True,
                requires_recording=True,
                required_modalities=["vision", "gripper"],
                steps=[
                    "Plan grasp pose",
                    "Lift and place to bin",
                    "Report safety margin and latency",
                ],
                estimated_duration="2m",
                recommended_mode="autonomous",
                telemetry_topic="telemetry/manipulation",
            ),
            "calibration-check": TaskDefinition(
                id="calibration-check",
                title="Calibration Check",
                description="Verify encoders and camera alignment before overnight runs.",
                group="Maintenance",
                tags=["maintenance", "calibration"],
                requires_motion=False,
                requires_recording=False,
                required_modalities=["vision", "arm"],
                steps=[
                    "Move to calibration pose",
                    "Capture depth + RGB alignment snapshot",
                    "Emit eligibility marker if drift detected",
                ],
                estimated_duration="1m",
                recommended_mode="manual_training",
                telemetry_topic="telemetry/calibration",
            ),
        }

    def list_entries(self) -> List[TaskDefinition]:
        return list(self._entries.values())

    def get_entry(self, task_id: str) -> Optional[TaskDefinition]:
        return self._entries.get(task_id)


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
        self.task_library = TaskLibrary()
        self.selected_task_id: Optional[str] = None

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

    def _ensure_mode_manager(self) -> RobotModeManager:
        """Guarantee a mode manager exists for telemetry and gates."""

        if self.mode_manager is None:
            self.mode_manager = RobotModeManager(
                config_dir=self.config_dir,
                system_instructions=self.system_instructions,
            )
            self.mode_manager.return_to_idle()
        return self.mode_manager

    def _now_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _build_task_eligibility(self, task: TaskDefinition) -> TaskEligibility:
        mode_manager = self._ensure_mode_manager()
        gates = mode_manager.get_gate_snapshot() if mode_manager else {}
        markers: List[TaskEligibilityMarker] = []

        if task.requires_motion and not gates.get("allow_motion"):
            markers.append(
                TaskEligibilityMarker(
                    code="MOTION_GATE",
                    label="Motion is gated; unlock to run autonomy",
                    severity="blocking",
                    blocking=True,
                    remediation="Enable motion gate from the command deck",
                )
            )

        if task.requires_recording and not gates.get("record_episodes"):
            markers.append(
                TaskEligibilityMarker(
                    code="RECORDING_GATE",
                    label="Recording disabled",
                    severity="warning",
                    blocking=True,
                    remediation="Enable recording to log the run",
                )
            )

        safety_head = self.health_checker.get_safety_head_status() if self.health_checker else {}
        if safety_head and safety_head.get("status") not in {None, "ok", "ready"}:
            markers.append(
                TaskEligibilityMarker(
                    code="SAFETY_HEAD",
                    label="Safety head degraded",
                    severity="warning",
                    blocking=False,
                    source="safety_head",
                    remediation="Reset safety head before autonomous motion",
                )
            )

        if (
            self.mode_manager
            and self.mode_manager.current_mode != RobotMode.AUTONOMOUS
            and task.recommended_mode == "autonomous"
        ):
            markers.append(
                TaskEligibilityMarker(
                    code="MODE_HINT",
                    label="Switch to autonomous for best latency",
                    severity="info",
                    blocking=False,
                    source="studio",
                    remediation="Use the command deck to enable autonomous mode",
                )
            )

        eligible = not any(marker.blocking for marker in markers)
        next_poll_ms = 120.0 if task.requires_motion else 400.0
        return TaskEligibility(eligible=eligible, markers=markers, next_poll_after_ms=next_poll_ms)

    def _serialize_task_entry(self, task: TaskDefinition) -> TaskLibraryEntry:
        eligibility = self._build_task_eligibility(task)
        return TaskLibraryEntry(
            id=task.id,
            title=task.title,
            description=task.description,
            group=task.group,
            tags=task.tags,
            eligibility=eligibility,
            estimated_duration=task.estimated_duration,
            recommended_mode=task.recommended_mode,
        )

    def _build_task_summary(self, task: TaskDefinition) -> TaskSummary:
        entry = self._serialize_task_entry(task)
        return TaskSummary(
            entry=entry,
            required_modalities=task.required_modalities,
            steps=task.steps,
            owner="robot",
            updated_at=self._now_iso(),
            telemetry_topic=task.telemetry_topic,
        )
    
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
                self._ensure_mode_manager()

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
                self._ensure_mode_manager()

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
                if self.drivetrain.is_mock:
                    status["drivetrain"].update(
                        {
                            "warning": "MOCK drivetrain: hardware output unavailable",
                            "unavailable_reason": "Mock drivers loaded instead of PCA9685",
                        }
                    )

            if self.health_checker:
                status["safety_head"] = self.health_checker.get_safety_head_status()

            if self.selected_task_id:
                task_entry = self.task_library.get_entry(self.selected_task_id)
                if task_entry:
                    status["current_task"] = self._build_task_summary(task_entry).to_dict()
            else:
                status["current_task"] = None

            return {"success": True, "status": status}

        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    async def GetLoopHealth(self) -> dict:
        """Expose HOPE/CMS loop metrics, safety head status, and gates."""
        try:
            mode_manager = self._ensure_mode_manager()

            metrics = mode_manager.get_loop_metrics()
            gates = mode_manager.get_gate_snapshot()
            safety_head = self.health_checker.get_safety_head_status() if self.health_checker else None

            return {
                "success": True,
                "metrics": metrics,
                "gates": gates,
                "safety_head": safety_head,
            }
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    async def ListTasks(self, include_ineligible: bool = False) -> dict:
        """List Task Library entries with eligibility markers."""

        try:
            entries: List[Dict[str, object]] = []
            for task in self.task_library.list_entries():
                entry = self._serialize_task_entry(task)
                if include_ineligible or entry.eligibility.eligible:
                    entries.append(entry.to_dict())

            return {
                "success": True,
                "tasks": entries,
                "selected_task_id": self.selected_task_id,
                "source": "live",
            }
        except Exception as exc:
            return {"success": False, "message": f"Error: {exc}"}

    async def GetTaskSummary(self, task_id: str) -> dict:
        """Return a detail-rich summary for a single task."""

        try:
            task = self.task_library.get_entry(task_id)
            if not task:
                return {"success": False, "message": f"Unknown task: {task_id}"}

            summary = self._build_task_summary(task)
            return {"success": True, "summary": summary.to_dict()}
        except Exception as exc:
            return {"success": False, "message": f"Error: {exc}"}

    async def SelectTask(self, task_id: str, reason: Optional[str] = None) -> dict:
        """Select the next task for autonomy loops and Studio previews."""

        try:
            if not task_id:
                return {"success": False, "accepted": False, "message": "task_id is required"}

            task = self.task_library.get_entry(task_id)
            if not task:
                return {"success": False, "accepted": False, "message": f"Unknown task: {task_id}"}

            summary = self._build_task_summary(task)
            eligibility = summary.entry.eligibility
            if not eligibility.eligible:
                return {
                    "success": True,
                    "accepted": False,
                    "message": "Task blocked by eligibility markers",
                    "selected_task": summary.entry.to_dict(),
                    "eligibility": eligibility.to_dict(),
                }

            self.selected_task_id = task_id

            selection_message = reason or "Studio/Robot selection"
            if self.mode_manager and task.requires_motion and self.mode_manager.current_mode != RobotMode.AUTONOMOUS:
                selection_message += " • enable autonomous mode to execute"

            return {
                "success": True,
                "accepted": True,
                "message": selection_message,
                "selected_task": summary.entry.to_dict(),
                "eligibility": eligibility.to_dict(),
            }
        except Exception as exc:
            return {"success": False, "accepted": False, "message": f"Error: {exc}"}

    async def GetGates(self) -> dict:
        """Expose gate snapshot and safety envelope for UI badges."""

        try:
            mode_manager = self._ensure_mode_manager()
            gates = mode_manager.get_gate_snapshot()
            safety_head = self.health_checker.get_safety_head_status() if self.health_checker else None

            return {
                "success": True,
                "gates": gates,
                "safety_head": safety_head,
                "mode": mode_manager.current_mode.value,
            }
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    async def TriggerSafetyHold(self) -> dict:
        """Shortcut to emergency stop while returning gate state."""

        try:
            mode_manager = self._ensure_mode_manager()
            mode_manager.emergency_stop("Web safety hold")
            return {
                "success": True,
                "mode": mode_manager.current_mode.value,
                "gates": mode_manager.get_gate_snapshot(),
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    async def ResetSafetyGates(self) -> dict:
        """Return to idle to clear gate state for mock and hardware modes."""

        try:
            mode_manager = self._ensure_mode_manager()
            mode_manager.return_to_idle()
            return {
                "success": True,
                "mode": mode_manager.current_mode.value,
                "gates": mode_manager.get_gate_snapshot(),
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}
    
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
    
    # Chat configuration
    CHAT_HISTORY_LIMIT = 50  # Maximum number of chat messages to persist
    
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
        query_params = parse_qs(full_path.split('?', 1)[1]) if '?' in full_path else {}
        
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
        elif path == "/api/gates":
            gates = await self.service.GetGates()
            response_body = json.dumps(gates)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path in {"/api/tasks", "/api/tasks/"}:
            include_ineligible = query_params.get("include_ineligible", ["false"])[0].lower() == "true"
            result = await self.service.ListTasks(include_ineligible=include_ineligible)
            response_body = json.dumps(result)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path.startswith("/api/tasks/summary/"):
            task_id = path.split("/")[-1]
            result = await self.service.GetTaskSummary(task_id)
            response_body = json.dumps(result)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/tasks/select" and method == "POST":
            content_length = int(headers.get('content-length', 0))
            body = await reader.read(content_length) if content_length > 0 else b''
            payload = json.loads(body.decode()) if body else {}
            result = await self.service.SelectTask(payload.get("task_id", ""), reason=payload.get("reason"))
            response_body = json.dumps(result)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/safety/hold":
            result = await self.service.TriggerSafetyHold()
            response_body = json.dumps(result)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/safety/reset":
            result = await self.service.ResetSafetyGates()
            response_body = json.dumps(result)
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
    
    def get_chat_overlay_js(self):
        """Generate shared JavaScript code for chat overlay functionality.
        
        This code handles:
        - Chat state persistence in localStorage
        - Chat UI minimize/maximize toggle
        - Message rendering and history management
        - Sending messages to the Gemma chat API
        """
        return """
    def get_shared_chat_javascript(self):
        """Generate shared chat overlay JavaScript for both /ui and /control pages."""
        return f"""
        // Chat overlay persistence (shared between /ui and /control)
        var chatMinimized = false;
        var chatHistory = [];
        var chatStoragePrefix = 'gemma_chat_' + (window.location.host || 'local');
        var chatHistoryKey = chatStoragePrefix + '_history';
        var chatMinimizedKey = chatStoragePrefix + '_minimized';

        function persistChatState() {
            try {
                localStorage.setItem(chatHistoryKey, JSON.stringify(chatHistory.slice(-50)));
                localStorage.setItem(chatMinimizedKey, chatMinimized ? 'true' : 'false');
            } catch (e) {
                console.warn('Unable to persist chat state', e);
            }
        }

        function applyChatMinimized() {
        function persistChatState() {{
            try {{
                localStorage.setItem(chatHistoryKey, JSON.stringify(chatHistory.slice(-{self.CHAT_HISTORY_LIMIT})));
                localStorage.setItem(chatMinimizedKey, chatMinimized ? 'true' : 'false');
            }} catch (e) {{
                console.warn('Unable to persist chat state', e);
            }}
        }}

        function applyChatMinimized() {{
            var panel = document.getElementById('chat-panel');
            var toggle = document.getElementById('chat-toggle');
            if (!panel || !toggle) return;

            if (chatMinimized) {
                panel.classList.add('minimized');
                toggle.textContent = '+';
            } else {
                panel.classList.remove('minimized');
                toggle.textContent = '−';
            }
        }

        function renderChatMessage(text, role, shouldPersist) {
            if (chatMinimized) {{
                panel.classList.add('minimized');
                toggle.textContent = '+';
            }} else {{
                panel.classList.remove('minimized');
                toggle.textContent = '−';
            }}
        }}

        function renderChatMessage(text, role, shouldPersist) {{
            if (typeof shouldPersist === 'undefined') shouldPersist = true;

            var messagesDiv = document.getElementById('chat-messages');
            if (!messagesDiv) return;

            var messageDiv = document.createElement('div');
            messageDiv.className = 'chat-message ' + role;
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            if (shouldPersist) {
                chatHistory.push({role: role, content: text});
                persistChatState();
            }
        }

        function hydrateChatOverlay() {
            try {
                var storedHistory = localStorage.getItem(chatHistoryKey);
                if (storedHistory) {
                    chatHistory = JSON.parse(storedHistory) || [];
                    chatHistory.forEach(function(msg) {
                        renderChatMessage(msg.content, msg.role, false);
                    });
                }

                var storedMinimized = localStorage.getItem(chatMinimizedKey);
                if (storedMinimized === 'true') {
                    chatMinimized = true;
                }
            } catch (e) {
                console.warn('Unable to hydrate chat state', e);
            }

            applyChatMinimized();
        }

        window.toggleChat = function() {
            chatMinimized = !chatMinimized;
            persistChatState();
            applyChatMinimized();
        };

        window.addChatMessage = function(text, role) {
            renderChatMessage(text, role, true);
        };

        window.sendChatMessage = function() {
            if (shouldPersist) {{
                chatHistory.push({{role: role, content: text}});
                persistChatState();
            }}
        }}

        function hydrateChatOverlay() {{
            try {{
                var storedHistory = localStorage.getItem(chatHistoryKey);
                if (storedHistory) {{
                    chatHistory = JSON.parse(storedHistory) || [];
                    chatHistory.forEach(function(msg) {{
                        renderChatMessage(msg.content, msg.role, false);
                    }});
                }}

                var storedMinimized = localStorage.getItem(chatMinimizedKey);
                if (storedMinimized === 'true') {{
                    chatMinimized = true;
                }}
            }} catch (e) {{
                console.warn('Unable to hydrate chat state', e);
            }}

            applyChatMinimized();
        }}

        window.toggleChat = function() {{
            chatMinimized = !chatMinimized;
            persistChatState();
            applyChatMinimized();
        }};

        window.addChatMessage = function(text, role) {{
            renderChatMessage(text, role, true);
        }};

        window.sendChatMessage = function() {{
            var input = document.getElementById('chat-input');
            var sendBtn = document.getElementById('chat-send');
            var message = input ? input.value.trim() : '';

            if (!message) return;

            // Add user message
            addChatMessage(message, 'user');
            if (input) input.value = '';

            // Disable input while processing
            if (input) input.disabled = true;
            if (sendBtn) sendBtn.disabled = true;

            // Send to Gemma endpoint
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/chat', true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.onload = function() {
                if (input) input.disabled = false;
                if (sendBtn) sendBtn.disabled = false;

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

                if (input) input.focus();
            };
            xhr.onerror = function() {
            xhr.onload = function() {{
                if (input) input.disabled = false;
                if (sendBtn) sendBtn.disabled = false;

                if (xhr.status === 200) {{
                    try {{
                        var data = JSON.parse(xhr.responseText);
                        if (data.response) {{
                            addChatMessage(data.response, 'assistant');
                        }} else if (data.error) {{
                            addChatMessage('Error: ' + data.error, 'system');
                        }}
                    }} catch (e) {{
                        addChatMessage('Error parsing response', 'system');
                    }}
                }} else {{
                    addChatMessage('Server error: ' + xhr.status, 'system');
                }}

                if (input) input.focus();
            }};
            xhr.onerror = function() {{
                if (input) input.disabled = false;
                if (sendBtn) sendBtn.disabled = false;
                addChatMessage('Connection error', 'system');
                if (input) input.focus();
            };

            // Include chat history for context
            xhr.send(JSON.stringify({
                message: message,
                history: chatHistory.slice(-10) // Last 10 messages for context
            }));
        };

        hydrateChatOverlay();
"""
            }};

            // Include chat history for context
            xhr.send(JSON.stringify({{
                message: message,
                history: chatHistory.slice(-10) // Last 10 messages for context
            }}));
        }};

        hydrateChatOverlay();"""
    
    def get_web_ui_html(self):
        """Generate simple web UI for robot control."""
        html_before_chat_js = """<!DOCTYPE html>
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
            grid-template-columns: 280px 1fr 320px;
            align-items: start;
            gap: 18px;
        }

        .ide-sidebar {
            background: linear-gradient(180deg, rgba(18, 28, 44, 0.95), rgba(15, 23, 41, 0.98));
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 24px 20px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
            display: flex;
            flex-direction: column;
            gap: 16px;
            min-width: 280px;
        }

        .sidebar-title {
            font-size: 13px;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 4px;
            font-weight: 600;
        }
        
        .sidebar-section {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 12px;
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

        .task-panel-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
        }

        .task-card {
            padding: 14px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.02);
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .task-card h3 { margin: 0; font-size: 16px; }
        .task-card .task-meta { color: var(--muted); font-size: 12px; }

        .task-tags { display: flex; flex-wrap: wrap; gap: 6px; }
        .task-tag { font-size: 12px; padding: 4px 8px; border-radius: 8px; background: rgba(255,255,255,0.06); border: 1px solid var(--border); }

        .eligibility-marker { font-size: 12px; padding: 6px 8px; border-radius: 8px; border: 1px solid var(--border); }
        .eligibility-marker.blocking { border-color: rgba(255, 77, 109, 0.6); color: #ffb2c0; }
        .eligibility-marker.warning { border-color: rgba(255, 185, 87, 0.6); color: #ffd9a0; }
        .eligibility-marker.info { border-color: rgba(122, 215, 255, 0.6); color: #c9edff; }
        .eligibility-marker.success { border-color: rgba(56, 217, 150, 0.6); color: #c0f8e5; }

        .task-actions { display: flex; gap: 8px; justify-content: flex-start; align-items: center; }
        .task-actions button { padding: 8px 10px; border-radius: 10px; border: 1px solid var(--border); background: rgba(255,255,255,0.04); color: var(--text); cursor: pointer; }
        .task-actions button[disabled] { opacity: 0.5; cursor: not-allowed; }

        .selected-task-pill { padding: 10px 12px; border-radius: 999px; border: 1px solid var(--border); background: rgba(122, 215, 255, 0.07); font-size: 13px; }
        .selected-task-pill.success { border-color: rgba(56, 217, 150, 0.7); color: #c0f8e5; }
        .selected-task-pill.warning { border-color: rgba(255, 185, 87, 0.7); color: #ffd9a0; }
        
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

        /* Chat overlay shared with /control */
        .chat-overlay {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 360px;
            background: rgba(10, 12, 20, 0.95);
            border-radius: 12px;
            border: 1px solid rgba(122, 215, 255, 0.2);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45);
            color: #e8f0ff;
            display: flex;
            flex-direction: column;
            backdrop-filter: blur(8px);
            overflow: hidden;
            z-index: 999;
        }

        .chat-overlay.minimized {
            height: 52px;
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(90deg, rgba(122, 215, 255, 0.15), rgba(79, 157, 255, 0.1));
            padding: 12px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(122, 215, 255, 0.2);
            cursor: pointer;
        }

        .chat-header h3 {
            margin: 0;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.2px;
            color: #e8f0ff;
        }

        .chat-toggle {
            background: rgba(255, 255, 255, 0.08);
            color: #e8f0ff;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 6px 10px;
            cursor: pointer;
            font-weight: 700;
        }

        .chat-messages {
            padding: 14px;
            height: 340px;
            overflow-y: auto;
            background: rgba(255, 255, 255, 0.02);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .chat-message {
            padding: 10px 12px;
            border-radius: 10px;
            max-width: 85%;
            font-size: 13px;
            line-height: 1.45;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        }

        .chat-message.user { margin-left: auto; background: #4f9dff; color: #0b1020; }
        .chat-message.assistant { background: rgba(122, 215, 255, 0.08); border: 1px solid rgba(122, 215, 255, 0.2); }
        .chat-message.system { margin: 0 auto; background: rgba(255,255,255,0.05); color: #9aa3ba; }

        .chat-input-area {
            display: flex;
            gap: 10px;
            padding: 12px 14px;
            border-top: 1px solid rgba(122, 215, 255, 0.15);
            background: rgba(0, 0, 0, 0.35);
        }

        .chat-input {
            flex: 1;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 10px 12px;
            color: #e8f0ff;
            font-size: 13px;
            outline: none;
        }

        .chat-input:focus { border-color: rgba(122, 215, 255, 0.6); box-shadow: 0 0 0 3px rgba(122, 215, 255, 0.12); }

        .chat-send-btn {
            background: linear-gradient(135deg, #4f9dff, #7ad7ff);
            color: #0b1020;
            border: none;
            border-radius: 10px;
            padding: 10px 14px;
            cursor: pointer;
            font-weight: 700;
            box-shadow: 0 10px 30px rgba(79, 157, 255, 0.35);
        }

        .chat-send-btn:disabled { opacity: 0.6; cursor: not-allowed; box-shadow: none; }
        .agent-rail {
            background: linear-gradient(180deg, rgba(16, 24, 38, 0.96), rgba(13, 20, 32, 0.96));
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.35);
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-width: 300px;
        }

        .rail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
        }

        .rail-title {
            margin: 4px 0 0 0;
            font-size: 18px;
        }

        .rail-actions {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .rail-btn {
            background: rgba(255, 255, 255, 0.04);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 12px;
            cursor: pointer;
        }

        .rail-btn:hover { border-color: var(--accent); }
        .rail-btn.primary { background: linear-gradient(135deg, #7ad7ff, #4f9dff); color: #0b1020; }

        .human-toggle {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid var(--border);
            padding: 8px 12px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.02);
            cursor: pointer;
        }

        .human-toggle.active { border-color: var(--accent); box-shadow: 0 0 0 1px rgba(122, 215, 255, 0.2); }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 12px;
            font-size: 12px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.03);
        }

        .status-chip.active { border-color: rgba(56, 217, 150, 0.5); color: #8df5c7; }
        .status-chip.paused { border-color: rgba(255, 149, 0, 0.5); color: #ffcf99; }
        .status-chip.focus { border-color: rgba(79, 157, 255, 0.5); color: #a9c7ff; }

        .agent-thread-list { display: flex; flex-direction: column; gap: 10px; }

        .agent-thread {
            padding: 12px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.02);
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 8px;
        }

        .agent-thread h4 { margin: 0; font-size: 15px; }
        .agent-meta { color: var(--muted); font-size: 12px; }

        .learning-list { display: flex; flex-direction: column; gap: 8px; margin-top: 6px; }
        .learning-item {
            padding: 10px;
            border-radius: 10px;
            border: 1px dashed var(--border);
            background: rgba(255, 255, 255, 0.02);
        }
        .learning-item strong { display: block; font-size: 13px; margin-bottom: 4px; }
        .learning-meta { color: var(--muted); font-size: 12px; }

        .milestone-list { display: flex; flex-direction: column; gap: 8px; margin-top: 6px; }
        .milestone-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 10px;
            border-radius: 10px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.02);
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.05);
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #4f9dff, #7ad7ff);
            transition: width 0.25s ease;
        }

        .agent-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
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
                <div class="sidebar-section">
                    <div class="sidebar-title">Command Deck</div>
                    <div class="command-grid">
                        <button id="manual-control-btn" class="command-btn primary" onclick="startManualControl(this)">🎮 Manual Control</button>
                        <button class="command-btn" onclick="setMode('autonomous')">🚀 Autonomous</button>
                        <button class="command-btn" onclick="setMode('sleep_learning')">💤 Sleep Learning</button>
                        <button class="command-btn subtle" onclick="setMode('idle')">⏸️ Idle</button>
                    </div>
                </div>
                
                <div class="sidebar-section">
                    <div class="sidebar-title">Safety & System</div>
                    <div class="command-grid">
                        <button class="command-btn danger" onclick="setMode('emergency_stop')">🛑 Emergency Stop</button>
                        <button class="command-btn" onclick="window.triggerSafetyHold()">🛡️ Safety Hold</button>
                        <button class="command-btn subtle" onclick="window.resetSafetyGates()">♻️ Reset Gates</button>
                        <button class="command-btn subtle" onclick="alert('Settings modal would open here')">⚙️ Settings</button>
                    </div>
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
                    <div class="agent-chip-row" id="agent-chip-row">
                        <!-- Agent status chips injected by JS -->
                    </div>
                </section>

                <section class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-eyebrow">Task Library</div>
                            <h2>Autonomy Task Deck</h2>
                            <p class="panel-subtitle">Group tasks by intent, check eligibility markers, and hand off to the robot.</p>
                        </div>
                        <div class="selected-task-pill" id="current-task-pill">No task selected</div>
                    </div>
                    <div class="task-panel-grid" id="task-groups">
                        <div class="status-item">
                            <span class="status-label">Loading tasks...</span>
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

                </section>

                <!-- Tabs for View Switcher -->
                <div class="tabs-container">
                    <button class="tab-btn active" data-tab="dashboard" onclick="switchHomeTab('dashboard')">📊 Dashboard</button>
                    <button class="tab-btn" data-tab="robot-view" onclick="switchHomeTab('robot-view')">🤖 Robot Layout</button>
                </div>

                <!-- Dashboard View (Sensors + Workspace) -->
                <div id="dashboard-panel" class="home-panel">
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

                    <section class="panel" style="margin-top: 14px;">
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
                                <p class="canvas-text">Snapshot of the current behavior lane.</p>
                            </div>
                            <div class="canvas-card">
                                <div class="canvas-title">Safety Boundaries</div>
                                <p class="canvas-text">Motion gates active.</p>
                            </div>
                        </div>
                    </section>
                </div>

                <!-- Robot Layout View -->
                <div id="robot-view-panel" class="home-panel" style="display: none;">
                    <section class="panel">
                        <div class="panel-header">
                            <div>
                                <div class="panel-eyebrow">Visual Status</div>
                                <h2>System Health Map</h2>
                                <p class="panel-subtitle">Component-level health and connectivity status.</p>
                            </div>
                        </div>
                        
                        <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; padding: 40px 0;">
                            <!-- Visual Robot Representation -->
                            <div style="position: relative; width: 300px; height: 300px; border: 1px dashed var(--border); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                                
                                <!-- Base -->
                                <div style="position: absolute; bottom: 40px; width: 120px; height: 60px; background: rgba(255,255,255,0.05); border: 1px solid var(--accent); border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <span style="font-size: 10px; color: var(--accent);">BASE / DRIVETRAIN</span>
                                    <span id="viz-base-status" style="font-size: 16px;">🟢</span>
                                </div>
                                
                                <!-- Arm -->
                                <div style="position: absolute; top: 80px; right: 40px; width: 40px; height: 120px; background: rgba(255,255,255,0.05); border: 1px solid var(--accent); border-radius: 8px; transform: rotate(15deg); display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <span style="font-size: 10px; color: var(--accent); writing-mode: vertical-lr;">ARM/GRIPPER</span>
                                    <span id="viz-arm-status" style="font-size: 16px; margin-top: 4px;">🟢</span>
                                </div>

                                <!-- Head/Camera -->
                                <div style="position: absolute; top: 40px; left: 80px; width: 80px; height: 60px; background: rgba(255,255,255,0.05); border: 1px solid var(--accent); border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <span style="font-size: 10px; color: var(--accent);">VISION</span>
                                    <span id="viz-vision-status" style="font-size: 16px;">🟢</span>
                                </div>
                                
                                <!-- Core -->
                                <div style="width: 80px; height: 80px; background: rgba(122, 215, 255, 0.1); border-radius: 50%; border: 2px solid var(--accent-strong); display: flex; align-items: center; justify-content: center; flex-direction: column; box-shadow: 0 0 30px rgba(79, 157, 255, 0.2);">
                                    <span style="font-size: 10px; color: #fff; font-weight: bold;">BRAIN</span>
                                    <span id="viz-brain-status" style="font-size: 20px;">🧠</span>
                                </div>
                            </div>
                            
                            <div style="flex: 1; min-width: 250px;">
                                <div class="status-item">
                                    <span class="status-label">Overall Safety Check</span>
                                    <span class="status-value status-good">PASSED</span>
                                </div>
                                <div class="status-item">
                                    <span class="status-label">Battery / Power</span>
                                    <span class="status-value">100% (Simulated)</span>
                                </div>
                                <div class="status-item">
                                    <span class="status-label">NPU Load</span>
                                    <span class="status-value">Low</span>
                                </div>
                            </div>
                        </div>
                    </section>
                </div>


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

            <aside class="agent-rail">
                <div class="rail-header">
                    <div>
                        <div class="panel-eyebrow">Agent Manager</div>
                        <div class="rail-title">Threads & Guidance</div>
                    </div>
                    <div class="human-toggle" id="human-toggle" onclick="toggleHumanMode()">
                        <span>Human mode</span>
                        <strong id="human-toggle-state">Off</strong>
                    </div>
                </div>

                <div class="rail-actions">
                    <button class="rail-btn primary" onclick="resumeAgents()">▶️ Resume agents</button>
                    <button class="rail-btn" onclick="pauseAgents()">⏸️ Pause agents</button>
                    <button class="rail-btn" onclick="reviewLearning()">📒 Review learning</button>
                </div>

                <div class="panel" style="padding: 14px; background: rgba(255,255,255,0.01); border-color: var(--border);">
                    <div class="panel-eyebrow">Active Agents</div>
                    <div class="agent-thread-list" id="agent-thread-list"></div>
                </div>

                <div class="panel" style="padding: 14px; background: rgba(255,255,255,0.01); border-color: var(--border);">
                    <div class="panel-eyebrow">Robot Mode Self-Learning</div>
                    <div class="milestone-list" id="learning-milestones"></div>
                </div>

                <div class="panel" style="padding: 14px; background: rgba(255,255,255,0.01); border-color: var(--border);">
                    <div class="panel-eyebrow">Recent Learning Events</div>
                    <div class="learning-list" id="learning-events"></div>
                </div>
            </aside>
        </div>
    </div>

    <!-- Chat Interface -->
    <div class="chat-overlay" id="chat-panel">
        <div class="chat-header" onclick="toggleChat()" onkeypress="if(event.key==='Enter'||event.key===' ') toggleChat()" tabindex="0" role="button" aria-label="Toggle chat panel">
            <h3>🤖 Gemma 3n Assistant</h3>
            <button class="chat-toggle" id="chat-toggle">−</button>
        </div>
        <div class="chat-messages" id="chat-messages">
        </div>
        <div class="chat-input-area">
            <input type="text" class="chat-input" id="chat-input" placeholder="Ask about robot status, control tips..." aria-label="Chat message input" onkeypress="if(event.key==='Enter') sendChatMessage()">
            <button class="chat-send-btn" id="chat-send" onclick="sendChatMessage()">➤</button>
        </div>
    </div>

    <script type="text/javascript">
        // Tab switching for Home Page
        window.switchHomeTab = function(tabName) {
            document.querySelectorAll('.tab-btn').forEach(btn => {
                if(btn.dataset.tab === tabName) btn.classList.add('active');
                else btn.classList.remove('active');
            });

            document.querySelectorAll('.home-panel').forEach(panel => {
                if(panel.id === tabName + '-panel') panel.style.display = 'block';
                else panel.style.display = 'none';
            });
        };
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

        const agentManagerState = {
            humanMode: false,
            agents: [
                { name: 'Safety Guardian', status: 'active', threads: 2, focus: 'Gates + envelopes', milestone: 'Monitoring' },
                { name: 'Navigator', status: 'active', threads: 1, focus: 'Drive pathfinding', milestone: 'Updating map' },
                { name: 'Trainer', status: 'paused', threads: 1, focus: 'Self-learning batches', milestone: 'Queued' },
            ],
            learningEvents: [
                { title: 'Episode flagged for replay', time: '2m ago', detail: 'Marked safe drive lane for offline finetune' },
                { title: 'Gate alignment saved', time: '8m ago', detail: 'Motion + recording gates synced to idle baseline' },
                { title: 'Autonomy pulse', time: '12m ago', detail: 'Wave/particle balance stable; buffer widened' },
            ],
        };

        function renderAgentChips(agents) {
            const chipRow = document.getElementById('agent-chip-row');
            if (!chipRow) return;

            chipRow.innerHTML = agents.map(agent => {
                const normalizedStatus = (agent.status || 'active').toLowerCase();
                return `<span class="status-chip ${normalizedStatus}"><span class="badge-dot"></span>${agent.name} — ${agent.threads || 1} thread${(agent.threads || 1) > 1 ? 's' : ''}</span>`;
            }).join('');
        }

        function renderAgentThreads(agents) {
            const list = document.getElementById('agent-thread-list');
            if (!list) return;

            if (!agents || !agents.length) {
                list.innerHTML = '<div class="agent-thread"><span class="agent-meta">No active agents</span></div>';
                return;
            }

            list.innerHTML = agents.map(agent => {
                const normalizedStatus = (agent.status || 'active').toLowerCase();
                const chip = `<span class="status-chip ${normalizedStatus}">${normalizedStatus === 'active' ? '🟢' : '⏸️'} ${normalizedStatus}</span>`;
                return `<div class="agent-thread">` +
                    `<div>` +
                        `<h4>${agent.name || 'Agent'}</h4>` +
                        `<div class="agent-meta">${agent.focus || 'Monitoring'} • ${agent.threads || 1} thread${(agent.threads || 1) > 1 ? 's' : ''}</div>` +
                    `</div>` +
                    `<div>${chip}</div>` +
                `</div>`;
            }).join('');
        }

        function renderLearningMilestones(status) {
            const container = document.getElementById('learning-milestones');
            if (!container) return;

            const gateSnapshot = status.gate_snapshot || status.gates || {};
            const modeLabel = (status.mode || 'idle').replace(/_/g, ' ');
            const uptime = gateSnapshot.mode_uptime_seconds || 0;
            const progress = Math.min(100, Math.round((uptime % 180) / 180 * 100));

            const milestones = [
                { label: 'Motion gate open', done: !!gateSnapshot.allow_motion, detail: gateSnapshot.allow_motion ? 'Ready to move' : 'Waiting for clearance' },
                { label: 'Recording allowed', done: !!gateSnapshot.record_episodes, detail: gateSnapshot.record_episodes ? 'Episodes gated in' : 'Recording gated off' },
                { label: 'Self-train enabled', done: !!gateSnapshot.self_train, detail: gateSnapshot.self_train ? 'Robot learning live' : 'Manual/human-in-loop' },
                { label: `${modeLabel} uptime`, done: progress > 40, detail: `${Math.round(uptime)}s in mode`, progress },
            ];

            container.innerHTML = milestones.map(item => {
                const chipClass = item.done ? 'status-chip active' : 'status-chip paused';
                const progressBar = typeof item.progress === 'number'
                    ? `<div class="progress-bar"><div class="progress-fill" style="width:${item.progress}%"></div></div>`
                    : '';
                return `<div class="milestone-row">` +
                    `<div>` +
                        `<div>${item.label}</div>` +
                        `<div class="learning-meta">${item.detail}</div>` +
                        `${progressBar}` +
                    `</div>` +
                    `<span class="${chipClass}">${item.done ? '✅' : '…'} ${item.done ? 'Complete' : 'Pending'}</span>` +
                `</div>`;
            }).join('');
        }

        function renderLearningEvents(events) {
            const list = document.getElementById('learning-events');
            if (!list) return;
            const items = events && events.length ? events : [{ title: 'No learning events yet', time: '', detail: 'Agents will surface new checkpoints here.' }];

            list.innerHTML = items.map(event => {
                return `<div class="learning-item">` +
                    `<strong>${event.title}</strong>` +
                    `<div class="learning-meta">${event.detail || 'Update pending'}</div>` +
                    (event.time ? `<div class="learning-meta">${event.time}</div>` : '') +
                `</div>`;
            }).join('');
        }

        function renderSelectedTask(selectedTask) {
            const pill = document.getElementById('current-task-pill');
            if (!pill) return;

            if (!selectedTask || !selectedTask.entry) {
                pill.textContent = 'No task selected';
                pill.className = 'selected-task-pill warning';
                return;
            }

            const entry = selectedTask.entry;
            const eligible = entry.eligibility ? entry.eligibility.eligible : false;
            pill.textContent = `${entry.title || 'Task'} • ${entry.group || 'Task Library'}`;
            pill.className = 'selected-task-pill ' + (eligible ? 'success' : 'warning');
        }

        function renderTaskLibrary(payload) {
            const container = document.getElementById('task-groups');
            if (!container) return;

            const tasks = (payload && payload.tasks) || [];
            const selectedId = payload ? payload.selected_task_id : null;
            if (!tasks.length) {
                container.innerHTML = '<div class="status-item"><span class="status-label">No tasks available</span></div>';
                return;
            }

            const groups = {};
            tasks.forEach(task => {
                const groupKey = task.group || 'Tasks';
                if (!groups[groupKey]) {
                    groups[groupKey] = [];
                }
                groups[groupKey].push(task);
            });

            container.innerHTML = Object.entries(groups).map(([groupLabel, entries]) => {
                const cards = entries.map(task => {
                    const eligibility = task.eligibility || {};
                    const markers = (eligibility.markers || []).map(marker => {
                        const severity = marker.blocking ? 'blocking' : (marker.severity || 'info');
                        const remediation = marker.remediation ? ` — ${marker.remediation}` : '';
                        const label = marker.label || marker.code || 'marker';
                        return `<div class="eligibility-marker ${severity}">${label}${remediation}</div>`;
                    }).join('') || '<div class="eligibility-marker success">Eligible</div>';

                    const tagRow = (task.tags || []).map(tag => `<span class="task-tag">${tag}</span>`).join('');
                    const isSelected = selectedId && selectedId === task.id;
                    const selectLabel = isSelected ? 'Selected' : 'Select task';
                    const disabledAttr = eligibility.eligible ? '' : 'disabled';
                    const highlightStyle = isSelected ? 'style="border-color: rgba(122,215,255,0.7);"' : '';

                    return `<div class="task-card" ${highlightStyle}>` +
                        `<div>` +
                            `<h3>${task.title || 'Task'}</h3>` +
                            `<div class="task-meta">${task.description || ''}</div>` +
                            `<div class="task-tags">${tagRow}</div>` +
                        `</div>` +
                        `<div>${markers}</div>` +
                        `<div class="task-actions">` +
                            `<button onclick="window.selectTask('${task.id}')" ${disabledAttr}>${selectLabel}</button>` +
                            `<span class="task-meta">${task.estimated_duration || ''} • ${task.recommended_mode || ''}</span>` +
                        `</div>` +
                    `</div>`;
                }).join('');

                return `<div>` +
                    `<div class="panel-eyebrow">${groupLabel}</div>` +
                    `<div class="task-panel-grid">${cards}</div>` +
                `</div>`;
            }).join('');
        }

        function renderAgentRail(status) {
            const agents = (status && Array.isArray(status.agent_threads) && status.agent_threads.length)
                ? status.agent_threads
                : agentManagerState.agents;

            const events = (status && Array.isArray(status.learning_events) && status.learning_events.length)
                ? status.learning_events
                : agentManagerState.learningEvents;

            renderAgentThreads(agents);
            renderAgentChips(agents);
            renderLearningMilestones(status || {});
            renderLearningEvents(events);

            const toggle = document.getElementById('human-toggle');
            const toggleState = document.getElementById('human-toggle-state');
            if (toggle) {
                toggle.classList.toggle('active', agentManagerState.humanMode);
            }
            if (toggleState) {
                toggleState.textContent = agentManagerState.humanMode ? 'On' : 'Off';
            }
        }

        window.toggleHumanMode = function() {
            agentManagerState.humanMode = !agentManagerState.humanMode;
            window.showMessage(agentManagerState.humanMode ? 'Human guidance injected' : 'Human mode disabled');
            renderAgentRail();
        };

        window.pauseAgents = function() {
            agentManagerState.agents = agentManagerState.agents.map(agent => ({ ...agent, status: 'paused' }));
            window.showMessage('Agents paused for inspection');
            renderAgentRail();
        };

        window.resumeAgents = function() {
            agentManagerState.agents = agentManagerState.agents.map(agent => ({ ...agent, status: 'active' }));
            window.showMessage('Agents resumed');
            renderAgentRail();
        };

        window.reviewLearning = function() {
            window.showMessage('Learning feed refreshed');
            renderAgentRail();
            const events = document.getElementById('learning-events');
            if (events) {
                events.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        };

        window.triggerSafetyHold = async function() {
            window.showMessage('Engaging safety hold...');
            try {
                const response = await fetch('/api/safety/hold', { method: 'POST' });
                const data = await response.json();
                if (data && data.success) {
                    window.showMessage('Safety hold engaged');
                    window.updateStatus();
                } else {
                    window.showMessage('Hold failed', true);
                }
            } catch (err) {
                console.error(err);
                window.showMessage('Hold request failed', true);
            }
        };

        window.resetSafetyGates = async function() {
            window.showMessage('Resetting gates to idle baseline...');
            try {
                const response = await fetch('/api/safety/reset', { method: 'POST' });
                const data = await response.json();
                if (data && data.success) {
                    window.showMessage('Gates reset to idle');
                    window.updateStatus();
                } else {
                    window.showMessage('Reset failed', true);
                }
            } catch (err) {
                console.error(err);
                window.showMessage('Reset request failed', true);
            }
        };

        function renderLoopTelemetry(status) {
            var loops = status.loop_metrics || status.metrics || {};
            var gates = status.gate_snapshot || status.gates || {};
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
                var beatAgeMs = heartbeat.last_beat ? (Date.now() - heartbeat.last_beat * 1000) : null;
                var beatAgeLabel = beatAgeMs ? ' • ' + Math.round(beatAgeMs) + 'ms ago' : '';
                heartbeatBadge.textContent = heartbeat.ok ? 'Heartbeat stable' + beatAgeLabel : 'Heartbeat delayed';
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
                var safetyBeat = safety.heartbeat || {};
                var beatDelta = safetyBeat.timestamp_ns ? ((Date.now() * 1e6 - safetyBeat.timestamp_ns) / 1e9).toFixed(1) : null;
                var beatLabel = beatDelta ? ' • ' + beatDelta + 's ago' : '';
                safetyHeartbeat.textContent = safetyBeat.ok ? 'Online (' + (safetyBeat.source || 'safety') + beatLabel + ')' : 'Simulated';
            }
        }

        window.selectTask = async function(taskId) {
            if (!taskId) return;
            try {
                const response = await fetch('/api/tasks/select', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task_id: taskId, reason: 'studio-selection' })
                });
                if (!response.ok) {
                    throw new Error('HTTP ' + response.status);
                }
                const payload = await response.json();
                window.showMessage(payload.message || 'Task selection updated', !payload.accepted);
                if (payload.selected_task) {
                    renderSelectedTask({ entry: payload.selected_task });
                }
                window.fetchTaskLibrary();
                window.updateStatus();
            } catch (err) {
                console.warn('Task selection failed', err);
                window.showMessage('Failed to select task', true);
            }
        };

        window.fetchTaskLibrary = async function() {
            try {
                const response = await fetch('/api/tasks?include_ineligible=true');
                if (!response.ok) { return; }
                const payload = await response.json();
                renderTaskLibrary(payload);

                if (payload && payload.selected_task_id && Array.isArray(payload.tasks)) {
                    const selected = payload.tasks.find(task => task.id === payload.selected_task_id);
                    if (selected) {
                        renderSelectedTask({ entry: selected });
                    }
                }
            } catch (err) {
                console.warn('Task library fetch failed', err);
            }
        };

        // Tab switching logic for Manual Control
        window.switchTab = function(tabName) {
            // Update tab buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {
                if(btn.dataset.tab === tabName) btn.classList.add('active');
                else btn.classList.remove('active');
            });

            // Update panels
            document.querySelectorAll('.control-panel').forEach(panel => {
                if(panel.id === tabName + '-panel') panel.style.display = 'block';
                else panel.style.display = 'none';
            });
        };
        
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
                            renderAgentRail(data.status);
                            renderSelectedTask(data.status.current_task);

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

        window.startManualControl = async function(buttonEl) {
            const manualButton = buttonEl || document.getElementById('manual-control-btn');
            if (!manualButton) {
                window.showMessage('Manual Control button not found', true);
                return;
            }

            if (manualButton.disabled) {
                return;
            }

            const originalText = manualButton.textContent;
            manualButton.disabled = true;
            manualButton.textContent = 'Switching...';

            try {
                window.showMessage('Switching to manual control...');
                const response = await fetch('/api/mode/manual_control', { method: 'POST' });
                if (!response.ok) {
                    throw new Error('Server responded with ' + response.status);
                }

                const payload = await response.json();
                if (payload && payload.success) {
                    window.showMessage('Manual control enabled, redirecting...');
                    window.updateStatus();
                    setTimeout(() => { window.location.href = '/control'; }, 300);
                } else {
                    const message = (payload && payload.message) ? payload.message : 'Unable to enable manual control';
                    window.showMessage(message, true);
                    window.updateStatus();
                    manualButton.disabled = false;
                    manualButton.textContent = originalText;
                }
            } catch (err) {
                console.error('Manual control switch failed', err);
                window.showMessage('Failed to switch to manual control', true);
                window.updateStatus();
                manualButton.disabled = false;
                manualButton.textContent = originalText;
            }
        };

        window.pollLoopHealth = async function() {
            try {
                const response = await fetch('/api/loops');
                if (!response.ok) { return; }
                const payload = await response.json();
                if (payload && payload.success) {
                    renderLoopTelemetry({
                        loop_metrics: payload.metrics,
                        gate_snapshot: payload.gates,
                        safety_head: payload.safety_head
                    });
                }
            } catch (err) {
                console.warn('Loop telemetry fetch failed', err);
            }
        };
"""
        
        html_after_chat_js = """
""" + self.get_shared_chat_javascript() + """

        // Chat overlay persistence (/ui + /control)
        var chatMinimized = false;
        var chatHistory = [];
        var chatStoragePrefix = 'gemma_chat_' + (window.location.host || 'local');
        var chatHistoryKey = chatStoragePrefix + '_history';
        var chatMinimizedKey = chatStoragePrefix + '_minimized';
        var MAX_MESSAGE_LENGTH = 10000; // Maximum allowed message length for DoS protection
        var initialChatMessage = 'Chat with Gemma 3n about robot control';

        // Sanitize text to prevent XSS attacks
        function sanitizeText(text) {
            if (typeof text !== 'string') {
                return '';
            }
            
            // Remove any HTML tags first
            var sanitized = text.replace(/<[^>]*>/g, '');
            
            // Remove any remaining < or > characters that might be part of incomplete tags
            sanitized = sanitized.replace(/[<>]/g, '');
            
            // Remove javascript: and data: URL schemes
            sanitized = sanitized.replace(/javascript:/gi, '');
            sanitized = sanitized.replace(/data:/gi, '');
            
            // Remove common XSS event handlers
            sanitized = sanitized.replace(/on\\w+\\s*=/gi, '');
            
            // Limit length to prevent DOS attacks
            return sanitized.substring(0, 10000);
        }

        function persistChatState() {
            try {
                // Trim in-memory history as well
                if (chatHistory.length > 50) {
                    chatHistory = chatHistory.slice(-50);
                }
                localStorage.setItem(chatHistoryKey, JSON.stringify(chatHistory));
                localStorage.setItem(chatMinimizedKey, chatMinimized ? 'true' : 'false');
            } catch (e) {
                console.warn('Unable to persist chat state', e);
                // Show a user-visible warning
                alert('Warning: Unable to save chat history. Your messages may not be saved.');
            }
        }

        function applyChatMinimized() {
            var panel = document.getElementById('chat-panel');
            var toggle = document.getElementById('chat-toggle');
            if (!panel || !toggle) return;

            if (chatMinimized) {
                panel.classList.add('minimized');
                toggle.textContent = '+';
            } else {
                panel.classList.remove('minimized');
                toggle.textContent = '−';
            }
        }

        function validateChatContent(content) {
            // Validate chat content to prevent injection attacks
            // Ensure content is a string and within reasonable bounds
            if (typeof content !== 'string') return '';
            // Limit message length to prevent DoS
            if (content.length > MAX_MESSAGE_LENGTH) return content.substring(0, MAX_MESSAGE_LENGTH);
            return content;
        }

        function validateChatRole(role) {
            // Validate role to prevent class injection
            var validRoles = ['user', 'assistant', 'system'];
            return validRoles.indexOf(role) !== -1 ? role : 'system';
        }

        function renderChatMessage(text, role, shouldPersist) {
            if (typeof shouldPersist === 'undefined') shouldPersist = true;

            var messagesDiv = document.getElementById('chat-messages');
            if (!messagesDiv) return;

            // Validate inputs to prevent injection attacks
            var validatedText = validateChatContent(text);
            var validatedRole = validateChatRole(role);

            var messageDiv = document.createElement('div');
            messageDiv.className = 'chat-message ' + validatedRole;
            messageDiv.textContent = validatedText; // textContent prevents XSS
            // Sanitize text before rendering
            var sanitized = sanitizeText(text);
            
            var messageDiv = document.createElement('div');
            messageDiv.className = 'chat-message ' + role;
            messageDiv.textContent = sanitized;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            if (shouldPersist) {
                chatHistory.push({role: validatedRole, content: validatedText});
                chatHistory.push({role: role, content: sanitized});
                persistChatState();
            }
        }

        function hydrateChatOverlay() {
            try {
                var storedHistory = localStorage.getItem(chatHistoryKey);
                if (storedHistory) {
                    try {
                        chatHistory = JSON.parse(storedHistory);
                        if (!Array.isArray(chatHistory)) {
                            chatHistory = [];
                        }
                    } catch (parseError) {
                        console.warn('Failed to parse chat history from localStorage, using empty history instead:', parseError);
                        chatHistory = [];
                    }
                    chatHistory.forEach(function(msg) {
                        if (msg && typeof msg === 'object' && msg.role && msg.content) {
                            renderChatMessage(msg.content, msg.role, false);
                        }
                    });
            var storedHistory = localStorage.getItem(chatHistoryKey);
            if (storedHistory) {
                try {
                    chatHistory = JSON.parse(storedHistory);
                    if (!Array.isArray(chatHistory)) {
                        chatHistory = [];
                    }
                } catch (parseError) {
                    console.warn('Failed to parse chat history from localStorage:', parseError);
                    chatHistory = [];
                }
                chatHistory.forEach(function(msg) {
                    renderChatMessage(msg.content, msg.role, false);
                });
            }

            try {
                var storedMinimized = localStorage.getItem(chatMinimizedKey);
                if (storedMinimized === 'true') {
                    chatMinimized = true;
                }
            } catch (e) {
                console.warn('Failed to parse chat minimized state from localStorage:', e);
                    console.warn('Failed to parse chat history from localStorage', parseError);
                    chatHistory = [];
            try {
                var storedHistory = localStorage.getItem(chatHistoryKey);
                if (storedHistory) {
                    try {
                        chatHistory = JSON.parse(storedHistory);
                        if (!Array.isArray(chatHistory)) {
                            console.warn('Chat history is not an array (found ' + typeof chatHistory + '), resetting to empty');
                            chatHistory = [];
                        }
                    } catch (parseError) {
                        console.warn('Failed to parse chat history from localStorage (invalid JSON), initializing with empty history', parseError);
                            chatHistory = [];
                        }
                    } catch (parseError) {
                        console.warn('Failed to parse chat history', parseError);
                        chatHistory = [];
                    }
                    chatHistory.forEach(function(msg) {
                        if (msg && typeof msg === 'object' && msg.role && msg.content) {
                            renderChatMessage(msg.content, msg.role, false);
                        }
                    });
                } else {
                    // Only add initial message if no history exists
                    chatHistory.push({role: 'system', content: initialChatMessage});
                    renderChatMessage(initialChatMessage, 'system', false);
                    persistChatState();
                }
                chatHistory.forEach(function(msg) {
                    if (msg && typeof msg === 'object' && typeof msg.role === 'string' && typeof msg.content === 'string') {
                        renderChatMessage(msg.content, msg.role, false);
                    }
                });
            }

            var storedMinimized = localStorage.getItem(chatMinimizedKey);
            if (storedMinimized === 'true') {
                chatMinimized = true;
            }

            applyChatMinimized();
        }

        window.toggleChat = function() {
            chatMinimized = !chatMinimized;
            persistChatState();
            applyChatMinimized();
        };

        window.addChatMessage = function(text, role) {
            renderChatMessage(text, role, true);
        };

        window.sendChatMessage = function() {
            var input = document.getElementById('chat-input');
            var sendBtn = document.getElementById('chat-send');
            var message = input ? input.value.trim() : '';

            if (!message) return;

            // Add user message
            addChatMessage(message, 'user');
            if (input) input.value = '';

            // Disable input while processing
            if (input) input.disabled = true;
            if (sendBtn) sendBtn.disabled = true;

            // Send to Gemma endpoint
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/chat', true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.onload = function() {
                if (input) input.disabled = false;
                if (sendBtn) sendBtn.disabled = false;

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

                if (input) input.focus();
            };
            xhr.onerror = function() {
                if (input) input.disabled = false;
                if (sendBtn) sendBtn.disabled = false;
                addChatMessage('Connection error', 'system');
                if (input) input.focus();
            };

            // Include chat history for context
            xhr.send(JSON.stringify({
                message: message,
                history: chatHistory.slice(-10) // Last 10 messages for context
            }));
        };

        hydrateChatOverlay();

        // Update status every 2 seconds
        renderAgentRail();
        window.updateStatus();
        window.pollLoopHealth();
        window.fetchTaskLibrary();
        setInterval(window.updateStatus, 2000);
        setInterval(window.pollLoopHealth, 1500);
        setInterval(window.fetchTaskLibrary, 1000);
    </script>
</body>
</html>"""
        
        return html_before_chat_js + self.get_chat_overlay_js() + html_after_chat_js
    
    def get_control_interface_html(self):
        """Generate live control interface with camera feed and system status."""
        html_before_chat_js = """<!DOCTYPE html>
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
            background: #000;
            color: #fff;
            overflow: hidden;
        }
        
        /* Fullscreen Video Background */
        .video-panel {
            position: absolute;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            z-index: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .video-feed {
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.8;
        }
        
        .video-overlay {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 1;
            background: rgba(0, 0, 0, 0.5);
            padding: 8px 12px;
            border-radius: 8px;
            pointer-events: none;
        }

        /* Floating Header */
        .header {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 10;
            background: rgba(15, 23, 41, 0.85);
            backdrop-filter: blur(12px);
            padding: 12px 24px;
            border-radius: 40px;
            display: flex;
            align-items: center;
            gap: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .header h1 {
            font-size: 16px;
            color: #fff;
            margin: 0;
        }
        
        .back-btn {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.2s;
        }
        .back-btn:hover { background: rgba(255, 255, 255, 0.2); }

        /* Tabs Centered Bottom */
        .tabs-container {
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 20;
            background: rgba(15, 23, 41, 0.9);
            backdrop-filter: blur(12px);
            padding: 6px;
            border-radius: 100px;
            display: flex;
            gap: 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .tab-btn {
            background: transparent;
            border: none;
            color: rgba(255,255,255,0.6);
            padding: 12px 24px;
            border-radius: 30px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .tab-btn.active {
            background: var(--accent);
            color: #0b1020;
            box-shadow: 0 4px 12px rgba(122, 215, 255, 0.3);
        }
        
        /* Layout Container (Right Side Overlay) */
        .main-container {
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            width: 360px;
            z-index: 10;
            padding: 80px 20px 20px 0;
            pointer-events: none; /* Let clicks pass through empty areas */
            display: block; /* Reset grid */
            background: transparent;
            height: auto;
            max-width: none;
            margin: 0;
        }

        .status-panel {
            pointer-events: auto;
            background: transparent;
            border: none;
            padding: 0;
            box-shadow: none;
            overflow: visible;
            display: flex;
            flex-direction: column;
            gap: 16px;
            height: 100%;
        }

        .control-panel {
            background: rgba(16, 22, 38, 0.85);
            backdrop-filter: blur(16px);
            padding: 20px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .status-section { display: none; } /* Hide generic status blocks in immersive mode */

        /* Joint Sliders & buttons styling (Keep existing logic but refine) */
        .joint-slider {
            background: rgba(0,0,0,0.2);
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 8px;
        }
        .joint-slider label { color: #ccc; font-size: 12px; display: flex; justify-content: space-between; margin-bottom: 4px; }
        .val-badge { color: var(--accent); font-family: monospace; }
        
        .arrow-group { background: rgba(0,0,0,0.2); padding: 12px; border-radius: 12px; }
        .arrow-btn { background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; }
        .arrow-btn:active { background: var(--accent); color: #000; }
        
        .emergency-btn {
            background: #ff3b30;
            color: white;
            border: none;
            padding: 16px;
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
    <div class="video-panel">
        <img src="/api/camera/stream" class="video-feed" onerror="this.src='data:image/svg+xml;base64,...'">
        <div class="video-overlay">
             <div class="status-item">
                <span style="width: 8px; height: 8px; background: #34c759; border-radius: 50%; display: inline-block; margin-right: 6px;"></span>
                LIVE FEED (CAM_1)
            </div>
        </div>
    </div>

    <div class="header">
        <h1>🎮 Manual Control</h1>
        <button class="back-btn" onclick="window.location.href='/ui'">Exit</button>
    </div>

    <!-- Tabs (Bottom Center) -->
    <div class="tabs-container">
        <button class="tab-btn active" data-tab="arm" onclick="switchTab('arm')">🦾 Arm</button>
        <button class="tab-btn" data-tab="drive" onclick="switchTab('drive')">🏎️ Drive</button>
    </div>

    <div class="main-container">
        <div class="status-panel">
            <!-- Joint/Arm Control Panel (Overlay) -->
            <div id="arm-panel" class="control-panel">
                <h3>Arm Manipulation</h3>
                <!-- ... existing controls ... -->
                <div class="joint-controls">
                    <div class="joint-slider">
                        <label>J0 (Base Rotation) <span class="val-badge" id="j0-display">0.0</span></label>
                        <input type="range" id="j0" min="-100" max="100" value="0" oninput="updateJointDisplay(0, this.value); sendJointCommand()">
                    </div>
                    <!-- ... other sliders ... -->
                    <div class="joint-slider">
                        <label>J1 (Shoulder) <span class="val-badge" id="j1-display">0.0</span></label>
                        <input type="range" id="j1" min="-100" max="100" value="0" oninput="updateJointDisplay(1, this.value); sendJointCommand()">
                    </div>
                    <div class="joint-slider">
                        <label>J2 (Elbow) <span class="val-badge" id="j2-display">0.0</span></label>
                        <input type="range" id="j2" min="-100" max="100" value="0" oninput="updateJointDisplay(2, this.value); sendJointCommand()">
                    </div>
                    <div class="joint-slider">
                        <label>J3 (Wrist Roll) <span class="val-badge" id="j3-display">0.0</span></label>
                        <input type="range" id="j3" min="-100" max="100" value="0" oninput="updateJointDisplay(3, this.value); sendJointCommand()">
                    </div>
                    <div class="joint-slider">
                        <label>J4 (Wrist Pitch) <span class="val-badge" id="j4-display">0.0</span></label>
                        <input type="range" id="j4" min="-100" max="100" value="0" oninput="updateJointDisplay(4, this.value); sendJointCommand()">
                    </div>
                    <div class="joint-slider">
                        <label>Gripper State <span class="val-badge" id="j5-display">-1.0</span></label>
                        <input type="range" id="j5" min="-100" max="100" value="-100" oninput="updateJointDisplay(5, this.value); sendJointCommand()">
                    </div>
                </div>
                
                 <div style="margin-top: 20px;">
                    <span class="status-label">Presets & Actions</span>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;">
                        <button class="action-btn" onclick="gotoHome()">🏠 Home</button>
                        <button class="action-btn" onclick="gotoZero()">0️⃣ Zero</button>
                        <button class="action-btn success" onclick="openGripper()">✋ Open</button>
                        <button class="action-btn warning" onclick="closeGripper()">✊ Close</button>
                    </div>
                </div>

                     <div class="arrow-controls" style="margin-top: 20px;">
                        <div class="arrow-group">
                            <div class="arrow-group-title">Fine Control (Selected Joint)</div>
                            <div class="arrow-grid">
                                <div></div>
                                <button class="arrow-btn" onmousedown="startMove(activeJoint, 0.1)" onmouseup="stopMove()">+</button>
                                <div></div>
                                <button class="arrow-btn" onmousedown="startMove(activeJoint, -0.1)" onmouseup="stopMove()">-</button>
                                <div></div>
                            </div>
                            <div class="keyboard-hint" style="margin-top: 8px;">Select a joint slider first</div>
                        </div>
                    </div>
            </div>

            <!-- Drive Control Panel (Overlay) -->
            <div id="drive-panel" class="control-panel" style="display: none;">
                <h3>Drivetrain</h3>
                <!-- ... existing controls ... -->
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

                    <div class="arrow-group" style="margin-top: 20px;">
                        <div class="arrow-group-title">Speed limit: <span id="speed-level">SLOW</span> (<span id="speed-value">0.3</span>)</div>
                        <div style="display: flex; gap: 4px;">
                            <button style="flex: 1; background: #34c759; color: white; border: none; padding: 12px; border-radius: 6px; cursor: pointer; font-size: 11px;" onclick="setSpeed(0.2, 'CRAWL')">🐌 Crawl</button>
                            <button style="flex: 1; background: #007aff; color: white; border: none; padding: 12px; border-radius: 6px; cursor: pointer; font-size: 11px;" onclick="setSpeed(0.3, 'SLOW')">🚪 Slow</button>
                            <button style="flex: 1; background: #ff9500; color: white; border: none; padding: 12px; border-radius: 6px; cursor: pointer; font-size: 11px;" onclick="setSpeed(0.5, 'MED')">🚶 Med</button>
                            <button style="flex: 1; background: #ff3b30; color: white; border: none; padding: 12px; border-radius: 6px; cursor: pointer; font-size: 11px;" onclick="setSpeed(0.7, 'FAST')">🏃 Fast</button>
                        </div>
                    </div>
                </div>
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
"""
        
        html_after_chat_js = """
        """ + self.get_shared_chat_javascript() + """
        
        // Chat functionality with persistence
        var chatMinimized = false;
        var chatHistory = [];
        var chatStoragePrefix = 'gemma_chat_' + (window.location.host || 'local');
        var chatHistoryKey = chatStoragePrefix + '_history';
        var chatMinimizedKey = chatStoragePrefix + '_minimized';
        var MAX_MESSAGE_LENGTH = 10000; // Maximum allowed message length for DoS protection
        var initialChatMessage = 'Chat with Gemma 3n about robot control';

        // Sanitize text to prevent XSS attacks
        function sanitizeText(text) {
            if (typeof text !== 'string') {
                return '';
            }
            
            // Remove any HTML tags first
            var sanitized = text.replace(/<[^>]*>/g, '');
            
            // Remove any remaining < or > characters that might be part of incomplete tags
            sanitized = sanitized.replace(/[<>]/g, '');
            
            // Remove javascript: and data: URL schemes
            sanitized = sanitized.replace(/javascript:/gi, '');
            sanitized = sanitized.replace(/data:/gi, '');
            
            // Remove common XSS event handlers
            sanitized = sanitized.replace(/on\\w+\\s*=/gi, '');
            
            // Limit length to prevent DOS attacks
            return sanitized.substring(0, 10000);
        }

        function persistChatState() {
            try {
                // Trim in-memory history as well
                if (chatHistory.length > 50) {
                    chatHistory = chatHistory.slice(-50);
                }
                localStorage.setItem(chatHistoryKey, JSON.stringify(chatHistory));
                localStorage.setItem(chatMinimizedKey, chatMinimized ? 'true' : 'false');
            } catch (e) {
                console.warn('Unable to persist chat state', e);
                // Show a user-visible warning
                alert('Warning: Unable to save chat history. Your messages may not be saved.');
            }
        }

        function applyChatMinimized() {
            var panel = document.getElementById('chat-panel');
            var toggle = document.getElementById('chat-toggle');
            if (!panel || !toggle) return;

            if (chatMinimized) {
                panel.classList.add('minimized');
                toggle.textContent = '+';
            } else {
                panel.classList.remove('minimized');
                toggle.textContent = '−';
            }
        }

        function validateChatContent(content) {
            // Validate chat content to prevent injection attacks
            // Ensure content is a string and within reasonable bounds
            if (typeof content !== 'string') return '';
            // Limit message length to prevent DoS
            if (content.length > MAX_MESSAGE_LENGTH) return content.substring(0, MAX_MESSAGE_LENGTH);
            return content;
        }

        function validateChatRole(role) {
            // Validate role to prevent class injection
            var validRoles = ['user', 'assistant', 'system'];
            return validRoles.indexOf(role) !== -1 ? role : 'system';
        }

        function renderChatMessage(text, role, shouldPersist) {
            if (typeof shouldPersist === 'undefined') shouldPersist = true;

            var messagesDiv = document.getElementById('chat-messages');
            if (!messagesDiv) return;

            // Validate inputs to prevent injection attacks
            var validatedText = validateChatContent(text);
            var validatedRole = validateChatRole(role);

            var messageDiv = document.createElement('div');
            messageDiv.className = 'chat-message ' + validatedRole;
            messageDiv.textContent = validatedText; // textContent prevents XSS
            // Sanitize text before rendering
            var sanitized = sanitizeText(text);
            
            var messageDiv = document.createElement('div');
            messageDiv.className = 'chat-message ' + role;
            messageDiv.textContent = sanitized;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            if (shouldPersist) {
                chatHistory.push({role: validatedRole, content: validatedText});
                chatHistory.push({role: role, content: sanitized});
                persistChatState();
            }
        }

        function hydrateChatOverlay() {
            var storedHistory = localStorage.getItem(chatHistoryKey);
            if (storedHistory) {
                try {
                    chatHistory = JSON.parse(storedHistory);
                    if (!Array.isArray(chatHistory)) {
                        chatHistory = [];
                    }
                } catch (parseError) {
                    console.warn('Failed to parse chat history', parseError);
                    chatHistory = [];
            try {
                var storedHistory = localStorage.getItem(chatHistoryKey);
                if (storedHistory) {
                    try {
                        chatHistory = JSON.parse(storedHistory);
                        if (!Array.isArray(chatHistory)) {
                            chatHistory = [];
                        }
                    } catch (parseError) {
                        console.warn('Failed to parse chat history from localStorage, using empty history instead:', parseError);
                        console.warn('Failed to parse chat history', parseError);
                        chatHistory = [];
                    }
                    chatHistory.forEach(function(msg) {
                        if (msg && typeof msg === 'object' && msg.role && msg.content) {
                            renderChatMessage(msg.content, msg.role, false);
                        }
                    });
                } else {
                    // Only add initial message if no history exists
                    chatHistory.push({role: 'system', content: initialChatMessage});
                    renderChatMessage(initialChatMessage, 'system', false);
                    persistChatState();
                }
                chatHistory.forEach(function(msg) {
                    if (msg && typeof msg === 'object' && msg.role && msg.content) {
                        renderChatMessage(msg.content, msg.role, false);
                    }
                });
            }

            var storedMinimized = localStorage.getItem(chatMinimizedKey);
            if (storedMinimized === 'true') {
                chatMinimized = true;
            }

            applyChatMinimized();
        }

        window.toggleChat = function() {
            chatMinimized = !chatMinimized;
            persistChatState();
            applyChatMinimized();
        };

        window.addChatMessage = function(text, role) {
            renderChatMessage(text, role, true);
        };

        window.sendChatMessage = function() {
            var input = document.getElementById('chat-input');
            var sendBtn = document.getElementById('chat-send');
            var message = input ? input.value.trim() : '';

            if (!message) return;

            // Add user message
            addChatMessage(message, 'user');
            if (input) input.value = '';

            // Disable input while processing
            if (input) input.disabled = true;
            if (sendBtn) sendBtn.disabled = true;

            // Send to Gemma endpoint
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/chat', true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.onload = function() {
                if (input) input.disabled = false;
                if (sendBtn) sendBtn.disabled = false;

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

                if (input) input.focus();
            };
            xhr.onerror = function() {
                if (input) input.disabled = false;
                if (sendBtn) sendBtn.disabled = false;
                addChatMessage('Connection error', 'system');
                if (input) input.focus();
            };

            // Include chat history for context
            xhr.send(JSON.stringify({
                message: message,
                history: chatHistory.slice(-10) // Last 10 messages for context
            }));
        };

        hydrateChatOverlay();
        
        // Update every 100ms for responsive control
        window.updateControlStatus();
        setInterval(window.updateControlStatus, 100);
        
        // Update camera feed at ~30 FPS
        window.updateCameraFrame();
        setInterval(window.updateCameraFrame, 33);
    </script>
</body>
</html>"""
        
        return html_before_chat_js + self.get_chat_overlay_js() + html_after_chat_js
    
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
                    elif method == "list_tasks":
                        params = command.get("params", {})
                        include_ineligible = bool(params.get("include_ineligible")) if isinstance(params, dict) else False
                        response = await self.service.ListTasks(include_ineligible=include_ineligible)
                    elif method == "get_task_summary":
                        response = await self.service.GetTaskSummary(
                            command.get("params", {}).get("task_id", "")
                        )
                    elif method == "select_task":
                        params = command.get("params", {})
                        response = await self.service.SelectTask(
                            params.get("task_id", ""), reason=params.get("reason") if isinstance(params, dict) else None
                        )
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