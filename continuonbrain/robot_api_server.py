"""
ContinuonBrain Robot API server for Pi5 robot arm.
Runs against real hardware by default with optional mock fallback for dev.
"""

from continuonbrain.actuators.pca9685_arm import PCA9685ArmController
from continuonbrain.actuators.drivetrain_controller import DrivetrainController
from continuonbrain.sensors.oak_depth import OAKDepthCapture
from continuonbrain.recording.arm_episode_recorder import ArmEpisodeRecorder
from continuonbrain.robot_modes import RobotModeManager, RobotMode
from continuonbrain.system_context import SystemContext
from continuonbrain.system_health import SystemHealthChecker
from continuonbrain.system_instructions import SystemInstructions
from continuonbrain.server.chat import build_chat_service
from continuonbrain.gemma_chat import create_gemma_chat as _create_gemma_chat
from continuonbrain.server.model_selector import select_model
from continuonbrain.server.devices import auto_detect_hardware, init_recorder
from continuonbrain.server.status import start_status_server
from continuonbrain.server import routes as server_routes
from continuonbrain.server.tasks import (
    TaskDefinition,
    TaskEligibilityMarker,
    TaskEligibility,
    TaskLibraryEntry,
    TaskSummary,
    TaskLibrary,
)
from continuonbrain.server.skills import (
    SkillDefinition,
    SkillEligibility,
    SkillEligibilityMarker,
    SkillLibraryEntry,
    SkillSummary,
    SkillLibrary,
)
from continuonbrain.services.chat_adapter import ChatAdapter
from continuonbrain.services.training_runner import TrainingRunner
from continuonbrain.services.manual_trainer import ManualTrainer, ManualTrainerRequest
from continuonbrain.services.wavecore_trainer import WavecoreTrainer
from continuonbrain.services.video_stream import VideoStreamHelper
from continuonbrain.eval.hope_eval_runner import run_hope_eval_and_log

# Use extracted SimpleJSONServer implementation
SimpleJSONServer = server_routes.SimpleJSONServer

# Backward-compatible chat factory for tests and mocks.
def create_gemma_chat(use_mock: bool = False):
    return _create_gemma_chat(use_mock=use_mock)



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
        allow_mock_fallback: bool = False,
        system_instructions: Optional[SystemInstructions] = None,
        skip_motion_hw: bool = False,
    ):
        self.config_dir = config_dir
        self.prefer_real_hardware = prefer_real_hardware
        self.auto_detect = auto_detect
        self.allow_mock_fallback = allow_mock_fallback
        self.skip_motion_hw = skip_motion_hw
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
        self.skill_library = SkillLibrary()
        self.selected_task_id: Optional[str] = None
        self.training_runner = TrainingRunner()
        self.wavecore_trainer = WavecoreTrainer()
        self.manual_trainer = ManualTrainer()
        self.stream_helper = VideoStreamHelper(self)

        # Prefer JAX-based models by default to avoid heavyweight transformers init on boot.
        self.prefer_jax_models = os.environ.get("CONTINUON_PREFER_JAX", "1").lower() in ("1", "true", "yes")
        # Record selected model for status reporting
        selection = select_model()
        self.selected_model = selection.get("selected")

        if self.prefer_jax_models and self.selected_model and self.selected_model.get("backend") == "jax":
            print("  CONTINUON_PREFER_JAX=1 and JAX detected -> skipping transformers chat init; use inference_router for JAX.")
            self.gemma_chat = None
        else:
            # Transformers path (fallback)
            try:
                self.gemma_chat = build_chat_service()
            except Exception as e:  # noqa: BLE001
                print(f"  Chat initialization failed ({e}); continuing without chat service.")
                self.gemma_chat = None

        # Status log for selected model/backend
        if self.selected_model:
            print(f" Model selected: {self.selected_model.get('name')} (backend={self.selected_model.get('backend')})")
        else:
            print("  No chat/model backend selected; running without chat.")

        # Start lightweight status endpoint
        status_port = int(os.environ.get("CONTINUON_STATUS_PORT", "8090"))
        try:
            self.status_server = start_status_server(self.selected_model, port=status_port)
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed to start status endpoint: {exc}")

        self.chat_adapter = ChatAdapter(
            config_dir=config_dir,
            status_provider=self.GetRobotStatus,
            gemma_chat=self.gemma_chat,
        )

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

    async def RunTraining(self):
        """Trigger a background training run using the shared TrainingRunner."""
        await self.training_runner.run()

    async def RunManualTraining(self, payload: Optional[dict] = None) -> dict:
        """Run manual JAX trainer with optional overrides from payload."""
        payload = payload or {}
        request = ManualTrainerRequest(
            rlds_dir=Path(payload["rlds_dir"]) if payload.get("rlds_dir") else None,
            use_synthetic=bool(payload.get("use_synthetic", False)),
            max_steps=int(payload.get("max_steps", 10)),
            batch_size=int(payload.get("batch_size", 4)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            obs_dim=int(payload.get("obs_dim", 128)),
            action_dim=int(payload.get("action_dim", 32)),
            output_dim=int(payload.get("output_dim", 32)),
            disable_jit=bool(payload.get("disable_jit", True)),
            metrics_path=Path(payload["metrics_path"]) if payload.get("metrics_path") else None,
        )
        return await self.manual_trainer.run(request)

    async def RunWavecoreLoops(self, payload: Optional[dict] = None) -> dict:
        """Run WaveCore fast/mid/slow loops using the JAX CoreModel seed."""
        payload = payload or {}
        # Ensure service is available to downstream eval runner
        payload.setdefault("service", self)
        return await self.wavecore_trainer.run_loops(payload)

    async def RunHopeEval(self, payload: Optional[dict] = None) -> dict:
        """Run graded HOPE Q&A, log RLDS episode, with fallback LLM ordering."""
        payload = payload or {}
        questions_path = Path(payload.get("questions_path") or (REPO_ROOT / "continuonbrain" / "eval" / "hope_eval_questions.json"))
        rlds_dir = Path(payload.get("rlds_dir") or "/opt/continuonos/brain/rlds/episodes")
        use_fallback = bool(payload.get("use_fallback", True))
        fallback_order = payload.get("fallback_order") or ["hailo", "google/gemma-370m", "google/gemma-3n-2b"]
        return await run_hope_eval_and_log(
            service=self,
            questions_path=questions_path,
            rlds_dir=rlds_dir,
            use_fallback=use_fallback,
            fallback_order=fallback_order,
        )

    async def RunFactsEval(self, payload: Optional[dict] = None) -> dict:
        """Run FACTS-lite eval and log RLDS episode."""
        payload = payload or {}
        questions_path = Path(payload.get("questions_path") or (REPO_ROOT / "continuonbrain" / "eval" / "facts_eval_questions.json"))
        rlds_dir = Path(payload.get("rlds_dir") or "/opt/continuonos/brain/rlds/episodes")
        use_fallback = bool(payload.get("use_fallback", True))
        fallback_order = payload.get("fallback_order") or ["hailo", "google/gemma-370m", "google/gemma-3n-2b"]
        return await run_hope_eval_and_log(
            service=self,
            questions_path=questions_path,
            rlds_dir=rlds_dir,
            use_fallback=use_fallback,
            fallback_order=fallback_order,
            episode_prefix="facts_eval",
            model_label="facts-lite",
        )

    async def initialize(self):
        """Initialize hardware components with auto-detection."""
        mode_label = "REAL HARDWARE" if self.prefer_real_hardware else "MOCK"
        print(f"Initializing Robot Service ({mode_label} MODE)...")
        print()

        self._ensure_system_instructions()

        # Auto-detect hardware (used for status reporting)
        if self.auto_detect:
            print(" Auto-detecting hardware...")
            detected = auto_detect_hardware()
            if detected.devices:
                self.detected_config = detected.config
                print(f" Detected devices: {len(detected.devices)}")
                for dev in detected.devices:
                    print(f" - {dev}")
                print()
            else:
                print("  No hardware detected!")
                print()

        # Initialize recorder and hardware (prefers real, falls back to mock if allowed)
        print(" Initializing episode recorder...")
        self.recorder = init_recorder(self.config_dir, max_steps=500)

        hardware_ready = False
        if self.prefer_real_hardware:
            print(" Initializing hardware via ContinuonBrain...")
            hardware_ready = self.recorder.initialize_hardware(
                use_mock=False,
                auto_detect=self.auto_detect,
            )
            self.arm = self.recorder.arm
            self.camera = self.recorder.camera

            if not hardware_ready:
                print("  Real hardware initialization incomplete")
                # If motion is skipped, allow camera-only success to proceed without raising.
                if not self.allow_mock_fallback and not self.skip_motion_hw:
                    raise RuntimeError("Failed to initialize arm or camera in real mode")
                print("  Falling back to mock mode")

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

        print(" Episode recorder ready")

        if self.skip_motion_hw:
            print("  Skipping arm/drivetrain init (skip-motion-hw). Motion will stay mock.")
            self.arm = None
            self.drivetrain = None
        else:
            # Initialize drivetrain controller for steering/throttle
            print(" Initializing drivetrain controller...")
            self.drivetrain = DrivetrainController()
            drivetrain_ready = self.drivetrain.initialize()
            if drivetrain_ready:
                print(f" Drivetrain ready ({self.drivetrain.mode.upper()} MODE)")
            else:
                print("  Drivetrain controller unavailable")

        # Initialize mode manager
        print(" Initializing mode manager...")
        self.mode_manager = RobotModeManager(
            config_dir=self.config_dir,
            system_instructions=self.system_instructions,
        )
        self.mode_manager.return_to_idle()  # Start in idle mode
        print(" Mode manager ready")

        print()
        print("=" * 60)
        print(f" Robot Service Ready ({'REAL' if self.use_real_hardware else 'MOCK'} MODE)")
        print("=" * 60)
        if self.use_real_hardware and self.detected_config.get("primary"):
            print("\n Using detected hardware:")
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

        # Hardware Capability Checks
        caps = self.capabilities
        for modality in task.required_modalities:
            if modality == "vision" and not caps["has_vision"]:
                markers.append(TaskEligibilityMarker(
                    code="MISSING_VISION", label="Vision hardware required",
                    severity="error", blocking=True, remediation="Connect camera"
                ))
            elif (modality == "arm" or modality == "gripper") and not caps["has_manipulator"]:
                markers.append(TaskEligibilityMarker(
                    code="MISSING_ARM", label="Manipulator required",
                    severity="error", blocking=True, remediation="Connect arm actuators"
                ))
            elif modality == "mobile_base" and not caps["has_mobile_base"]:
                markers.append(TaskEligibilityMarker(
                    code="MISSING_BASE", label="Mobile base required",
                    severity="error", blocking=True, remediation="Connect drivetrain"
                ))

        eligible = not any(marker.blocking for marker in markers)
        next_poll_ms = 120.0 if task.requires_motion else 400.0
        return TaskEligibility(eligible=eligible, markers=markers, next_poll_after_ms=next_poll_ms)

    @property
    def capabilities(self) -> Dict[str, bool]:
        """Return detected hardware capabilities."""
        # In mock mode, we usually simulate everything, but let's respect the initialization state
        # effectively if self.arm is None, we don't have an arm.
        return {
            "has_vision": self.camera is not None,
            "has_manipulator": self.arm is not None,
            "has_mobile_base": self.drivetrain is not None and self.drivetrain.initialized,
            "has_audio": bool(self.recorder and self.recorder.audio_enabled),
        }

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

    def _build_skill_eligibility(self, skill: SkillDefinition) -> SkillEligibility:
        markers: List[SkillEligibilityMarker] = []
        caps = self.capabilities
        for cap in skill.capabilities:
            if cap == "manipulator" and not caps["has_manipulator"]:
                markers.append(SkillEligibilityMarker(code="MISSING_ARM", label="Manipulator required", severity="error", blocking=True))
            if cap == "vision" and not caps["has_vision"]:
                markers.append(SkillEligibilityMarker(code="MISSING_VISION", label="Vision required", severity="error", blocking=True))
            if cap == "teleop" and not caps["has_mobile_base"]:
                markers.append(SkillEligibilityMarker(code="MISSING_BASE", label="Mobile base needed", severity="warning", blocking=False))
        eligible = not any(m.blocking for m in markers)
        return SkillEligibility(eligible=eligible, markers=markers, next_poll_after_ms=400.0)

    def _serialize_skill_entry(self, skill: SkillDefinition) -> SkillLibraryEntry:
        eligibility = self._build_skill_eligibility(skill)
        return SkillLibraryEntry(
            id=skill.id,
            title=skill.title,
            description=skill.description,
            group=skill.group,
            tags=skill.tags,
            capabilities=skill.capabilities,
            eligibility=eligibility,
            estimated_duration=skill.estimated_duration,
            publisher=skill.publisher,
            version=skill.version,
        )

    def _build_skill_summary(self, skill: SkillDefinition) -> SkillSummary:
        entry = self._serialize_skill_entry(skill)
        steps = [
            "Load skill policy or planner",
            "Validate constraints and hardware",
            "Execute and stream telemetry",
        ]
        return SkillSummary(
            entry=entry,
            steps=steps,
            publisher=skill.publisher,
            version=skill.version,
            provenance=skill.provenance,
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
        async for state in self.stream_helper.stream_robot_state(client_id):
            yield state
    
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
                    action_source = "human_teleop_xr"
                    if self.mode_manager and self.mode_manager.current_mode == RobotMode.AUTONOMOUS:
                        action_source = "vla_policy"

                    if action_source not in {
                        "human_teleop_xr",
                        "human_dev_xr",
                        "human_supervisor",
                        "vla_policy",
                    }:
                        return {
                            "success": False,
                            "message": f"Invalid action_source {action_source}",
                        }

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
                "capabilities": self.capabilities,
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
            
            # Add battery status if available
            try:
                from continuonbrain.sensors.battery_monitor import BatteryMonitor
                monitor = BatteryMonitor()
                battery_status = monitor.get_diagnostics()
                if battery_status:
                    status["battery"] = battery_status
            except Exception:
                pass  # Battery monitor unavailable, continue without it

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

    async def ListSkills(self, include_ineligible: bool = False) -> dict:
        """List Skill Library entries with eligibility markers."""

        try:
            entries: List[Dict[str, object]] = []
            for skill in self.skill_library.list_entries():
                entry = self._serialize_skill_entry(skill)
                if include_ineligible or entry.eligibility.eligible:
                    entries.append(entry.to_dict())

            return {
                "success": True,
                "skills": entries,
                "selected_skill_id": None,
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

    async def GetSkillSummary(self, skill_id: str) -> dict:
        """Return summary for a skill."""

        try:
            skill = self.skill_library.get_entry(skill_id)
            if not skill:
                return {"success": False, "message": f"Unknown skill: {skill_id}"}

            summary = self._build_skill_summary(skill)
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
        return await self.stream_helper.get_depth_frame()
    
    async def GetCameraFrameJPEG(self) -> Optional[bytes]:
        """Get latest RGB camera frame as JPEG bytes."""
        return await self.stream_helper.get_camera_frame_jpeg()
    
    async def ChatWithGemma(self, message: str, history: list) -> dict:
        """
        Chat with Gemma 3n model acting as the Agent Manager.
        
        Args:
            message: User's message
            history: Chat history for context
            
        Returns:
            dict with 'response' or 'error'
        """
        return await self.chat_adapter.chat(message, history)
    
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


class _LegacySimpleJSONServer:
    """Deprecated stub; use continuonbrain.server.routes.SimpleJSONServer instead."""
    pass


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
        "--skip-motion-hw",
        action="store_true",
        help="Skip arm/drivetrain init (use mock motion even in real mode)"
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
        skip_motion_hw=args.skip_motion_hw,
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