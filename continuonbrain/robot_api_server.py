"""
ContinuonBrain Robot API server for Pi5 robot arm.
Runs against real hardware by default with optional mock fallback for dev.
"""

import asyncio
import threading
import os
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

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
import subprocess
import time
from continuonbrain.services.audio_io import record_wav, speak_text
from continuonbrain.services.pairing_manager import PairingManager
try:
    # Optional; requires torch.
    from continuonbrain.services.background_learner import BackgroundLearner
except Exception:  # noqa: BLE001
    BackgroundLearner = None  # type: ignore
from continuonbrain.services.video_stream import VideoStreamHelper
from continuonbrain.eval.hope_eval_runner import run_hope_eval_and_log
from continuonbrain.resource_monitor import ResourceMonitor, ResourceLevel

# Use extracted SimpleJSONServer implementation
SimpleJSONServer = server_routes.SimpleJSONServer

# Repo root for resolving bundled question files when running from source.
REPO_ROOT = Path(__file__).resolve().parent.parent

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
        # JAX-based trainers/tool-router are intentionally lazy-imported to keep the
        # Robot API server robust on devices that don't have working JAX/tensorstore
        # wheels (common on Pi images).
        self.wavecore_trainer = None
        self.manual_trainer = None
        self.tool_router_trainer = None
        self.tool_router_evaluator = None
        self.stream_helper = VideoStreamHelper(self)
        self._tool_router_bundle = None
        self.pairing = PairingManager(config_dir)

        # Control-loop timing telemetry (real scheduling + work-time in this process).
        # Used by the Research dashboard to prove "<=100ms tick" claims.
        self._control_loop_stop = threading.Event()
        self._control_loop_thread: Optional[threading.Thread] = None
        self._control_loop_period_ms = deque(maxlen=600)
        self._control_loop_work_ms = deque(maxlen=600)
        self._control_loop_deadline_misses = 0
        self._control_loop_samples = 0
        self._control_loop_target_allow_motion_s = _parse_env_float("CONTINUON_CONTROL_TICK_S", 0.05)
        self._control_loop_target_idle_s = _parse_env_float("CONTINUON_IDLE_TICK_S", 0.25)
        self._control_loop_target_ms = _parse_env_int("CONTINUON_CONTROL_TICK_TARGET_MS", 100)

        # Prefer JAX-based models by default to avoid heavyweight transformers init on boot.
        self.prefer_jax_models = os.environ.get("CONTINUON_PREFER_JAX", "1").lower() in ("1", "true", "yes")
        # Record selected model for status reporting
        selection = select_model()
        self.selected_model = selection.get("selected")

        # Detect Hailo accelerator for offloading
        accelerator_device = None
        try:
            from pathlib import Path
            hailo_devices = list(Path("/dev").glob("hailo*"))
            if hailo_devices:
                accelerator_device = "hailo8l"
                print(f"  Detected Hailo AI HAT+ ({len(hailo_devices)} device(s)) - enabling offloading")
        except Exception:
            pass

        if self.prefer_jax_models and self.selected_model and self.selected_model.get("backend") == "jax":
            print("  CONTINUON_PREFER_JAX=1 and JAX detected -> skipping transformers chat init; use inference_router for JAX.")
            self.gemma_chat = None
        else:
            # Transformers path (fallback)
            try:
                self.gemma_chat = build_chat_service()
                # If build_chat_service didn't detect Hailo, try to set it here
                if accelerator_device and self.gemma_chat and hasattr(self.gemma_chat, 'accelerator_device'):
                    if not self.gemma_chat.accelerator_device:
                        self.gemma_chat.accelerator_device = accelerator_device
                        print(f"  Set accelerator_device={accelerator_device} on GemmaChat")
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
        # Background supervisors (threads; keep working even if no asyncio loop is kept alive).
        self._bg_stop_event = threading.Event()
        self._chat_learn_thread: Optional[threading.Thread] = None
        self._autonomous_learner_thread: Optional[threading.Thread] = None
        self._orchestrator_thread: Optional[threading.Thread] = None
        self._orchestrator_lock = threading.Lock()
        self.hope_brain = None
        self.background_learner = None
        self._last_autonomous_learner_action: Optional[dict] = None
        self.resource_monitor = ResourceMonitor(config_dir=Path(config_dir))
        self._last_orchestrator: dict = {"last_run_ts": 0.0, "last_actions": {}}

    def _check_safety_protocol(self, action: str, context: dict = None) -> tuple[bool, str]:
        """Check if an action complies with safety protocol.
        
        Args:
            action: Description of the action to be taken
            context: Additional context for the decision
            
        Returns:
            (is_safe, reason) tuple
        """
        if not self.system_instructions:
            return False, "Safety protocol not loaded"
        
        if context is None:
            context = {}
        
        # Add current system state to context
        if self.mode_manager:
            context['current_mode'] = self.mode_manager.current_mode.value
            context['gates'] = self.mode_manager.get_gate_snapshot()
        
        # Validate action against safety protocol
        is_safe, reason, violated_rules = self.system_instructions.safety_protocol.validate_action(
            action, context
        )
        
        # Log the safety decision (if logging is available)
        if violated_rules:
            import logging
            logging.getLogger(__name__).warning(
                f"Safety check failed for action '{action}': {reason}. "
                f"Violated rules: {len(violated_rules)}"
            )
        
        return is_safe, reason

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
        # Lazy import: JAX stack can be absent/broken on embedded targets.
        try:
            from continuonbrain.services.manual_trainer import ManualTrainer, ManualTrainerRequest  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": "jax_unavailable",
                "message": f"Manual JAX training is unavailable on this device ({exc}).",
                "hint": "Install/repair JAX + tensorstore wheels, or run training elsewhere.",
            }

        payload = payload or {}
        if self.manual_trainer is None:
            self.manual_trainer = ManualTrainer()

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
        try:
            from continuonbrain.services.wavecore_trainer import WavecoreTrainer  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": "jax_unavailable",
                "message": f"WaveCore loops require JAX and are unavailable on this device ({exc}).",
                "hint": "Install/repair JAX + tensorstore wheels, or run WaveCore loops on a machine with working JAX.",
            }

        payload = payload or {}
        if self.wavecore_trainer is None:
            self.wavecore_trainer = WavecoreTrainer()
        # Ensure service is available to downstream eval runner
        payload.setdefault("service", self)
        return await self.wavecore_trainer.run_loops(payload)

    async def RunToolRouterTrain(self, payload: Optional[dict] = None) -> dict:
        """Train a lightweight JAX tool-router model from imported toolchat_hf_* RLDS episodes."""
        try:
            from continuonbrain.services.tool_router_trainer import ToolRouterTrainer, ToolRouterTrainRequest  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": "jax_unavailable",
                "message": f"Tool-router training requires JAX and is unavailable on this device ({exc}).",
                "hint": "Install/repair JAX + tensorstore wheels, or run tool-router training elsewhere.",
            }

        payload = payload or {}
        if self.tool_router_trainer is None:
            self.tool_router_trainer = ToolRouterTrainer()
        req = ToolRouterTrainRequest(
            episodes_root=Path(payload["episodes_root"]) if payload.get("episodes_root") else None,
            include_dirs_prefix=str(payload.get("include_dirs_prefix", "toolchat_hf")),
            max_episodes_scan=int(payload.get("max_episodes_scan", 20000)),
            top_k_tools=int(payload.get("top_k_tools", 128)),
            features_dim=int(payload.get("features_dim", 4096)),
            batch_size=int(payload.get("batch_size", 64)),
            max_steps=int(payload.get("max_steps", 600)),
            learning_rate=float(payload.get("learning_rate", 3e-3)),
            seed=int(payload.get("seed", 0)),
        )
        return await self.tool_router_trainer.run(req)

    async def ToolRouterPredict(self, payload: Optional[dict] = None) -> dict:
        """Suggest top-k tool names for a prompt using the latest exported tool-router bundle."""
        try:
            from continuonbrain.jax_models.infer.tool_router_infer import (  # noqa: WPS433
                load_tool_router_bundle,
                predict_topk,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": "jax_unavailable",
                "message": f"Tool-router inference requires JAX and is unavailable on this device ({exc}).",
                "hint": "Install/repair JAX + tensorstore wheels, or disable tool-router endpoints.",
            }

        payload = payload or {}
        prompt = str(payload.get("prompt") or "")
        k = int(payload.get("k") or 5)
        export_dir = Path(payload.get("export_dir") or "/opt/continuonos/brain/model/adapters/candidate/tool_router_seed")

        # Lazy-load; keep cached for repeated calls.
        if self._tool_router_bundle is None or getattr(self._tool_router_bundle, "manifest_path", None) != (export_dir / "tool_router_manifest.json"):
            self._tool_router_bundle = load_tool_router_bundle(export_dir)
        preds = predict_topk(self._tool_router_bundle, prompt, k=k)
        return {"status": "ok", "export_dir": str(export_dir), "k": k, "predictions": preds}

    async def RunToolRouterEval(self, payload: Optional[dict] = None) -> dict:
        """Run heldout eval for tool-router (top1/top5 on deterministic split)."""
        try:
            from continuonbrain.services.tool_router_evaluator import ToolRouterEvaluator, ToolRouterEvalRequest  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": "jax_unavailable",
                "message": f"Tool-router eval requires JAX and is unavailable on this device ({exc}).",
                "hint": "Install/repair JAX + tensorstore wheels, or run tool-router eval elsewhere.",
            }

        payload = payload or {}
        if self.tool_router_evaluator is None:
            self.tool_router_evaluator = ToolRouterEvaluator()
        req = ToolRouterEvalRequest(
            episodes_root=Path(payload["episodes_root"]) if payload.get("episodes_root") else None,
            include_dirs_prefix=str(payload.get("include_dirs_prefix", "toolchat_hf")),
            export_dir=Path(payload["export_dir"]) if payload.get("export_dir") else None,
            max_episodes_scan=int(payload.get("max_episodes_scan", 30000)),
            eval_mod=int(payload.get("eval_mod", 10)),
            eval_bucket=int(payload.get("eval_bucket", 0)),
            k=int(payload.get("k", 5)),
        )
        return await self.tool_router_evaluator.run(req)

    async def SpeakText(self, payload: Optional[dict] = None) -> dict:
        """Robot-side TTS (offline-first)."""
        payload = payload or {}
        text = str(payload.get("text", "") or payload.get("message", ""))
        voice = str(payload.get("voice", "en") or "en")
        rate_wpm = payload.get("rate_wpm", 175)
        return speak_text(text, voice=voice, rate_wpm=rate_wpm)

    async def RecordMicrophone(self, payload: Optional[dict] = None) -> dict:
        """Robot-side microphone capture (offline-first)."""
        payload = payload or {}
        seconds = payload.get("seconds", 4)
        sample_rate_hz = payload.get("sample_rate_hz", 16000)
        num_channels = payload.get("num_channels", 1)
        device = payload.get("device")
        res, status = record_wav(seconds=seconds, sample_rate_hz=sample_rate_hz, num_channels=num_channels, device=device)
        if not res:
            return status
        return {
            "status": "ok",
            "path": str(res.path),
            "sample_rate_hz": res.sample_rate_hz,
            "num_channels": res.num_channels,
            "duration_s": res.duration_s,
            "backend": res.backend,
        }

    async def ListAudioDevices(self) -> dict:
        """List ALSA capture devices (best-effort)."""
        out = {"status": "ok", "arecord_l": "", "arecord_L": ""}
        try:
            proc = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=3)
            out["arecord_l"] = (proc.stdout or proc.stderr or "").strip()
        except Exception as exc:  # noqa: BLE001
            out["arecord_l"] = f"error: {exc}"
        try:
            proc = subprocess.run(["arecord", "-L"], capture_output=True, text=True, timeout=3)
            out["arecord_L"] = (proc.stdout or proc.stderr or "").strip()
        except Exception as exc:  # noqa: BLE001
            out["arecord_L"] = f"error: {exc}"
        return out

    async def StartPairing(self, payload: Optional[dict] = None) -> dict:
        payload = payload or {}
        base_url = str(payload.get("base_url") or "").strip()
        ttl_s = int(payload.get("ttl_s") or 300)
        session = self.pairing.start(base_url=base_url, ttl_s=ttl_s)
        return {
            "status": "ok",
            "token": session.token,
            "confirm_code": session.confirm_code,
            "expires_unix_s": session.expires_unix_s,
            "url": session.url,
        }

    async def ConfirmPairing(self, payload: Optional[dict] = None) -> dict:
        payload = payload or {}
        ok, msg, ownership = self.pairing.confirm(
            token=str(payload.get("token") or ""),
            confirm_code=str(payload.get("confirm_code") or payload.get("code") or ""),
            owner_id=str(payload.get("owner_id") or payload.get("owner") or ""),
        )
        if not ok:
            return {"status": "error", "message": msg}
        return {"status": "ok", "ownership": ownership}

    async def GetOwnershipStatus(self) -> dict:
        # Back-compat: older clients expect flat keys (owned/subscription_active/seed_installed/etc).
        ownership = self.pairing.ownership_status()
        flat = {
            "owned": bool(ownership.get("owned", False)),
            "owner_id": ownership.get("owner_id"),
            "account_type": ownership.get("account_type"),
            "paired_unix_s": ownership.get("paired_unix_s"),
            # Default placeholders (local pairing doesn't set these yet)
            "subscription_active": False,
            "seed_installed": False,
            "account_id": None,
        }
        return {"status": "ok", "ownership": ownership, **flat}

    async def GetArchitectureStatus(self) -> dict:
        """Report which learning subsystems are active so we can verify 'whole architecture' participation."""
        status = await self.GetRobotStatus()
        block = status.get("status", {}) if isinstance(status, dict) else {}
        mode = (block.get("mode") or "unknown") if isinstance(block, dict) else "unknown"
        chat_settings = {}
        agent_mgr_settings = {}
        try:
            from continuonbrain.settings_manager import SettingsStore

            settings = SettingsStore(Path(self.config_dir)).load()
            chat_settings = (settings or {}).get("chat") or {}
            agent_mgr_settings = (settings or {}).get("agent_manager") or {}
        except Exception:
            pass
        learner_status = None
        try:
            if self.background_learner is not None:
                learner_status = self.background_learner.get_status()
        except Exception:
            learner_status = {"enabled": True, "running": getattr(self.background_learner, "running", None)}
        res = None
        try:
            res = self.resource_monitor.check_resources().to_dict()
        except Exception:
            res = None
        thread_meta = {}
        try:
            def _tmeta(t: Optional[threading.Thread]) -> dict:
                if t is None:
                    return {"present": False}
                return {"present": True, "alive": t.is_alive(), "name": t.name}

            thread_meta = {
                "chat_learn_thread": _tmeta(self._chat_learn_thread),
                "autonomous_learner_thread": _tmeta(self._autonomous_learner_thread),
                "orchestrator_thread": _tmeta(self._orchestrator_thread),
            }
        except Exception:
            thread_meta = {}

        # Hailo status (best-effort, never fatal). Hailo is used as an accelerator
        # for vision/core-model inference, not as a text model id.
        hailo_state = None
        try:
            if getattr(self, "chat_adapter", None) is not None:
                hailo_state = self.chat_adapter.get_hailo_state()
        except Exception:
            hailo_state = None
        hailo_hw = None
        try:
            hef_path = "/opt/continuonos/brain/model/base_model/model.hef"
            hailo_hw = {
                "dev_nodes": [p.name for p in Path("/dev").glob("hailo*")],
                "hef_present": Path(hef_path).exists(),
                "hef_path": hef_path,
            }
        except Exception:
            hailo_hw = None

        return {
            "status": "ok",
            "mode": mode,
            "recording": bool(self.is_recording),
            "chat_rlds_enabled": bool(chat_settings.get("log_rlds", False)),
            "chat_learn": (agent_mgr_settings.get("chat_learn") or {}) if isinstance(agent_mgr_settings, dict) else {},
            "autonomy_orchestrator": (agent_mgr_settings.get("autonomy_orchestrator") or {}) if isinstance(agent_mgr_settings, dict) else {},
            "hope_brain_loaded": bool(self.hope_brain is not None),
            "background_learner": learner_status,
            "last_autonomous_learner_action": self._last_autonomous_learner_action,
            "orchestrator_state": self._last_orchestrator,
            "tasks": thread_meta,
            "resources": res,
            "hailo": {"vision": hailo_state, "hardware": hailo_hw},
            "wavecore_metrics": {
                "fast": "/opt/continuonos/brain/trainer/logs/wavecore_fast_metrics.json",
                "mid": "/opt/continuonos/brain/trainer/logs/wavecore_mid_metrics.json",
                "slow": "/opt/continuonos/brain/trainer/logs/wavecore_slow_metrics.json",
            },
            "tool_router": {
                "export_dir": "/opt/continuonos/brain/model/adapters/candidate/tool_router_seed",
                "metrics_path": "/opt/continuonos/brain/trainer/logs/tool_router_metrics.json",
                "eval_metrics_path": "/opt/continuonos/brain/trainer/logs/tool_router_eval_metrics.json",
            },
        }

    async def RunChatLearn(self, payload: Optional[dict] = None) -> dict:
        """Run a bounded multi-turn learning conversation using Gemma 3n and log via chat->RLDS when enabled."""
        payload = payload or {}
        turns = int(payload.get("turns", 10) or 10)
        turns = max(1, min(turns, 50))
        session_id = str(payload.get("session_id") or f"chat_learn_{int(time.time())}")
        model_hint = str(payload.get("model_hint") or "google/gemma-3n-2b")
        delegate_model_hint = payload.get("delegate_model_hint")
        topic = str(payload.get("topic") or "tool use + planning + safety")

        # Enhanced prompt for Agent Manager to Subagent conversations - CURIOUS about system
        message = (
            "We are training the HOPE Agent Manager through multi-agent conversations.\n"
            "You are the Agent Manager (primary orchestrator) with CURIOSITY about the system.\n"
            "\n"
            "CURIOSITY DIRECTIVE: Be curious about how the system works, what it can learn, and how to improve it.\n"
            "\n"
            "For each turn:\n"
            "1) As the Agent Manager, be CURIOUS about the system:\n"
            "   - What aspects of HOPE's architecture are most interesting or mysterious?\n"
            "   - What learning capabilities could be enhanced?\n"
            "   - How does the system actually work internally (CMS, WaveCore, symbolic search)?\n"
            "   - What patterns emerge from the training data?\n"
            "   - How can we make HOPE more helpful through better understanding?\n"
            "\n"
            "2) Formulate CURIOUS questions that explore:\n"
            "   - System internals: How does CMS compaction actually work? What triggers it?\n"
            "   - Learning mechanisms: How does WaveCore fast/mid/slow differ? What do they learn?\n"
            "   - Symbolic search: How does tool router map language to actions? Can it improve?\n"
            "   - Training data: What patterns exist in RLDS episodes? What's missing?\n"
            "   - Safety: How are safety policies enforced? Can they be more effective?\n"
            "   - Performance: What bottlenecks exist? How can we optimize?\n"
            "\n"
            "3) Consult the subagent (Gemma 3n) with your curious questions about system internals.\n"
            "\n"
            "4) As Agent Manager, synthesize the subagent's insights and decide:\n"
            "   - What did we learn about how the system works?\n"
            "   - How can HOPE learn from this understanding?\n"
            "   - What concrete improvements would this enable?\n"
            "\n"
            "5) Be CURIOUS about learning itself:\n"
            "   - How does continuous learning actually happen?\n"
            "   - What makes some learning episodes more valuable than others?\n"
            "   - How can we make the system more curious and exploratory?\n"
            "\n"
            "Example curious conversation flow:\n"
            "- Agent Manager: 'I'm curious: How does CMS compaction actually consolidate memories? "
            "What triggers it, and could we make it more adaptive based on memory pressure patterns?'\n"
            "- Subagent: 'CMS compaction uses energy transfer from episodic memory to long-term parameters. "
            "It's triggered every 300s, but could be adaptive based on memory usage trends...'\n"
            "- Agent Manager: 'Fascinating! So HOPE could learn to predict when compaction is needed "
            "and trigger it proactively. This would improve memory efficiency and stability...'\n"
            "\n"
            f"Topic focus: {topic}.\n"
            "Be GENUINELY CURIOUS about the system. Ask questions that explore how things work, "
            "what could be better, and how learning actually happens.\n"
            "End each turn with the required structured JSON line.\n"
        )
        history: list = []
        outputs = []
        
        # Chat-learn RLDS logging is opt-in (privacy). Respect settings/env.
        log_enabled = False
        try:
            from continuonbrain.settings_manager import SettingsStore

            settings = SettingsStore(Path(self.config_dir)).load()
            log_enabled = bool((settings.get("chat", {}) or {}).get("log_rlds", False))
        except Exception:
            log_enabled = False
        if os.environ.get("CONTINUON_LOG_CHAT_RLDS", "0").lower() in ("1", "true", "yes", "on"):
            log_enabled = True
        
        # Direct RLDS logging for chat learning (bypass chat adapter's logging).
        # Only active when explicitly enabled above.
        rlds_cfg = None
        log_chat_turn = None
        if log_enabled:
            from continuonbrain.rlds.chat_rlds_logger import ChatRldsLogConfig, log_chat_turn as _log_chat_turn

            log_chat_turn = _log_chat_turn
            rlds_cfg = ChatRldsLogConfig(
                episodes_dir=Path("/opt/continuonos/brain/rlds/episodes"),
                group_by_session=True,
            )
        
        for i in range(turns):
            resp = await self.ChatWithGemma(
                message,
                history,
                session_id=session_id,
                model_hint=model_hint,
                delegate_model_hint=delegate_model_hint,
            )
            outputs.append(resp)
            # Feed response back as next message to create a self-play style refinement loop.
            assistant_text = resp.get("response", "") if isinstance(resp, dict) else str(resp)
            structured_data = resp.get("structured", {}) if isinstance(resp, dict) else {}
            
            # Check if we got a fallback response (generic status message)
            is_fallback = (
                assistant_text and (
                    "Status snapshot" in assistant_text or
                    "Robot status:" in assistant_text or
                    "Ready for XR" in assistant_text or
                    (assistant_text.startswith("[model=") and len(assistant_text) < 200 and "Status" in assistant_text)
                )
            )
            
            if is_fallback:
                import logging
                logging.getLogger(__name__).warning(
                    f"Chat learning turn {i+1}: Got fallback response instead of actual conversation. "
                    f"Model may not be available. Skipping RLDS log for this turn."
                )
            else:
                # Directly log to RLDS for chat learning (ensures we capture actual conversation)
                if log_enabled and rlds_cfg is not None and log_chat_turn is not None:
                    try:
                        status_data = await self.GetRobotStatus()
                        status = status_data.get("status", {}) if isinstance(status_data, dict) else {}

                        log_chat_turn(
                            rlds_cfg,
                            user_message=message,
                            assistant_response=assistant_text,
                            structured=structured_data if isinstance(structured_data, dict) else {},
                            status_context=status,
                            session_id=session_id,
                            model_hint=model_hint,
                            agent_label=model_hint or "agent_manager",
                        )
                    except Exception as exc:  # noqa: BLE001
                        import logging
                        logging.getLogger(__name__).warning(f"Failed to log chat learning turn to RLDS: {exc}")
            
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": assistant_text})
            
            # Enhanced conversation flow for agent manager to subagent discussions
            if i < turns - 1:  # Not the last turn
                # Alternate between agent manager questions and subagent synthesis
                if i % 2 == 0:
                    # Agent Manager asks a new question about HOPE learning
                    message = (
                        f"As the Agent Manager, identify another area where HOPE should learn to be more helpful. "
                        f"Formulate a specific question about: {topic}. "
                        f"Turn {i+2}/{turns}."
                    )
                else:
                    # Agent Manager synthesizes subagent advice and proposes learning
                    message = (
                        f"As the Agent Manager, synthesize the subagent's previous advice. "
                        f"Explain how HOPE should learn from this and what concrete improvements it would enable. "
                        f"Turn {i+2}/{turns}."
                    )
            else:
                # Final turn: summarize key learnings
                message = (
                    f"Final turn: As the Agent Manager, summarize the top 3 things HOPE should learn "
                    f"based on our conversation, and explain how each would make HOPE more helpful. "
                    f"Turn {i+2}/{turns}."
                )
        return {
            "status": "ok",
            "session_id": session_id,
            "turns": turns,
            "model_hint": model_hint,
            "delegate_model_hint": delegate_model_hint,
            "rlds_logging": bool(log_enabled),
            "results": outputs[-3:],
        }

    def _chat_learn_loop(self) -> None:
        """Background periodic chat learning loop (offline-first)."""
        last_run = 0.0
        while not self._bg_stop_event.is_set():
            try:
                from continuonbrain.settings_manager import SettingsStore

                settings = SettingsStore(Path(self.config_dir)).load()
                chat = (settings or {}).get("chat") or {}
                agent_mgr = (settings or {}).get("agent_manager") or {}
                cfg = (agent_mgr.get("chat_learn") or {}) if isinstance(agent_mgr, dict) else {}
                enabled = bool(cfg.get("enabled", False)) and bool(chat.get("log_rlds", False))
                interval_s = int(cfg.get("interval_s", 600) or 600)
                turns = int(cfg.get("turns_per_cycle", 10) or 10)
                model_hint = str(cfg.get("model_hint") or "google/gemma-3n-2b")
                delegate_model_hint = cfg.get("delegate_model_hint")  # Support subagent conversations
                topic = str(cfg.get("topic") or "coding this repository")
                modes_allowed = cfg.get("modes") if isinstance(cfg, dict) else None
                if not isinstance(modes_allowed, list) or not modes_allowed:
                    modes_allowed = ["idle"]

                now = time.time()
                due = (now - last_run) >= float(max(30, interval_s))
                # Only run when robot is idle (and skip if recording is active).
                mode = "unknown"
                try:
                    if self.mode_manager is not None:
                        mode = self.mode_manager.current_mode.value
                except Exception:
                    mode = "unknown"

                mode_ok = str(mode).lower() in {str(m).lower() for m in modes_allowed}
                if enabled and due and mode_ok and not self.is_recording:
                    asyncio.run(
                        self.RunChatLearn(
                            {
                                "turns": turns,
                                "model_hint": model_hint,
                                "delegate_model_hint": delegate_model_hint,  # Enable subagent conversations
                                "topic": topic,
                                "session_id": f"chat_learn_sched_{int(time.time())}",
                            }
                        )
                    )
                    last_run = now
            except Exception:
                # Never crash the runtime due to the background learner.
                pass
            time.sleep(5)

    def _ensure_hope_brain(self) -> bool:
        """Best-effort HOPE brain initialization (torch/hope_impl are required)."""
        if self.hope_brain is not None:
            return True
        try:
            from continuonbrain.hope_impl.config import HOPEConfig
            from continuonbrain.hope_impl.brain import HOPEBrain

            cfg = HOPEConfig.pi5_optimized()
            # Keep dims modest; this is currently used for background curiosity learning + monitoring.
            self.hope_brain = HOPEBrain(config=cfg, obs_dim=10, action_dim=4, output_dim=4)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  HOPE brain init unavailable: {exc}")
            self.hope_brain = None
            return False

    def _autonomous_learning_loop(self) -> None:
        """Start/pause/stop HOPE BackgroundLearner based on robot mode + settings."""
        while not self._bg_stop_event.is_set():
            try:
                from continuonbrain.settings_manager import SettingsStore

                settings = SettingsStore(Path(self.config_dir)).load()
                agent_mgr = (settings or {}).get("agent_manager") or {}
                enable = bool(agent_mgr.get("enable_autonomous_learning", True))
                steps_per_cycle = int(agent_mgr.get("autonomous_learning_steps_per_cycle", 100) or 100)
                checkpoint_interval = int(agent_mgr.get("autonomous_learning_checkpoint_interval", 1000) or 1000)
                orch_cfg = (agent_mgr.get("autonomy_orchestrator") or {}) if isinstance(agent_mgr, dict) else {}
                headroom_mb = int(orch_cfg.get("min_memory_headroom_mb", 512) or 512)

                mode = "unknown"
                try:
                    if self.mode_manager is not None:
                        mode = self.mode_manager.current_mode.value
                except Exception:
                    mode = "unknown"
                mode = str(mode).lower()

                want_running = enable and (mode == "autonomous") and (not self.is_recording)
                res_status = self.resource_monitor.check_resources() if self.resource_monitor else None
                low_headroom = False
                if res_status and self.resource_monitor:
                    reserve_plus_headroom = self.resource_monitor.limits.system_reserve_mb + headroom_mb
                    low_headroom = res_status.available_memory_mb < reserve_plus_headroom

                if want_running and low_headroom:
                    if self.background_learner is not None and self.background_learner.running and not self.background_learner.paused:
                        self.background_learner.pause()
                    self._last_autonomous_learner_action = {
                        "action": "pause_low_memory",
                        "ts": time.time(),
                        "available_mb": res_status.available_memory_mb if res_status else None,
                        "headroom_mb": headroom_mb,
                    }
                    time.sleep(5)
                    continue

                if want_running:
                    ok = self._ensure_hope_brain()
                    if ok and self.background_learner is None and BackgroundLearner is not None:
                        # Enable RLDS-ish logging under config_dir (doesn't affect WaveCore datasets).
                        self.background_learner = BackgroundLearner(
                            brain=self.hope_brain,
                            config={
                                "steps_per_cycle": steps_per_cycle,
                                "cycle_interval_sec": 0.5,
                                "checkpoint_interval": checkpoint_interval,
                                "checkpoint_dir": f"{self.config_dir}/checkpoints/autonomous",
                                "rlds_log_dir": f"{self.config_dir}/rlds/autonomous_learning",
                            },
                            resource_monitor=self.resource_monitor,
                        )
                        self.background_learner.start()
                        self._last_autonomous_learner_action = {"action": "start", "ts": time.time()}
                    elif self.background_learner is not None and self.background_learner.paused:
                        self.background_learner.resume()
                        self._last_autonomous_learner_action = {"action": "resume", "ts": time.time()}

                    # Stability guard: if HOPE is unstable, reduce learning rate and compact CMS.
                    try:
                        if self.hope_brain is not None and not self.hope_brain.stability_monitor.is_stable():
                            state = self.hope_brain.get_state()
                            cur_eta = float(getattr(state.params, "eta", 0.05))
                            new_eta = max(0.005, min(cur_eta * 0.5, cur_eta))
                            state.params.eta = new_eta
                            self.hope_brain.set_state(state)
                            try:
                                self.hope_brain.compact_memory()
                            except Exception:
                                pass
                            self._last_autonomous_learner_action = {"action": "stability_guard", "ts": time.time(), "eta": new_eta}
                    except Exception:
                        pass
                else:
                    if self.background_learner is not None and self.background_learner.running and not self.background_learner.paused:
                        self.background_learner.pause()
                        self._last_autonomous_learner_action = {"action": "pause", "ts": time.time(), "mode": mode}
            except Exception:
                pass
            time.sleep(5)

    def _autonomy_orchestrator_loop(self) -> None:
        """Resource-aware orchestrator to ensure the whole architecture participates without breaking constraints."""
        last: dict = {
            "cms": 0.0,
            "hope_eval": 0.0,
            "facts_eval": 0.0,
            "wavecore": 0.0,
            "tool_router": 0.0,
        }
        while not self._bg_stop_event.is_set():
            try:
                from continuonbrain.settings_manager import SettingsStore

                settings = SettingsStore(Path(self.config_dir)).load()
                chat = (settings or {}).get("chat") or {}
                agent_mgr = (settings or {}).get("agent_manager") or {}
                orch = (agent_mgr.get("autonomy_orchestrator") or {}) if isinstance(agent_mgr, dict) else {}
                enabled = bool(orch.get("enabled", False))
                modes_allowed = orch.get("modes") if isinstance(orch, dict) else None
                if not isinstance(modes_allowed, list) or not modes_allowed:
                    modes_allowed = ["autonomous"]

                # Determine current mode (avoid calling GetRobotStatus from background tasks).
                mode = "unknown"
                try:
                    if self.mode_manager is not None:
                        mode = self.mode_manager.current_mode.value
                except Exception:
                    mode = "unknown"
                mode = str(mode).lower()
                mode_ok = mode in {str(m).lower() for m in modes_allowed}

                # Enforce min interval guard
                min_interval_s = int(orch.get("min_interval_s", 30) or 30)
                now = time.time()
                if (now - float(self._last_orchestrator.get("last_run_ts", 0.0))) < float(min_interval_s):
                    time.sleep(5)
                    continue

                # Only orchestrate when allowed, and never during recording.
                if (not enabled) or (not mode_ok) or self.is_recording:
                    time.sleep(5)
                    continue

                # Resource gate
                res = self.resource_monitor.check_resources()
                headroom_mb = int(orch.get("min_memory_headroom_mb", 512) or 512)
                reserve_plus_headroom = self.resource_monitor.limits.system_reserve_mb + headroom_mb
                if res.available_memory_mb < reserve_plus_headroom:
                    if self.background_learner is not None and getattr(self.background_learner, "running", False) and not getattr(self.background_learner, "paused", False):
                        self.background_learner.pause()
                    self._last_orchestrator = {
                        "last_run_ts": now,
                        "skipped": "low_memory_headroom",
                        "resource": res.to_dict(),
                        "headroom_mb": headroom_mb,
                    }
                    time.sleep(5)
                    continue
                if res.level in (ResourceLevel.EMERGENCY,):
                    self._last_orchestrator = {"last_run_ts": now, "skipped": "resource_emergency", "resource": res.to_dict()}
                    time.sleep(10)
                    continue

                # Pause continuous learner during heavy jobs to avoid thrashing.
                def _pause_bg():
                    try:
                        if self.background_learner is not None and getattr(self.background_learner, "running", False) and not getattr(self.background_learner, "paused", False):
                            self.background_learner.pause()
                    except Exception:
                        pass

                def _resume_bg():
                    try:
                        if self.background_learner is not None and getattr(self.background_learner, "running", False) and getattr(self.background_learner, "paused", False):
                            self.background_learner.resume()
                    except Exception:
                        pass

                with self._orchestrator_lock:
                    actions = {}
                    # IMPORTANT: JAX/tensorstore wheels are often broken/mismatched on Pi images.
                    # Importing them can hard-crash the process (abort), which cannot be caught.
                    # Therefore, we only attempt JAX-based tasks when explicitly enabled.
                    jax_tasks_ok = os.environ.get("CONTINUON_ENABLE_JAX_TASKS", "0").lower() in ("1", "true", "yes", "on")

                    # 1) HOPE CMS compaction (cheap, helps stability/memory).
                    cms_every = int(orch.get("cms_compact_every_s", 600) or 600)
                    if self.hope_brain is not None and (now - last["cms"] >= cms_every):
                        try:
                            _pause_bg()
                            result = self.hope_brain.compact_memory()
                            actions["cms_compact"] = {"ok": True, "result": result, "timestamp": now}
                            import logging
                            logging.getLogger(__name__).info(f"CMS compaction completed: {result}")
                        except Exception as exc:  # noqa: BLE001
                            actions["cms_compact"] = {"ok": False, "error": str(exc), "timestamp": now}
                            import logging
                            logging.getLogger(__name__).warning(f"CMS compaction failed: {exc}")
                        finally:
                            _resume_bg()
                        last["cms"] = now

                    # 2) Evals (bounded; offline-first).
                    hope_every = int(orch.get("hope_eval_every_s", 1800) or 1800)
                    if (now - last["hope_eval"] >= hope_every):
                        try:
                            _pause_bg()
                            asyncio.run(self.RunHopeEval({"rlds_dir": "/opt/continuonos/brain/rlds/episodes", "use_fallback": True}))
                            actions["hope_eval"] = {"ok": True}
                        except Exception as exc:  # noqa: BLE001
                            actions["hope_eval"] = {"ok": False, "error": str(exc)}
                        finally:
                            _resume_bg()
                        last["hope_eval"] = now

                    facts_every = int(orch.get("facts_eval_every_s", 3600) or 3600)
                    if (now - last["facts_eval"] >= facts_every):
                        try:
                            _pause_bg()
                            asyncio.run(self.RunFactsEval({"rlds_dir": "/opt/continuonos/brain/rlds/episodes", "use_fallback": True}))
                            actions["facts_eval"] = {"ok": True}
                        except Exception as exc:  # noqa: BLE001
                            actions["facts_eval"] = {"ok": False, "error": str(exc)}
                        finally:
                            _resume_bg()
                        last["facts_eval"] = now

                    # 3) WaveCore (SSM-ish/JAX seed loops): only if resources are OK.
                    wave_every = int(orch.get("wavecore_every_s", 1800) or 1800)
                    if (not jax_tasks_ok) and (now - last["wavecore"] >= wave_every):
                        actions["wavecore"] = {"ok": False, "skipped": "jax_tasks_disabled"}
                        last["wavecore"] = now
                    elif res.level not in (ResourceLevel.CRITICAL,) and (now - last["wavecore"] >= wave_every):
                        try:
                            _pause_bg()
                            fast_steps = int(orch.get("wavecore_steps_fast", 60) or 60)
                            mid_steps = int(orch.get("wavecore_steps_mid", 120) or 120)
                            slow_steps = int(orch.get("wavecore_steps_slow", 180) or 180)
                            # If RLDS is mostly eval-only, fall back to synthetic so we still train the SSM/loop machinery.
                            use_synth = not bool(chat.get("log_rlds", False))  # weak heuristic; keep cheap
                            asyncio.run(
                                self.RunWavecoreLoops(
                                    {
                                        "fast": {"arch_preset": "pi5", "max_steps": fast_steps, "batch_size": 8, "learning_rate": 1e-3, "disable_jit": True, "use_synthetic": use_synth},
                                        "mid": {"arch_preset": "pi5", "max_steps": mid_steps, "batch_size": 8, "learning_rate": 5e-4, "disable_jit": True, "use_synthetic": use_synth},
                                        "slow": {"arch_preset": "pi5", "max_steps": slow_steps, "batch_size": 8, "learning_rate": 2e-4, "disable_jit": True, "use_synthetic": use_synth},
                                        "compact_export": True,
                                        "run_hope_eval": False,
                                        "run_facts_eval": False,
                                    }
                                )
                            )
                            actions["wavecore"] = {"ok": True, "use_synthetic": use_synth}
                        except Exception as exc:  # noqa: BLE001
                            actions["wavecore"] = {"ok": False, "error": str(exc)}
                        finally:
                            _resume_bg()
                        last["wavecore"] = now

                    # 4) Tool-router refresh (heavy).
                    tool_every = int(orch.get("tool_router_every_s", 3600) or 3600)
                    if (not jax_tasks_ok) and (now - last["tool_router"] >= tool_every):
                        actions["tool_router"] = {"ok": False, "skipped": "jax_tasks_disabled"}
                        last["tool_router"] = now
                    elif res.level == ResourceLevel.NORMAL and (now - last["tool_router"] >= tool_every):
                        try:
                            _pause_bg()
                            steps = int(orch.get("tool_router_steps", 200) or 200)
                            asyncio.run(self.RunToolRouterTrain({"max_steps": steps, "batch_size": 64, "learning_rate": 3e-3, "max_episodes_scan": 20000, "top_k_tools": 128, "include_dirs_prefix": "toolchat_hf"}))
                            asyncio.run(self.RunToolRouterEval({"eval_mod": 10, "eval_bucket": 0, "k": 5, "max_episodes_scan": 20000, "include_dirs_prefix": "toolchat_hf"}))
                            actions["tool_router"] = {"ok": True, "steps": steps}
                        except Exception as exc:  # noqa: BLE001
                            actions["tool_router"] = {"ok": False, "error": str(exc)}
                        finally:
                            _resume_bg()
                        last["tool_router"] = now

                    self._last_orchestrator = {
                        "last_run_ts": now,
                        "mode": mode,
                        "resource": res.to_dict(),
                        "last_actions": actions,
                        "last_markers": last,
                        "last_cms_compact": last.get("cms", 0.0),
                        "last_hope_eval": last.get("hope_eval", 0.0),
                        "last_wavecore": last.get("wavecore", 0.0),
                    }
            except Exception as exc:  # noqa: BLE001
                # Never crash the runtime due to orchestrator issues, but surface the error for visibility.
                try:
                    self._last_orchestrator = {"last_run_ts": time.time(), "error": str(exc)}
                except Exception:
                    pass
            time.sleep(5)


    async def RunHopeEval(self, payload: Optional[dict] = None) -> dict:
        """Run graded HOPE Q&A, log RLDS episode, with fallback LLM ordering."""
        payload = payload or {}
        questions_path = Path(payload.get("questions_path") or (REPO_ROOT / "continuonbrain" / "eval" / "hope_eval_questions.json"))
        rlds_dir = Path(payload.get("rlds_dir") or "/opt/continuonos/brain/rlds/episodes")
        use_fallback = bool(payload.get("use_fallback", True))
        # Hailo is an accelerator, not a text model id.
        fallback_order = payload.get("fallback_order") or ["google/gemma-370m", "google/gemma-3n-2b"]
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
        fallback_order = payload.get("fallback_order") or ["google/gemma-370m", "google/gemma-3n-2b"]
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
                # If motion is skipped, keep camera-only real mode when available.
                if self.skip_motion_hw and self.camera is not None:
                    print("  Motion HW missing, but camera is available; keeping camera in REAL mode (skip-motion-hw).")
                    hardware_ready = True
                else:
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
        # Load persisted mode (set by startup manager) or default to idle
        persisted_mode = self.mode_manager.load_state()
        if persisted_mode and persisted_mode != RobotMode.IDLE:
            # Restore the persisted mode (e.g., AUTONOMOUS set by startup manager)
            self.mode_manager.set_mode(persisted_mode, metadata={"restored_from_persisted": True})
        else:
            # No persisted mode, start in idle
            self.mode_manager.return_to_idle()
        print(" Mode manager ready")

        # Start lightweight control-loop telemetry monitor (safe even in mock mode).
        self._start_control_loop_monitor()

        # Start background chat-learn scheduler (thread; controlled via settings).
        disable_chat_learn = os.environ.get("CONTINUON_DISABLE_CHAT_LEARN", "0").lower() in ("1", "true", "yes", "on")
        if disable_chat_learn:
            print(" Chat-learn scheduler disabled via CONTINUON_DISABLE_CHAT_LEARN=1")
        elif self._chat_learn_thread is None or not self._chat_learn_thread.is_alive():
            try:
                self._chat_learn_thread = threading.Thread(target=self._chat_learn_loop, daemon=True)
                self._chat_learn_thread.start()
                print(" Chat-learn scheduler started (thread; settings-gated)")
            except Exception as exc:  # noqa: BLE001
                print(f"  Chat-learn scheduler failed to start: {exc}")

        disable_autonomous_learner = os.environ.get("CONTINUON_DISABLE_AUTONOMOUS_LEARNER", "0").lower() in ("1", "true", "yes", "on")
        if disable_autonomous_learner:
            print(" HOPE autonomous learner supervisor disabled via CONTINUON_DISABLE_AUTONOMOUS_LEARNER=1")
        elif self._autonomous_learner_thread is None or not self._autonomous_learner_thread.is_alive():
            try:
                self._autonomous_learner_thread = threading.Thread(target=self._autonomous_learning_loop, daemon=True)
                self._autonomous_learner_thread.start()
                print(" HOPE autonomous learner supervisor started (thread; mode-gated)")
            except Exception as exc:  # noqa: BLE001
                print(f"  HOPE autonomous learner supervisor failed to start: {exc}")

        disable_orchestrator = os.environ.get("CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR", "0").lower() in ("1", "true", "yes", "on")
        if disable_orchestrator:
            print(" Autonomy orchestrator disabled via CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR=1")
        elif self._orchestrator_thread is None or not self._orchestrator_thread.is_alive():
            try:
                self._orchestrator_thread = threading.Thread(target=self._autonomy_orchestrator_loop, daemon=True)
                self._orchestrator_thread.start()
                print(" Autonomy orchestrator started (thread; resource-aware)")
            except Exception as exc:  # noqa: BLE001
                print(f"  Autonomy orchestrator failed to start: {exc}")

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

                # Safety check: Validate action against safety protocol
                action_description = f"Arm control command: {control_mode} from {client_id}"
                if hasattr(self, '_check_safety_protocol'):
                    is_safe, safety_reason = self._check_safety_protocol(
                        action_description,
                        context={
                            "current_mode": self.mode_manager.current_mode.value if self.mode_manager else "unknown",
                            "client_id": client_id,
                            "control_mode": control_mode,
                        }
                    )
                    if not is_safe:
                        return {
                            "success": False,
                            "message": f"Safety check failed: {safety_reason}",
                            "safety_blocked": True,
                        }

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

            # Safety check: Validate action against safety protocol
            action_description = f"Drive with steering={steering_value:.2f}, throttle={throttle_value:.2f}"
            if hasattr(self, '_check_safety_protocol'):
                is_safe, safety_reason = self._check_safety_protocol(
                    action_description,
                    context={
                        "current_mode": current_mode.value,
                        "requires_human_approval": abs(throttle_value) > 0.8,  # High throttle requires approval
                        "human_approved": context.get("human_approved", False) if hasattr(self, 'context') else False,
                    }
                )
                if not is_safe:
                    return _record_result({
                        "success": False,
                        "message": f"Safety check failed: {safety_reason}",
                        "safety_blocked": True,
                    })

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
                status["control_loop"] = self._control_loop_summary()
            
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

    async def GetControlLoopMetrics(self, payload: Optional[dict] = None) -> dict:
        """Expose control-loop telemetry (period/work p50/p95/p99 + recent points)."""
        payload = payload or {}
        try:
            limit = int(payload.get("limit") or 180)
        except Exception:
            limit = 180
        limit = max(20, min(600, limit))

        period_points = list(self._control_loop_period_ms)[-limit:]
        work_points = list(self._control_loop_work_ms)[-limit:]

        return {
            "status": "ok",
            "target_period_ms": self._control_loop_target_ms,
            "target_allow_motion_s": self._control_loop_target_allow_motion_s,
            "target_idle_s": self._control_loop_target_idle_s,
            "samples": self._control_loop_samples,
            "deadline_misses": self._control_loop_deadline_misses,
            "period_ms": {
                "points": period_points,
                "p50": _percentile(period_points, 0.50),
                "p95": _percentile(period_points, 0.95),
                "p99": _percentile(period_points, 0.99),
            },
            "work_ms": {
                "points": work_points,
                "p50": _percentile(work_points, 0.50),
                "p95": _percentile(work_points, 0.95),
                "p99": _percentile(work_points, 0.99),
            },
        }

    def _start_control_loop_monitor(self) -> None:
        if self._control_loop_thread is not None and self._control_loop_thread.is_alive():
            return
        self._control_loop_stop.clear()
        self._control_loop_thread = threading.Thread(
            target=self._control_loop_monitor_loop,
            name="control_loop_monitor",
            daemon=True,
        )
        self._control_loop_thread.start()

    def _control_loop_monitor_loop(self) -> None:
        """A lightweight periodic loop that measures tick cadence + work time."""
        prev_start = None
        next_deadline = time.perf_counter()
        while not self._control_loop_stop.is_set():
            # Determine desired tick cadence based on mode gates.
            allow_motion = False
            try:
                if self.mode_manager is not None:
                    allow_motion = bool(self.mode_manager.get_mode_config(self.mode_manager.current_mode).allow_motion)
            except Exception:
                allow_motion = False
            period_s = self._control_loop_target_allow_motion_s if allow_motion else self._control_loop_target_idle_s
            period_s = max(0.01, min(2.0, float(period_s)))

            # Sleep until next tick (drift-resistant).
            now = time.perf_counter()
            sleep_s = next_deadline - now
            if sleep_s > 0:
                time.sleep(sleep_s)

            start = time.perf_counter()
            if prev_start is not None:
                period_ms = (start - prev_start) * 1000.0
                self._control_loop_period_ms.append(period_ms)
                if period_ms > float(self._control_loop_target_ms):
                    self._control_loop_deadline_misses += 1
            prev_start = start

            # Minimal tick work (safe in mock mode; never blocks on network).
            try:
                if self.health_checker is not None:
                    _ = self.health_checker.get_safety_head_status()
            except Exception:
                pass

            end = time.perf_counter()
            self._control_loop_work_ms.append((end - start) * 1000.0)
            self._control_loop_samples += 1

            next_deadline = max(next_deadline + period_s, time.perf_counter() + period_s)

    def _control_loop_summary(self) -> dict:
        pts = list(self._control_loop_period_ms)
        return {
            "target_period_ms": self._control_loop_target_ms,
            "samples": self._control_loop_samples,
            "deadline_misses": self._control_loop_deadline_misses,
            "period_ms_p50": _percentile(pts, 0.50),
            "period_ms_p95": _percentile(pts, 0.95),
            "period_ms_p99": _percentile(pts, 0.99),
        }

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
    
    async def ChatWithGemma(
        self,
        message: str,
        history: list,
        session_id: str = None,
        *,
        model_hint: str = None,
        delegate_model_hint: str = None,
        image_jpeg: bytes = None,
        image_source: str = None,
        vision_requested: bool = False,
    ) -> dict:
        """
        Chat with Gemma 3n model acting as the Agent Manager.
        
        Args:
            message: User's message
            history: Chat history for context
            session_id: Optional session identifier for multi-turn context (best-effort)
            
        Returns:
            dict with 'response' or 'error'
        """
        return await self.chat_adapter.chat(
            message,
            history,
            model_hint=model_hint,
            session_id=session_id,
            delegate_model_hint=delegate_model_hint,
            image_jpeg=image_jpeg,
            image_source=image_source,
            vision_requested=vision_requested,
        )

    async def PlanArmSearch(self, payload: Optional[dict] = None) -> dict:
        """
        Run an arm-focused planner (beam search) using the Mamba world model interface.

        Safety notes:
        - This endpoint returns a plan + diagnostics by default.
        - It will only execute the first step if `execute=true` AND motion gate allows motion.
        """
        payload = payload or {}
        try:
            # Pull current status (includes motion gate + joint positions if available).
            status_resp = await self.GetRobotStatus()
            status = status_resp.get("status", {}) if isinstance(status_resp, dict) else {}
            allow_motion = bool(status.get("allow_motion", False))
            joints = status.get("joint_positions")
            simulated_state = False

            # In mock-hardware mode we may not have a real arm controller but still want to
            # exercise the planner stack end-to-end (world model + search + RLDS traces).
            if (not isinstance(joints, list) or len(joints) != 6) and status.get("hardware_mode") == "mock":
                joints = [0.0] * 6
                simulated_state = True

            if not isinstance(joints, list) or len(joints) != 6:
                return {"success": False, "message": "Joint positions unavailable; cannot plan", "plan": None}

            # Parse goal from payload.
            goal_joints = payload.get("goal_joint_pos")
            if not isinstance(goal_joints, list) or len(goal_joints) != 6:
                return {"success": False, "message": "goal_joint_pos (len=6) required", "plan": None}

            execute = bool(payload.get("execute", False))
            horizon = int(payload.get("horizon", 6))
            beam_width = int(payload.get("beam_width", 6))
            action_step = float(payload.get("action_step", 0.05))
            time_budget_ms = int(payload.get("time_budget_ms", 150))

            from continuonbrain.mamba_brain import build_world_model
            from continuonbrain.reasoning.arm_state_codec import ArmGoal, state_from_joints
            from continuonbrain.reasoning.tree_search import beam_search_plan
            from pathlib import Path
            from continuonbrain.rlds.planning_trace import write_planning_episode

            wm = build_world_model(prefer_mamba=True, joint_dim=6)
            start_state = state_from_joints(joints)
            goal = ArmGoal(target_joint_pos=[float(x) for x in goal_joints])

            plan = beam_search_plan(
                world_model=wm,
                start_state=start_state,
                goal=goal,
                horizon=horizon,
                beam_width=beam_width,
                action_step=action_step,
                time_budget_ms=time_budget_ms,
            )

            # Optionally execute the first step (single-step only; caller can loop).
            executed = False
            exec_result = None
            if execute:
                if not allow_motion:
                    exec_result = {"success": False, "message": "Motion gate locked; refusing to execute"}
                elif not plan.ok or not plan.steps:
                    exec_result = {"success": False, "message": "No plan to execute"}
                elif self.arm is None:
                    exec_result = {"success": False, "message": "Arm controller unavailable"}
                else:
                    from continuonbrain.reasoning.arm_state_codec import apply_action_to_arm

                    executed = apply_action_to_arm(self.arm, joints, plan.steps[0].action)
                    exec_result = {"success": bool(executed), "step_index": 0}

            # RLDS trace: log the full plan + diagnostics as a canonical episode_dir.
            try:
                rlds_root = Path(payload.get("rlds_dir") or "/opt/continuonos/brain/rlds/episodes")
                plan_payload = {
                    "ok": plan.ok,
                    "reason": plan.reason,
                    "diagnostics": plan.diagnostics,
                    "steps": [
                        {
                            "joint_delta": step.action.joint_delta,
                            "predicted_joint_pos": step.predicted_state.joint_pos,
                            "score": step.score,
                            "uncertainty": step.uncertainty,
                        }
                        for step in plan.steps
                    ],
                }
                exec_payload = {"requested": execute, "executed": executed, "result": exec_result}
                ep_dir = write_planning_episode(
                    rlds_root=rlds_root,
                    environment_id=str(status.get("hardware_mode", "pi5-dev")),
                    control_role="human_supervisor",
                    goal={"joint_pos": [float(x) for x in goal_joints]},
                    start_state={"joint_pos": [float(x) for x in joints]},
                    plan=plan_payload,
                    execute=exec_payload,
                    model_debug={"world_model_backend": getattr(getattr(plan.steps[0], "predicted_state", None), "debug", None) if plan.steps else None},
                    tags=["execute:true" if execute else "execute:false"],
                )
                rlds_episode_dir = str(ep_dir)
            except Exception:  # noqa: BLE001
                rlds_episode_dir = None

            # Return response.
            return {
                "success": True,
                "allow_motion": allow_motion,
                "simulated_state": simulated_state,
                "plan": {
                    "ok": plan.ok,
                    "reason": plan.reason,
                    "diagnostics": plan.diagnostics,
                    "steps": [
                        {
                            "joint_delta": step.action.joint_delta,
                            "predicted_joint_pos": step.predicted_state.joint_pos,
                            "score": step.score,
                            "uncertainty": step.uncertainty,
                        }
                        for step in plan.steps
                    ],
                },
                "execute": {"requested": execute, "executed": executed, "result": exec_result},
                "rlds_episode_dir": rlds_episode_dir,
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    async def ExecuteArmDelta(self, payload: Optional[dict] = None) -> dict:
        """
        Execute a single joint delta step (for step-by-step planner execution).
        Requires motion gate to be open.
        """
        payload = payload or {}
        try:
            status_resp = await self.GetRobotStatus()
            status = status_resp.get("status", {}) if isinstance(status_resp, dict) else {}
            allow_motion = bool(status.get("allow_motion", False))
            joints = status.get("joint_positions")
            if not allow_motion:
                return {"success": False, "message": "Motion gate locked; refusing to execute", "allow_motion": False}
            if self.arm is None:
                return {"success": False, "message": "Arm controller unavailable", "allow_motion": allow_motion}
            if not isinstance(joints, list) or len(joints) != 6:
                return {"success": False, "message": "Joint positions unavailable", "allow_motion": allow_motion}

            joint_delta = payload.get("joint_delta")
            if not isinstance(joint_delta, list) or len(joint_delta) != 6:
                return {"success": False, "message": "joint_delta (len=6) required", "allow_motion": allow_motion}

            from continuonbrain.reasoning.arm_state_codec import apply_action_to_arm, action_from_delta

            action = action_from_delta([float(x) for x in joint_delta])
            ok = apply_action_to_arm(self.arm, joints, action)
            return {"success": bool(ok), "allow_motion": allow_motion}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "message": str(exc)}
    
    def shutdown(self):
        """Graceful shutdown."""
        print("Shutting down Robot Service...")

        try:
            self._bg_stop_event.set()
        except Exception:
            pass
        try:
            self._control_loop_stop.set()
        except Exception:
            pass
        try:
            if self.background_learner:
                self.background_learner.stop()
        except Exception:
            pass
        
        if self.is_recording and self.recorder:
            self.recorder.end_episode(success=False)
        
        if self.recorder:
            self.recorder.shutdown()
        
        if self.camera:
            self.camera.stop()
        
        if self.arm:
            self.arm.shutdown()
        
        print("✅ Shutdown complete")


def _percentile(values: list, q: float) -> Optional[float]:
    """Simple nearest-rank percentile for small arrays (no numpy dependency)."""
    if not values:
        return None
    q = max(0.0, min(1.0, float(q)))
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    idx = int(round(q * (len(xs) - 1)))
    idx = max(0, min(len(xs) - 1, idx))
    return xs[idx]


def _parse_env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _parse_env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


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