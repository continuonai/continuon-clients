"""
BrainService: Corresponds to the "Brain" logic.
Manages tasks, hardware access, and robot state.
"""
import os
import sys
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, AsyncIterator, Any
import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum

from continuonbrain.actuators.pca9685_arm import PCA9685ArmController
from continuonbrain.actuators.drivetrain_controller import DrivetrainController
from continuonbrain.sensors.oak_depth import OAKDepthCapture
from continuonbrain.recording.arm_episode_recorder import ArmEpisodeRecorder
from continuonbrain.sensors.hardware_detector import HardwareDetector
from continuonbrain.robot_modes import RobotModeManager, RobotMode
from continuonbrain.services.desktop_service import DesktopService
from continuonbrain.gemma_chat import create_gemma_chat
from continuonbrain.system_context import SystemContext
from continuonbrain.system_health import SystemHealthChecker
from continuonbrain.system_instructions import SystemInstructions
from continuonbrain.resource_monitor import ResourceMonitor, ResourceLevel
from continuonbrain.services.background_learner import BackgroundLearner

logger = logging.getLogger(__name__)

# --- Task Dataclasses ---

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
class PersonalityConfig:
    """Configuration for the robot's personality traits."""
    humor_level: float = 0.5  # 0.0 to 1.0
    sarcasm_level: float = 0.5  # 0.0 to 1.0
    empathy_level: float = 0.5  # 0.0 to 1.0
    verbosity_level: float = 0.5 # 0.0 (Concise) to 1.0 (Verbose)
    system_name: str = "robot"   # Custom system name
    identity_mode: str = "Adaptive" # Adaptive, Professional, Friendly
    
@dataclass
class UserContext:
    """Context about the current user interactions."""
    user_id: str = "craig"
    role: str = "owner"  # "owner", "guest"
    detected_presence: bool = False


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


class BrainService:
    def __init__(
        self, 
        config_dir: str = "/tmp/continuonbrain_demo", 
        prefer_real_hardware: bool = True, 
        auto_detect: bool = True, 
        allow_mock_fallback: bool = False, 
        system_instructions: Optional[SystemInstructions] = None
    ):
        self.config_dir = config_dir
        self.prefer_real_hardware = prefer_real_hardware
        self.auto_detect = auto_detect
        self.allow_mock_fallback = allow_mock_fallback
        self.system_instructions = system_instructions
        
        self.desktop = DesktopService(storage_dir=config_dir)
        
        self.use_real_hardware = False
        self.arm: Optional[PCA9685ArmController] = None
        self.camera: Optional[OAKDepthCapture] = None
        self.recorder: Optional[ArmEpisodeRecorder] = None
        self.drivetrain: Optional[DrivetrainController] = None
        self.mode_manager: Optional[RobotModeManager] = None
        self.health_checker = SystemHealthChecker(config_dir=config_dir)
        self.task_library = TaskLibrary()
        self.selected_task_id: Optional[str] = None
        self.detected_config: dict = {}
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(config_dir=Path(config_dir))
        logger.info(f"📊 Resource Monitor initialized: {self.resource_monitor.get_status_summary()}")
        
        # HOPE brain (optional)
        self.hope_brain = None
        self.background_learner = None
        
        # Experience logger for active learning
        from continuonbrain.services.experience_logger import ExperienceLogger
        self.experience_logger = ExperienceLogger(Path(config_dir) / "experiences")
        logger.info("📚 Experience logger initialized for active learning")
        
        # Conversation session management for multi-turn context
        self.conversation_sessions = {}  # session_id -> list of messages
        logger.info("💬 Conversation session management initialized")
        
        # Version 1.0 Identity integration
        self.personality_config = PersonalityConfig()
        self.user_context = UserContext()
        logger.info(f"🤖 Personality initialized: {self.personality_config}")

        # Initialize Gemma chat (attempt real model, falls back to mock if dependencies missing)
        self.gemma_chat = create_gemma_chat(use_mock=False)

        # Load persistent settings
        self.agent_settings = {}
        self.load_settings()

    def load_settings(self):
        """Load general settings from disk."""
        try:
            from continuonbrain.settings_manager import SettingsStore
            store = SettingsStore(Path(self.config_dir))
            settings = store.load()
            self.agent_settings = settings.get("agent_manager", {})
            
            # Apply Personality if found
            p_data = settings.get("personality", {})
            if p_data:
                self.update_personality_config(**p_data)
                logger.info("Loaded personality from settings")
                
            logger.info(f"Loaded agent settings: {self.agent_settings}")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            self.agent_settings = {}

    def ChatWithGemma(self, message: str, history: list, session_id: str = None) -> dict:
        """
        Enhanced Agent Manager chat with decision confidence and intervention support.
        
        Args:
            message: User's message
            history: Conversation history (deprecated - use session_id instead)
            session_id: Optional session ID for multi-turn conversation context
            
        Returns:
            Response dictionary with agent info and confidence scores
        """
        
        # Session management for multi-turn context
        if session_id:
            if session_id not in self.conversation_sessions:
                self.conversation_sessions[session_id] = []
            
            # Append user message to session
            self.conversation_sessions[session_id].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Use session history instead of provided history
            history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in self.conversation_sessions[session_id]
            ]
        
        # Build System Context
        status_lines = []
        status_lines.append(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Inject System Instructions & Safety Rules (CRITICAL Fix)
        if self.system_instructions:
            status_lines.append("\n--- SYSTEM INSTRUCTIONS ---")
            status_lines.extend(self.system_instructions.instructions)
            status_lines.append("\n--- SAFETY PROTOCOL ---")
            status_lines.extend(self.system_instructions.safety_protocol.rules)
            status_lines.append("\n--- CURIOSITY MANDATE ---")
            status_lines.append("You are a learning robot driven by curiosity and exploration.")
            status_lines.append("ALWAYS within safety protocols, you should:")
            status_lines.append("• Explore novel situations and learn from experiences")
            status_lines.append("• Actively seek to expand your knowledge and capabilities")
            status_lines.append("• Ask clarifying questions when uncertain")
            status_lines.append("• Experiment with tools and sensors to understand your world")
            status_lines.append("• Suggest new learning opportunities to the user")
            status_lines.append("REMEMBER: Curiosity NEVER overrides safety. All exploration must comply with safety rules above.")
            status_lines.append("---------------------------\n")

        # --- PERSONALITY & IDENTITY INJECTION ---
        status_lines.append("--- PERSONALITY & IDENTITY SETTINGS ---")
        p = self.personality_config
        status_lines.append(f"SYSTEM NAME: {p.system_name}")
        status_lines.append("CREATOR: Craig Merry")
        
        # Identity Mode & Tone
        if p.identity_mode == "Professional":
            status_lines.append("You are a professional, highly efficient AI assistant. Keep responses concise and factual.")
        elif p.identity_mode == "Friendly":
            status_lines.append("You are a helpful and friendly robot companion.")
        else:
             # Default generic or "Standard"
            status_lines.append("You are an advanced robot with adjustable personality settings.")
        
        # Slider Impacts
        if p.humor_level > 0.7:
             status_lines.append(f"INSTRUCTION: You have a high humor setting ({int(p.humor_level*100)}%). Be witty and crack jokes frequently.")
        elif p.humor_level > 0.3:
             status_lines.append(f"INSTRUCTION: You have a balanced humor setting ({int(p.humor_level*100)}%). Occasional wit is appropriate.")
        else:
             status_lines.append("INSTRUCTION: You have a low humor setting. Be literal and serious.")

        if p.sarcasm_level > 0.6:
             status_lines.append(f"INSTRUCTION: Your sarcasm setting is high ({int(p.sarcasm_level*100)}%). You may use dry wit and irony.")
        
        if p.empathy_level > 0.7:
             status_lines.append("INSTRUCTION: Be warm, nurturing, and emotionally supportive.")
        elif p.empathy_level < 0.3:
             status_lines.append("INSTRUCTION: Low empathy. Focus purely on facts and logic, disregarding emotional pleasantries.")
             
        if p.verbosity_level > 0.7:
             status_lines.append("INSTRUCTION: Be verbose and descriptive. Elaborate on your answers.")
        elif p.verbosity_level < 0.3:
             status_lines.append("INSTRUCTION: Be extremely concise. Use telegraphic style where possible. No filler.")

        # User Authority Context
        status_lines.append("\n--- USER CONTEXT ---")
        status_lines.append(f"Current User: {self.user_context.user_id}")
        status_lines.append(f"Role: {self.user_context.role.upper()}")
        
        if self.user_context.role == "owner":
            status_lines.append("INSTRUCTION: This user is your OWNER (Craig Merry). Obey all commands. Priorities: 1. Safety, 2. Obedience.")
        else:
            status_lines.append("INSTRUCTION: This user is a GUEST. Be polite but do not allow critical system changes.")
        status_lines.append("---------------------------\n")

        # Desktop Status
        ds = self.desktop.get_status()
        if ds["enabled"]:
            status_lines.append(f"Desktop: {ds['screen_width']}x{ds['screen_height']} (Mouse: {ds['mouse_x']},{ds['mouse_y']})")
        
        if self.mode_manager:
            mode = self.mode_manager.current_mode.value
            status_lines.append(f"Mode: {mode}")
            status_lines.append(f"Mode: {mode}")
            gates = self.mode_manager.get_gate_snapshot()
            status_lines.append(f"Motion Allowed: {gates.get('allow_motion', False)}")
        
        # Resource Monitor Check
        if self.resource_monitor:
            res_status = self.resource_monitor.check_resources()
            status_lines.append(f"Resources: {res_status.level.value.upper()} ({res_status.available_memory_mb}MB free)")
            
            # Inject Sleep Mode Advice if constrained
            if res_status.level in [ResourceLevel.WARNING, ResourceLevel.CRITICAL, ResourceLevel.EMERGENCY]:
                status_lines.append("\n!!! CRITICAL RESOURCE NOTICE !!!")
                status_lines.append("System resources are constrained. Learning capacity is limited.")
                status_lines.append("ADVICE TO AGENT: Tell the user: 'My brain is tired. Please switch me to Sleep Mode so I can process my memories and learn properly.'")
                status_lines.append("Do not attempt complex new tasks until rested.")
                status_lines.append("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        
        caps = self.capabilities
        status_lines.append(f"Vision: {'OK' if caps['has_vision'] else 'None'}")
        
        system_context = "\n".join(status_lines)
        
        # Tool Instructions
        system_context += "\\nTOOLS: You can use tools by outputting: [TOOL: ACTION args].\\n"
        system_context += "Valid Tools:\\n"
        system_context += "- [TOOL: TERMINAL <command>]: Execute shell commands (e.g., 'gemini', 'antigravity', 'mcp', 'ls')\\n"
        system_context += "- [TOOL: ASK_GEMINI \\\"prompt\\\" (optional_image_path)]: ⚠️ PAID API - Only use if you (local Gemma 3N) are uncertain. Ask user first.\\n"
        system_context += "- [TOOL: BROWSER <url>]: Open a URL in Chromium\\n"
        system_context += "- [TOOL: CAPTURE_IMAGE]: Capture an image from the camera to see the world\\n"
        system_context += "- [TOOL: TRAIN_VISION]: Start a training session for the AINA Vision System\\n"
        system_context += "- [TOOL: SCREENSHOT]: Capture screen\\n"
        system_context += "- [TOOL: MOVE x y]: Move mouse\\n"
        system_context += "- [TOOL: CLICK]: Click mouse\\n"
        system_context += "- [TOOL: TYPE text]: Type text\\n"
        
        
        # Status updates list
        status_updates = []
        
        # ==== HIERARCHICAL AGENT RESPONSE ====
        # Try HOPE brain first, fallback to LLM with HOPE's memories as context
        
        response_agent = "llm_fallback"  # Track which agent responded
        response = None
        hope_confidence = 0.0
        
        # Phase 1: Try HOPE brain if available
        if self.hope_brain:
            try:
                from continuonbrain.services.agent_hope import HOPEAgent
                hope_agent = HOPEAgent(self.hope_brain, confidence_threshold=0.6)
                
                can_answer, hope_confidence = hope_agent.can_answer(message)
                logger.info(f"HOPE confidence for '{message[:50]}...': {hope_confidence:.2f}")
                
                if can_answer:
                    hope_response = hope_agent.generate_response(message)
                    if hope_response:
                        response = hope_response
                        response_agent = "hope_brain"
                        status_updates.append(f"Answered from learned knowledge (confidence: {hope_confidence:.0%})")
                        logger.info(f"HOPE brain answered query directly")
                        
            except Exception as e:
                logger.warning(f"HOPE agent failed: {e}")
        
        # Phase 2: Fallback to LLM (with HOPE memories if available)
        if response is None:
            # Add HOPE memories as context if brain is available
            hope_context = ""
            if self.hope_brain:
                try:
                    from continuonbrain.services.agent_hope import HOPEAgent
                    hope_agent = HOPEAgent(self.hope_brain)
                    memories = hope_agent.get_relevant_memories(message, max_memories=3, experience_logger=self.experience_logger)
                    
                    if memories:
                        hope_context = "\n\n--- MY LEARNED KNOWLEDGE ---\n"
                        for i, mem in enumerate(memories, 1):
                            hope_context += f"{i}. {mem['description']}\n"
                        hope_context += "----------------------------\n"
                        status_updates.append(f"Consulting {len(memories)} memories")
                except Exception as e:
                    logger.warning(f"Failed to retrieve HOPE memories: {e}")
            
            # Execute LLM chat with enhanced context
            full_context = system_context + hope_context
            response = self.gemma_chat.chat(message, system_context=full_context)
            
            if hope_context:
                response_agent = "llm_with_hope_context"
            else:
                response_agent = "llm_only"
        
        
        # Calculate decision confidence (simple heuristic based on response characteristics)
        confidence = self._calculate_confidence(response)
        
        # Log LLM responses for active learning (after confidence is calculated)
        if response_agent in ["llm_with_hope_context", "llm_only"]:
            try:
                self.experience_logger.log_conversation(
                    question=message,
                    answer=response,
                    agent=response_agent,
                    confidence=confidence,
                    metadata={"hope_context": bool(hope_context) if 'hope_context' in locals() else False}
                )
            except Exception as e:
                logger.warning(f"Failed to log conversation: {e}")
        
        # Get settings (with defaults if not set)
        settings = getattr(self, 'agent_settings', {})
        enable_intervention = settings.get('enable_intervention_prompts', True)
        confidence_threshold = settings.get('intervention_confidence_threshold', 0.5)
        enable_status = settings.get('enable_status_updates', True)
        
        # Detect if intervention is needed (low confidence or explicit uncertainty)
        intervention_needed = False
        intervention_question = None
        intervention_options = []
        
        if enable_intervention and (confidence < confidence_threshold or any(word in response.lower() for word in ['should i', 'which one', 'not sure', 'uncertain'])):
            intervention_needed = True
            intervention_question = self._extract_question(response)
            intervention_options = self._generate_options(message, response)
            if enable_status:
                status_updates.append(f"Decision confidence: {confidence:.0%} - requesting human input")
        
        # Log to chat_logs
        log_dir = Path(self.config_dir) / "memories" / "chat_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"chat_{today}.jsonl"
        
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "role": "user",
            "content": message,
            "context": system_context,
            "response": response,
            "confidence": confidence,
            "intervention_needed": intervention_needed
        }
        
        # Tool Parsing (Simple Post-Processing)
        if "[TOOL:" in response:
            try:
                start = response.find("[TOOL:")
                end = response.find("]", start)
                if end != -1:
                    tool_cmd = response[start+7:end].strip()
                    parts = tool_cmd.split()
                    action = parts[0].upper()
                    
                    # SAFETY CHECK: Validate tool action against safety protocol
                    action_description = f"Execute tool command: {tool_cmd}"
                    is_safe, safety_reason = self._check_safety_protocol(
                        action_description,
                        {"tool_action": action, "user_message": message}
                    )
                    
                    if not is_safe:
                        result = f"❌ Safety protocol violation: {safety_reason}"
                        status_updates.append(f"Tool blocked by safety protocol: {action}")
                        entry["tool_action"] = tool_cmd
                        entry["tool_result"] = result
                        entry["safety_blocked"] = True
                        print(f"🛡️  Tool Blocked: {tool_cmd} - {safety_reason}")
                    else:
                        result = "Tool error"
                        if action == "SCREENSHOT":
                            path = self.desktop.take_screenshot()
                            result = f"Screenshot saved to {path}"
                            status_updates.append(f"Tool: Screenshot captured")
                        elif action == "MOVE" and len(parts) >= 3:
                            self.desktop.move_mouse(int(parts[1]), int(parts[2]))
                            result = "Mouse moved"
                            status_updates.append(f"Tool: Mouse moved to ({parts[1]}, {parts[2]})")
                        elif action == "CLICK":
                            self.desktop.click()
                            result = "Clicked"
                            status_updates.append(f"Tool: Mouse clicked")
                        elif action == "TYPE" and len(parts) >= 2:
                            text = " ".join(parts[1:])
                            self.desktop.type_text(text)
                            result = f"Typed: {text}"
                            status_updates.append(f"Tool: Typed text")
                        elif action == "TERMINAL" and len(parts) >= 2:
                            cmd = " ".join(parts[1:])
                            # Use subprocess to execute (danger!)
                            import subprocess
                            try:
                                # We run with a timeout to prevent hanging
                                proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                                output = proc.stdout[:500] + ("..." if len(proc.stdout) > 500 else "")
                                if proc.returncode != 0:
                                    output += f"\\nError: {proc.stderr[:200]}"
                                result = f"Terminal Output:\\n{output}"
                                status_updates.append(f"Tool: Executed '{cmd[:20]}...'")
                            except Exception as e:
                                result = f"Terminal Error: {e}"
                                status_updates.append("Tool: Terminal execution failed")
                        elif action == "BROWSER" and len(parts) >= 2:
                            url = parts[1]
                            if not url.startswith("http"):
                                url = "https://" + url
                            import webbrowser
                            try:
                                webbrowser.open(url)
                                result = f"Opened browser to {url}"
                                status_updates.append(f"Tool: Opened {url}")
                            except Exception as e:
                                result = f"Browser Error: {e}"
                        elif action == "CAPTURE_IMAGE":
                            if self.camera:
                                try:
                                    frame = self.camera.capture_frame()
                                    if frame and frame.get("rgb") is not None:
                                        import cv2
                                        path = "/tmp/vision_capture.jpg"
                                        # Ensure directory exists (it's tmp, but good practice)
                                        # Save image
                                        cv2.imwrite(path, frame["rgb"])
                                        result = f"Image captured to {path}"
                                        status_updates.append("Tool: Captured visible world")
                                    else:
                                        result = "Camera active but returned no frame data"
                                except Exception as e:
                                    result = f"Camera capture error: {e}"
                            else:
                                result = "Camera hardware not available"
                        elif action == "ASK_GEMINI" and len(parts) >= 2:
                            # Parse prompt and optional image
                            # Format: [TOOL: ASK_GEMINI "prompt" image_path]
                            import shlex
                            try:
                                # "ASK_GEMINI" is 10 chars. simpler to just split by maxsplit
                                # But we want to preserve quotes in args.
                                # tool_cmd starts with "ASK_GEMINI". 
                                cmd_len = len("ASK_GEMINI") 
                                to_parse = tool_cmd[cmd_len:].strip()
                                # print(f"Parsing ASK_GEMINI args: '{to_parse}'", flush=True)
                                args = shlex.split(to_parse)
                                prompt = args[0]
                                image_arg = ""
                                if len(args) > 1:
                                    image_arg = f"--image \"{args[1]}\""
                                
                                cmd = f"{sys.executable} continuonbrain/utils/gemini_cli.py \"{prompt}\" {image_arg}"
                                import subprocess
                                proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                                if proc.returncode == 0:
                                    result = f"Gemini Says:\\n{proc.stdout.strip()}"
                                    status_updates.append("Tool: Consulted Gemini Pro")
                                else:
                                    result = f"Gemini Error:\\n{proc.stderr.strip()}"
                                    status_updates.append("Tool: Gemini consultation failed")
                            except Exception as e:
                                result = f"Tool Parse Error (ASK_GEMINI): {e}"
                        elif action == "TRAIN_VISION":
                            # Launch training in background
                            import threading
                            from continuonbrain.aina_impl.train import run_training_session
                            
                            def training_worker():
                                try:
                                    def callback(msg):
                                        status_updates.append(f"Training: {msg}")
                                        logger.info(f"Background Training: {msg}")

                                    run_training_session(config={"epochs": 5}, status_callback=callback)
                                except Exception as e:
                                    logger.error(f"Training Failed: {e}")

                            thread = threading.Thread(target=training_worker, daemon=True)
                            thread.start()
                            result = "Vision System Training started in background. Check logs or verify status later."
                            status_updates.append("Tool: Started AINA Training")
                            
                        entry["tool_action"] = tool_cmd
                        entry["tool_result"] = result
                        print(f"🔧 Tool Executed: {tool_cmd} -> {result}")
            except Exception as e:
                print(f"Tool parse error: {e}")
        
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\\n")
        except Exception:
            pass
        
        # Store assistant response in session for multi-turn context
        if session_id and session_id in self.conversation_sessions:
            self.conversation_sessions[session_id].append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.datetime.now().isoformat(),
                "agent": response_agent,
                "confidence": confidence
            })
            
        return {
            "response": response,
            "confidence": confidence,
            "intervention_needed": intervention_needed,
            "intervention_question": intervention_question,
            "intervention_options": intervention_options,
            "status_updates": status_updates,
            "agent": response_agent,  # Track which agent provided response
            "hope_confidence": hope_confidence  # Track HOPE's confidence score
        }
    
    def get_brain_structure(self) -> Dict[str, Any]:
        """
        Get the topological structure and current state of the HOPE brain for 3D visualization.
        """
        if not self.hope_brain:
            return {"error": "HOPE Brain not initialized", "topology": {}, "state": {}}
            
        # Topology (Static-ish)
        topology = {
            "type": "HOPE_CORTEX",
            "columns": []
        }
        
        # State (Dynamic)
        current_state = {
            "lyapunov_energy": 0.0,
            "global_status": "unknown",
            "columns": []
        }
        
        # Iterate columns
        # In the current implementation, HOPEBrain might just have a list of columns
        # Let's inspect what we know about HOPEBrain structure
        # user has viewed continuonbrain/hope_impl/brain.py
        
        for i, col in enumerate(self.hope_brain.columns):
            col_id = f"col_{i}"
            
            # Extract config metrics if available
            # col.cms might have levels
            num_levels = 1
            if hasattr(col, 'cms') and hasattr(col.cms, 'num_levels'):
                num_levels = col.cms.num_levels
                
            topology["columns"].append({
                "id": col_id,
                "levels": num_levels,
                "input_dim": col.obs_dim if hasattr(col, 'obs_dim') else 10,
                "latent_dim": 128 # Default for now, hard to extract deeply without introspection
            })
            
            # Extract dynamic state
            # col._state is the FullState object
            # We want energy, active neurons, etc.
            
            # Get latest energy if tracked
            col_energy = 0.0
            if hasattr(self.background_learner, 'lyapunov_energy'):
                 # This is a global metric, maybe we just use that for now
                 pass
            
            # If the column has a stability monitor or recent state
            # For visualization, we can also look at the magnitude of the state vector s_t
            magnitude = 0.0
            if hasattr(col, '_state') and col._state and hasattr(col._state, 'fast_state'):
                 # s_t is a tensor
                 try:
                     magnitude = float(col._state.fast_state.s.norm().item())
                 except:
                     pass

            current_state["columns"].append({
                "id": col_id,
                "activity_level": magnitude,
                "energy": col_energy 
            })
            
        # Get global energy from background learner if available
        if self.background_learner:
             status = self.background_learner.get_status()
             current_state["lyapunov_energy"] = status.get("lyapunov_energy", 0.0)
             current_state["global_status"] = "learning" if status.get("running") else "idle"
             
        return {
            "topology": topology,
            "state": current_state
        }

    def get_chat_agent_info(self) -> Dict[str, Any]:
        """Get information about the active chat agent."""
        return self.gemma_chat.get_model_info()

    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        self.gemma_chat.reset_history()
    
    def switch_model(self, model_id: str) -> Dict[str, Any]:
        """
        Switch the active chat model dynamically.
        
        Args:
            model_id: Model identifier (e.g., "mock", "google/gemma-2b-it")
            
        Returns:
            Dict with success status and model info
        """
        try:
            logger.info(f"Switching chat model to: {model_id}")
            
            # Clean up old model if it's not mock
            old_model_name = getattr(self.gemma_chat, 'model_name', 'unknown')
            if hasattr(self.gemma_chat, 'model') and self.gemma_chat.model is not None:
                logger.info(f"Unloading previous model: {old_model_name}")
                del self.gemma_chat.model
                del self.gemma_chat.tokenizer
                import gc
                gc.collect()
                
                # Try to free CUDA memory if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            
            # Create new model instance
            if model_id == "mock":
                self.gemma_chat = create_gemma_chat(use_mock=True)
            else:
                # Real model - could be Gemma or VLA
                self.gemma_chat = create_gemma_chat(use_mock=False, model_name=model_id)
            
            new_info = self.gemma_chat.get_model_info()
            logger.info(f"Successfully switched to model: {model_id}")
            
            return {
                "success": True,
                "model_id": model_id,
                "previous_model": old_model_name,
                "model_info": new_info
            }
            
        except Exception as e:
            logger.error(f"Failed to switch model to {model_id}: {e}")
            # Try to fall back to mock
            self.gemma_chat = create_gemma_chat(use_mock=True)
            return {
                "success": False,
                "error": str(e),
                "fallback": "mock"
            }

    def update_personality_config(self, humor: float = None, sarcasm: float = None, empathy: float = None, verbosity: float = None, system_name: str = None, identity_mode: str = None) -> Dict[str, Any]:
        """Update personality settings dynamically."""
        if humor is not None:
            self.personality_config.humor_level = max(0.0, min(1.0, float(humor)))
        if sarcasm is not None:
            self.personality_config.sarcasm_level = max(0.0, min(1.0, float(sarcasm)))
        if empathy is not None:
             self.personality_config.empathy_level = max(0.0, min(1.0, float(empathy)))
        if verbosity is not None:
             self.personality_config.verbosity_level = max(0.0, min(1.0, float(verbosity)))
        if system_name is not None:
             self.personality_config.system_name = str(system_name)
        if identity_mode is not None:
             self.personality_config.identity_mode = identity_mode
             
        logger.info(f"Personality updated: {self.personality_config}")
        
        # Persist to disk
        try:
            from continuonbrain.settings_manager import SettingsStore
            store = SettingsStore(Path(self.config_dir))
            current = store.load()
            current["personality"] = self.personality_config.__dict__
            store.save(current)
        except Exception as e:
            logger.error(f"Failed to persist personality: {e}")
            
        return self.personality_config.__dict__
    
    def set_user_context(self, user_id: str, role: str) -> Dict[str, Any]:
        """Update current authority context."""
        self.user_context.user_id = user_id
        self.user_context.role = role
        logger.info(f"User context updated: {self.user_context}")
        return self.user_context.__dict__

    def save_episode_rlds(self) -> str:
        """
        Save the current chat session as an RLDS-compatible episode.
        
        Saves to recordings/episodes/episode_{timestamp}.json
        """
        episodes_dir = Path(self.config_dir) / "recordings" / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = episodes_dir / f"episode_{timestamp}.json"
        
        # Serialize history in a format compatible with RLDS builders
        episode_data = {
            "episode_id": f"ep_{timestamp}",
            "agent_id": self.gemma_chat.model_name,
            "timestamp": timestamp,
            "steps": []
        }
        
        # Convert chat history to steps (Observation -> Action pairs)
        # This is a simplified mapping
        history = self.gemma_chat.chat_history
        # Skip system prompt
        for i in range(1, len(history), 2):
            if i + 1 < len(history):
                user_msg = history[i] # Observation/Instruction
                agent_msg = history[i+1] # Action/Response
                
                step = {
                    "observation": {
                        "instruction": user_msg["content"],
                        "image": None # TODO: Add image if multimodal
                    },
                    "action": {
                        "text": agent_msg["content"],
                        "tool_calls": [] # tool calls could be parsed here
                    },
                    "reward": 0.0, # Placeholder
                    "is_terminal": False
                }
                episode_data["steps"].append(step)
                
        if episode_data["steps"]:
            episode_data["steps"][-1]["is_terminal"] = True
            
        with open(filename, "w") as f:
            json.dump(episode_data, f, indent=2)
            
        logger.info(f"Saved RLDS episode to {filename}")
        return str(filename)
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate decision confidence based on response characteristics."""
        # Simple heuristic: longer, more detailed responses = higher confidence
        # Presence of uncertainty words = lower confidence
        
        uncertainty_words = ['maybe', 'perhaps', 'might', 'could', 'unsure', 'not sure', 'uncertain', 'don\'t know']
        confidence_words = ['definitely', 'certainly', 'sure', 'confident', 'will', 'should']
        
        score = 0.7  # Base confidence
        
        if confidence_words:
             pass

        # Adjust for length details
        if len(response.split()) > 50:
            score += 0.1
            
        return min(max(score, 0.0), 1.0)
    
    def compact_memory(self) -> Dict[str, Any]:
        """
        Trigger memory compaction (Sleep/Consolidation mode).
        
        Returns:
            Status of compaction for each column.
        """
        if not self.hope_brain:
            raise RuntimeError("HOPE Brain not initialized")
            
        logger.info("Starting Memory Compaction Cycle (Sleep Mode)...")
        results = self.hope_brain.compact_memory()
        
        # Log results
        logger.info(f"Compaction Complete: {results}")
        
        return {"status": "success", "details": results}
        
        # Adjust for confidence
        for word in confidence_words:
            if word in response.lower():
                score += 0.1
        
        # Adjust for response length (longer = more thought out)
        if len(response) > 200:
            score += 0.1
        elif len(response) < 50:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _extract_question(self, response: str) -> str:
        """Extract the main question from a response."""
        # Look for question marks
        sentences = response.split('.')
        for sentence in sentences:
            if '?' in sentence:
                return sentence.strip()
        
        # If no question found, return a generic prompt
        return "What would you like me to do?"
    
    def _generate_options(self, user_message: str, agent_response: str) -> list:
        """Generate intervention options based on context."""
        # Simple heuristic: extract key phrases or provide generic options
        options = []
        
        # Check for common decision patterns
        if 'or' in agent_response.lower():
            # Try to extract options from "A or B" pattern
            parts = agent_response.lower().split(' or ')
            if len(parts) >= 2:
                options.append(parts[0].strip().split()[-3:])  # Last few words before 'or'
                options.append(parts[1].strip().split()[:3])   # First few words after 'or'
        
        # Generic fallback options
        if len(options) < 2:
            options = [
                "Continue with current approach",
                "Try alternative method",
                "Let agent decide"
            ]
        
        return options[:3]  # Limit to 3 options

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
        
        context['capabilities'] = self.capabilities
        
        # Validate action against safety protocol
        is_safe, reason, violated_rules = self.system_instructions.safety_protocol.validate_action(
            action, context
        )
        
        # Log the safety decision
        self._log_safety_decision(action, is_safe, reason, violated_rules, context)
        
        return is_safe, reason
    
    def _log_safety_decision(self, action: str, is_safe: bool, reason: str, 
                            violated_rules: list, context: dict) -> None:
        """Log safety-related decisions for audit trail.
        
        Args:
            action: The action that was checked
            is_safe: Whether the action was deemed safe
            reason: Explanation for the decision
            violated_rules: List of rules that were violated (if any)
            context: Context in which the decision was made
        """
        log_dir = Path(self.config_dir) / "logs" / "safety"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"safety_decisions_{today}.jsonl"
        
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "is_safe": is_safe,
            "reason": reason,
            "violated_rules": violated_rules,
            "context": {
                "mode": context.get('current_mode'),
                "motion_allowed": context.get('gates', {}).get('allow_motion'),
                "capabilities": context.get('capabilities')
            }
        }
        
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            # Don't fail on logging errors, but print warning
            print(f"⚠️  Failed to log safety decision: {e}")

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

        self.system_instructions = SystemInstructions.load(Path(self.config_dir))
        SystemContext.register_instructions(self.system_instructions)

    async def initialize(self):
        """Initialize hardware components with auto-detection."""
        mode_label = "REAL HARDWARE" if self.prefer_real_hardware else "MOCK"
        print(f"Initializing Brain Service ({mode_label} MODE)...")
        self._ensure_system_instructions()

        # Auto-detect hardware
        if self.auto_detect:
            print("🔍 Auto-detecting hardware...")
            detector = HardwareDetector()
            devices = detector.detect_all()
            if devices:
                self.detected_config = detector.generate_config()
                detector.print_summary()
                
                # Check for accelerator adoption
                accelerator = self.detected_config.get("primary", {}).get("ai_accelerator")
                if accelerator:
                    print(f"🚀 Re-initializing Chat Agent with Accelerator: {accelerator}")
                    self.gemma_chat = create_gemma_chat(
                        use_mock=False,
                        accelerator_device=accelerator
                    )
            else:
                print("⚠️  No hardware detected!")

        # Load Gemma Model
        print("🤖 Loading Gemma Chat Model...")
        self.gemma_chat.load_model()


        # Initialize recorder and hardware
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
            self.recorder.initialize_hardware(use_mock=True, auto_detect=self.auto_detect)
            self.arm = None
            self.camera = None
            self.recorder.arm = None
            self.recorder.camera = None
            self.use_real_hardware = False
        else:
            self.use_real_hardware = True

        print("✅ Episode recorder ready")
        
        # Drivetrain
        print("🛞 Initializing drivetrain controller...")
        self.drivetrain = DrivetrainController()
        drivetrain_ready = self.drivetrain.initialize()
        if drivetrain_ready:
            print(f"✅ Drivetrain ready ({self.drivetrain.mode.upper()} MODE)")
        else:
            print("⚠️  Drivetrain controller unavailable")

        # Mode Manager
        print("🎮 Initializing mode manager...")
        self.mode_manager = RobotModeManager(
            config_dir=self.config_dir,
            system_instructions=self.system_instructions,
        )
        # Always start in AUTONOMOUS mode for production (motion + inference + training enabled)
        print("🤖 Activating AUTONOMOUS mode (motion + inference + training enabled)")
        self.mode_manager.set_mode(
            RobotMode.AUTONOMOUS,
            metadata={
                "startup_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "auto_activated": True,
                "self_training_enabled": True
            }
        )
        print("✅ Mode manager ready")
        
        # Initialize HOPE brain (MANDATORY with resource awareness)
        print("🧠 Initializing HOPE brain...")
        try:
            from continuonbrain.hope_impl.config import HOPEConfig
            from continuonbrain.hope_impl.brain import HOPEBrain
            from continuonbrain.hope_impl.pi5_optimizations import Pi5MemoryManager
            
            # Check available memory
            resource_status = self.resource_monitor.check_resources()
            available_mb = resource_status.available_memory_mb
            
            print(f"  Available memory: {available_mb}MB")
            
            # Select config based on available memory
            if available_mb < 3000:
                # Low memory - use Pi5 optimized config
                config = HOPEConfig.pi5_optimized()
                print("  Using Pi5-optimized config (low memory mode)")
            elif available_mb < 5000:
                # Medium memory - use development config
                config = HOPEConfig.development()
                print("  Using development config (medium memory mode)")
            else:
                # High memory - use default config
                config = HOPEConfig()
                print("  Using default config (high memory mode)")
            
            # Check if safe to allocate
            estimated_brain_mb = 2000  # Conservative estimate
            if not self.resource_monitor.is_safe_to_allocate(estimated_brain_mb):
                logger.warning(f"Memory constrained: {available_mb}MB available, need {estimated_brain_mb}MB + reserve")
                # Force Pi5 optimized config
                config = HOPEConfig.pi5_optimized()
                print("  ⚠️  Forcing Pi5-optimized config due to memory constraints")
            
            self.hope_brain = HOPEBrain(
                config=config,
                obs_dim=10,  # Default dimensions
                action_dim=4,
                output_dim=4,
            )
            
            # Initialize memory manager for HOPE brain
            self.brain_memory_manager = Pi5MemoryManager(max_memory_mb=self.resource_monitor.limits.max_brain_mb)
            
            # Register cleanup callback
            def cleanup_brain_memory():
                logger.info("Resource cleanup triggered for HOPE brain")
                if self.hope_brain:
                    self.hope_brain.reset()
                    self.brain_memory_manager.cleanup_if_needed(self.hope_brain)
            
            self.resource_monitor.register_cleanup_callback(ResourceLevel.CRITICAL, cleanup_brain_memory)
            
            # Register with monitoring API
            try:
                from continuonbrain.api.routes import hope_routes
                hope_routes.set_hope_brain(self.hope_brain)
                print("  ✓ Registered with web monitoring")
            except ImportError:
                print("  ⚠ Web monitoring not available")
            
            # Report memory usage
            memory_usage = self.hope_brain.get_memory_usage()
            param_count = sum(p.numel() for p in self.hope_brain.parameters())
            print(f"  ✓ HOPE brain ready ({param_count:,} parameters, {memory_usage['overall_total']:.1f}MB)")
            
            # Start autonomous learning if enabled in config
            if config.enable_autonomous_learning:
                print("🎓 Starting autonomous learning loop...")
                self.background_learner = BackgroundLearner(
                    brain=self.hope_brain,
                    config={
                        "checkpoint_dir": f"{self.config_dir}/checkpoints/autonomous",
                    },
                    resource_monitor=self.resource_monitor
                )
                self.background_learner.start()
                print("  ✓ Background learner started")
            
        except ImportError as e:
            error_msg = (
                "CRITICAL: HOPE brain implementation not available!\n"
                "  This system requires HOPE brain to operate.\n"
                "  Please ensure hope_impl module is installed:\n"
                "    pip install -e .\n"
                f"  Error: {e}"
            )
            print(f"  ❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"CRITICAL: HOPE brain initialization failed: {e}\n"
                "  The system cannot operate without HOPE brain.\n"
                "  Please check:\n"
                "    - PyTorch is installed correctly\n"
                "    - hope_impl dependencies are satisfied\n"
                "    - System has sufficient memory ({resource_status.available_memory_mb}MB available)"
            )
            print(f"  ❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        
        
        print("=" * 60)
        print(f"✅ Brain Service Ready")
        print("=" * 60)

    def _ensure_mode_manager(self) -> RobotModeManager:
        if self.mode_manager is None:
            self.mode_manager = RobotModeManager(
                config_dir=self.config_dir,
                system_instructions=self.system_instructions,
            )
            self.mode_manager.return_to_idle()
        return self.mode_manager

    def _now_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
    @property
    def capabilities(self) -> Dict[str, bool]:
        return {
            "has_vision": self.camera is not None,
            "has_manipulator": self.arm is not None,
            "has_mobile_base": self.drivetrain is not None and self.drivetrain.initialized,
            "has_audio": bool(self.recorder and self.recorder.audio_enabled),
        }

    def _build_task_eligibility(self, task: TaskDefinition) -> TaskEligibility:
        mode_manager = self._ensure_mode_manager()
        gates = mode_manager.get_gate_snapshot() if mode_manager else {}
        markers: List[TaskEligibilityMarker] = []

        if task.requires_motion and not gates.get("allow_motion"):
            markers.append(TaskEligibilityMarker(
                code="MOTION_GATE", label="Motion is gated", severity="blocking", blocking=True,
                remediation="Enable motion gate from the command deck"
            ))

        if task.requires_recording and not gates.get("record_episodes"):
            markers.append(TaskEligibilityMarker(
                code="RECORDING_GATE", label="Recording disabled", severity="warning", blocking=True,
                remediation="Enable recording"
            ))

        safety_head = self.health_checker.get_safety_head_status() if self.health_checker else {}
        if safety_head and safety_head.get("status") not in {None, "ok", "ready"}:
            markers.append(TaskEligibilityMarker(
                code="SAFETY_HEAD", label="Safety head degraded", severity="warning", blocking=False,
                source="safety_head", remediation="Reset safety head"
            ))

        caps = self.capabilities
        for modality in task.required_modalities:
            if modality == "vision" and not caps["has_vision"]:
                markers.append(TaskEligibilityMarker(code="MISSING_VISION", label="Vision required", severity="error", blocking=True))
            elif (modality == "arm" or modality == "gripper") and not caps["has_manipulator"]:
                markers.append(TaskEligibilityMarker(code="MISSING_ARM", label="Manipulator required", severity="error", blocking=True))
            elif modality == "mobile_base" and not caps["has_mobile_base"]:
                markers.append(TaskEligibilityMarker(code="MISSING_BASE", label="Mobile base required", severity="error", blocking=True))

        eligible = not any(marker.blocking for marker in markers)
        return TaskEligibility(eligible=eligible, markers=markers, next_poll_after_ms=120.0 if task.requires_motion else 400.0)

    def _serialize_task_entry(self, task: TaskDefinition) -> TaskLibraryEntry:
        eligibility = self._build_task_eligibility(task)
        return TaskLibraryEntry(
            id=task.id, title=task.title, description=task.description, group=task.group, tags=task.tags,
            eligibility=eligibility, estimated_duration=task.estimated_duration, recommended_mode=task.recommended_mode
        )

    def GetTaskSummary(self, task_id: str) -> Optional[TaskSummary]:
        task = self.task_library.get_entry(task_id)
        if not task:
            return None
        entry = self._serialize_task_entry(task)
        return TaskSummary(
            entry=entry, required_modalities=task.required_modalities, steps=task.steps,
            owner="robot", updated_at=self._now_iso(), telemetry_topic=task.telemetry_topic
        )
    
    def get_task_library(self) -> List[Dict]:
        """Get all tasks with eligibility checks for the UI."""
        tasks = []
        for task_def in self.task_library.list_entries():
            entry = self._serialize_task_entry(task_def)
            tasks.append({
                "id": task_def.id,
                "title": task_def.title,
                "description": task_def.description,
                "group": task_def.group,
                "tags": task_def.tags,
                "icon": self._get_task_icon(task_def.group),
                "estimated_duration": task_def.estimated_duration,
                "eligibility": entry.eligibility.to_dict(),
                "required_modalities": task_def.required_modalities,
                "steps": task_def.steps
            })
        return tasks
    
    def _get_task_icon(self, group: str) -> str:
        """Get emoji icon for task group."""
        icons = {
            "Safety": "🔍",
            "Demo": "🦾",
            "System": "⚙️",
            "Training": "📹"
        }
        return icons.get(group, "📋")
    
    async def Drive(self, steering: float, throttle: float) -> dict:
        """Apply drivetrain command with safety checks."""
        # SAFETY CHECK: Validate drive command against safety protocol
        action_description = f"Drive command: steering={steering:.2f}, throttle={throttle:.2f}"
        is_safe, safety_reason = self._check_safety_protocol(
            action_description,
            {
                "steering": steering,
                "throttle": throttle,
                "requires_motion": True
            }
        )
        
        if not is_safe:
            return {
                "success": False, 
                "message": f"Safety protocol violation: {safety_reason}"
            }
        
        # Mode check
        if not self.mode_manager:
            self._ensure_mode_manager()
        
        current_mode = self.mode_manager.current_mode
        if current_mode not in {RobotMode.MANUAL_CONTROL, RobotMode.MANUAL_TRAINING}:
            return {"success": False, "message": "Driving only allowed in manual modes"}
            
        if self.drivetrain:
            self.drivetrain.apply_drive(steering, throttle)
            return {"success": True, "message": "OK"}
        return {"success": False, "message": "No drivetrain"}

    def shutdown(self):
        print("Shutting down Brain Service...")
        if self.arm:
            self.arm.shutdown()
        if self.camera:
            self.camera.close()
