import datetime
import json
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional


class ChatAdapter:
    """Lightweight wrapper around Gemma chat with logging and fallback responses."""

    def __init__(
        self,
        config_dir: str,
        status_provider: Callable[[], Awaitable[dict]],
        gemma_chat: Optional[object] = None,
    ) -> None:
        self.config_dir = config_dir
        self.status_provider = status_provider
        self.gemma_chat = gemma_chat

    async def chat(self, message: str, history: list, model_hint: Optional[str] = None) -> dict:
        """Chat with Gemma (or a local fallback) using live status for context."""
        try:
            status_data = await self.status_provider()
            status = status_data.get("status", {})
            mode = status.get("mode", "unknown")
            hardware = status.get("hardware_mode", "unknown")
            allow_motion = status.get("allow_motion", False)

            prompt = self._build_prompt(mode, hardware, allow_motion)
            response = self._call_model(message, prompt, history, model_hint=model_hint)
            self._log_chat(message, response, status)
            return {"response": response}
        except Exception as exc:  # noqa: BLE001
            print(f"Chat error: {exc}")
            return {"error": str(exc)}

    def _build_prompt(self, mode: str, hardware: str, allow_motion: bool) -> str:
        return (
            "You are the Agent Manager for the Continuon Brain Studio (🤖 Gemma 3n on-device orchestrator).\n"
            "Responsibilities:\n"
            "- Host the primary on-device model and fall back to other available models when confidence is low or resources are constrained.\n"
            "- Connect and supervise sub-agents/tools to plan, review, and safely execute actions.\n"
            "- Run training/evaluation loops for researchers using local memories and RLDS datasets saved manually or automatically; surface gaps and next actions.\n"
            "- Help end users improve the robot itself: suggest routines, safety checks, and self-improvements that can be applied on-device.\n"
            "- Preserve and use conversation + experience context to incrementally self-improve.\n"
            f"Current Status:\n- Mode: {mode}\n- Hardware: {hardware}\n- Motion Allowed: {allow_motion}\n\n"
            "Always answer as the Agent Manager. State when you are using the primary model, a fallback, or a sub-agent/tool. Be concise, technical, and action-oriented."
        )

    def _call_model(self, message: str, prompt: str, _history: list, model_hint: Optional[str] = None) -> str:
        if self.gemma_chat:
            return self.gemma_chat.chat(message, system_context=prompt, model_hint=model_hint)

        # Robustly extract mode/hardware even if prompt structure changes
        lines = prompt.splitlines()
        def _extract(prefix: str, default: str = "unknown") -> str:
            for line in lines:
                if line.lower().startswith(prefix.lower()):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        return parts[1].strip()
            return default

        context = (
            f"Mode={_extract('- Mode', 'unknown')} "
            f"Hardware={_extract('- Hardware', 'unknown')}"
        )
        return self._generate_response(message, context)

    def _log_chat(self, message: str, response: str, status: Dict[str, object]) -> None:
        try:
            log_dir = Path(self.config_dir) / "memories" / "chat_logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "user_message": message,
                "agent_response": response,
                "context": status,
            }

            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            log_file = log_dir / f"chat_{date_str}.jsonl"
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_entry) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to log chat: {exc}")

    def _generate_response(self, message: str, context: str) -> str:
        """Generate a lightweight fallback response when Gemma is unavailable."""
        msg_lower = message.lower()

        if any(word in msg_lower for word in ["status", "state", "how", "what"]):
            return (
                f"Robot status: {context}. The robot is ready for your commands. "
                "Use the arrow controls or keyboard to move the arm joints and drive the car."
            )

        if any(word in msg_lower for word in ["control", "move", "drive", "steer"]):
            return (
                "Control the arm with joint sliders or arrow buttons. For the car, use the driving controls "
                "- default speed is set to SLOW (0.3) for safety. Hold Ctrl+Arrow keys for keyboard driving, "
                "or use WASD for arm control."
            )

        if any(word in msg_lower for word in ["joint", "arm", "gripper"]):
            return (
                "The arm has 6 joints: J0 (base rotation), J1 (shoulder), J2 (elbow), "
                "J3 (wrist roll), J4 (wrist pitch), and J5 (gripper). Use the sliders or arrow buttons "
                "to control each joint. Values range from -1.0 to 1.0."
            )

        if any(word in msg_lower for word in ["car", "speed", "throttle"]):
            return (
                "The car is based on a DonkeyCar RC platform. Speed is preset to SLOW (0.3) for safety "
                "- you can adjust using the speed buttons (Crawl, Slow, Med, Fast). Use arrow buttons or "
                "keyboard to steer and control throttle."
            )

        if any(word in msg_lower for word in ["record", "episode", "training"]):
            return (
                "Episode recording captures your manual control demonstrations for training. "
                "Make sure you're in manual_training mode and motion is enabled. "
                "Your actions will be recorded as RLDS episodes."
            )

        if any(word in msg_lower for word in ["safe", "stop", "emergency"]):
            return (
                "For safety, the speed is preset to SLOW. Use the Emergency Stop button if needed - "
                "it will halt all motion immediately. Always start with slow movements to test the robot's response."
            )

        return (
            f"I'm here to help with robot control! Current status: {context}. "
            "Ask me about controls, status, movement, or safety."
        )

