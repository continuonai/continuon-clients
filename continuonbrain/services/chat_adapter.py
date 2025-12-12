import datetime
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple


STRUCTURED_PREFIX = "__CONTINUON_STRUCTURED__:"


def _extract_structured_block(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract an optional single-line structured JSON payload from the model output.

    Expected format (single line, valid JSON):
      __CONTINUON_STRUCTURED__:{...}

    Returns:
      (clean_text, structured_dict)
    """
    if not text:
        return "", {}

    lines = text.splitlines()
    structured: Dict[str, Any] = {}
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(STRUCTURED_PREFIX):
            raw = stripped[len(STRUCTURED_PREFIX) :].strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    structured = parsed
            except Exception:
                structured = {}
            continue
        kept.append(line)
    clean = "\n".join(kept).strip()
    return clean, structured


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

    async def chat(
        self,
        message: str,
        history: list,
        model_hint: Optional[str] = None,
        *,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Chat with Gemma (or a local fallback) using live status for context.

        Returns a BrainService-compatible envelope so UIs/clients can rely on a stable schema:
        - response: str
        - structured: optional dict with goals/probes/plan/approvals
        - intervention_needed/intervention_question/intervention_options: optional human-in-the-loop
        - status_updates: list[str] for lightweight traceability
        - agent: label for which backend answered (best-effort)
        - confidence/hope_confidence: optional floats when available (best-effort)
        """
        try:
            status_data = await self.status_provider()
            status = status_data.get("status", {})
            mode = status.get("mode", "unknown")
            hardware = status.get("hardware_mode", "unknown")
            allow_motion = status.get("allow_motion", False)

            prompt = self._build_prompt(mode, hardware, allow_motion)
            raw_response = self._call_model(message, prompt, history, model_hint=model_hint)
            response, structured = _extract_structured_block(raw_response)
            self._log_chat(message, response, status)
            status_updates = [
                f"Mode={mode}",
                f"Hardware={hardware}",
                f"MotionAllowed={bool(allow_motion)}",
            ]
            if model_hint:
                status_updates.append(f"ModelHint={model_hint}")
            if session_id:
                status_updates.append(f"SessionId={session_id}")
            if structured:
                status_updates.append("StructuredPlan=present")

            # Keep fields present even when this lightweight adapter can't compute them yet.
            return {
                "response": response,
                "confidence": None,
                "intervention_needed": False,
                "intervention_question": None,
                "intervention_options": [],
                "status_updates": status_updates,
                "agent": model_hint or "agent_manager",
                "hope_confidence": None,
                "structured": structured,
                # Convenience accessors (optional, best-effort)
                "goals": structured.get("goals") if isinstance(structured, dict) else None,
                "probes": structured.get("probes") if isinstance(structured, dict) else None,
                "plan": structured.get("plan") if isinstance(structured, dict) else None,
            }
        except Exception as exc:  # noqa: BLE001
            print(f"Chat error: {exc}")
            return {"error": str(exc), "response": "", "status_updates": ["ChatAdapterError"]}

    def _build_prompt(self, mode: str, hardware: str, allow_motion: bool) -> str:
        return (
            "You are the Agent Manager for the Continuon Brain Studio (Gemma 3n on-device orchestrator).\n"
            "Responsibilities:\n"
            "- Host the primary on-device model and fall back to other available models when confidence is low or resources are constrained.\n"
            "- Connect and supervise sub-agents/tools to plan, review, and safely execute actions.\n"
            "- Run training/evaluation loops for researchers using local memories and RLDS datasets saved manually or automatically; surface gaps and next actions.\n"
            "- Help end users improve the robot itself: suggest routines, safety checks, and self-improvements that can be applied on-device.\n"
            "- Preserve and use conversation + experience context to incrementally self-improve.\n"
            "\n"
            "Output requirements:\n"
            "- First, write a normal human-readable answer.\n"
            "- Then output EXACTLY ONE additional line with machine-readable JSON, prefixed with:\n"
            f"  {STRUCTURED_PREFIX}\n"
            "- The JSON must be valid and must be an object. Use this shape:\n"
            '  {"goals":[{"id":"g1","text":"...","priority":0.0}],"probes":[{"id":"p1","text":"...","risk":"low"}],'
            '"plan":{"steps":[{"id":"s1","text":"...","requires_approval":false}],"requires_human_approval":false},'
            '"approvals":[{"id":"a1","question":"...","options":["approve","deny"]}]}\n'
            "- If you have no probes/goals/plan, return empty arrays and requires_human_approval=false.\n"
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
        return self._generate_response(message, context, model_hint=model_hint)

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

    def _generate_response(self, message: str, context: str, model_hint: Optional[str] = None) -> str:
        """Generate a lightweight fallback response when Gemma is unavailable."""
        model_tag = f"[model={model_hint}]" if model_hint else "[model=primary]"
        msg_lower = message.lower()

        # Model-specific phrasing variants to avoid identical outputs across hints.
        variants = {
            None: {
                "status": f"{model_tag} Robot status: {context}. The robot is ready for your commands. Use the arrow controls or keyboard to move the arm joints and drive the car.",
                "control": f"{model_tag} Control the arm with joint sliders or arrow buttons. For the car, use the driving controls (default SLOW=0.3). Hold Ctrl+Arrow keys or use WASD for arm control.",
                "joints": f"{model_tag} Arm joints: J0 base, J1 shoulder, J2 elbow, J3 wrist roll, J4 wrist pitch, J5 gripper. Range is -1.0 to 1.0 via sliders/arrow buttons.",
                "car": f"{model_tag} DonkeyCar platform; speed preset SLOW (0.3). Adjust via Crawl/Slow/Med/Fast and steer with arrows/keyboard.",
                "record": f"{model_tag} Recording saves your manual demos as RLDS episodes. Ensure manual_training mode and motion enabled.",
                "safety": f"{model_tag} Safety first: speed preset to SLOW. Emergency Stop halts motion; start with small moves.",
                "default": f"{model_tag} I'm here to help with robot control! Current status: {context}. Ask about controls, status, movement, or safety.",
            },
            "google/gemma-3n-2b": {
                "status": f"{model_tag} Status snapshot → {context}. Controls are live; keyboard/arrow input ready for arm and drive.",
                "control": f"{model_tag} Use sliders/arrow keys for the arm; driving uses the same arrows. Keep speed at SLOW=0.3 unless testing.",
                "joints": f"{model_tag} Six-DOF arm: J0 base, J1 shoulder, J2 elbow, J3 roll, J4 pitch, J5 gripper. Target range [-1,1].",
                "car": f"{model_tag} DonkeyCar RC: default SLOW (0.3). Change speed with preset buttons; steer with arrows/keyboard.",
                "record": f"{model_tag} RLDS capture on: manual_training mode + motion enabled. Your demonstrations are written as episodes.",
                "safety": f"{model_tag} Keep it safe: stay on SLOW, use E-Stop if anything misbehaves. Start with micro-movements.",
                "default": f"{model_tag} Ready for XR control assistance. Status={context}. Ask for controls or safety tips.",
            },
            "google/gemma-370m": {
                "status": f"{model_tag} Current mode/hw → {context}. Controls armed; arrow/keyboard inputs accepted for arm and drive.",
                "control": f"{model_tag} Drive/arm via arrows; sliders for arm joints. Speed preset SLOW(0.3); bump up only after checks.",
                "joints": f"{model_tag} Arm DOFs: base, shoulder, elbow, wrist roll, wrist pitch, gripper. Command range [-1,1] via sliders/arrows.",
                "car": f"{model_tag} DonkeyCar baseline. Speed presets: Crawl/Slow/Med/Fast; steering with arrows/keys. Default SLOW for safety.",
                "record": f"{model_tag} Recording writes RLDS episodes. Enable manual_training + motion to log your demonstrations.",
                "safety": f"{model_tag} Safety: run at SLOW, keep E-Stop handy. Start gentle, then scale up.",
                "default": f"{model_tag} Control helper online. Status={context}. Ask for control/safety guidance.",
            },
        }

        bucket = None
        if model_hint in variants:
            bucket = variants[model_hint]
        else:
            bucket = variants[None]

        def pick(key: str, fallback_key: str = "default") -> str:
            return bucket.get(key) or bucket.get(fallback_key) or variants[None]["default"]

        if any(word in msg_lower for word in ["status", "state", "how", "what"]):
            return pick("status")

        if any(word in msg_lower for word in ["control", "move", "drive", "steer"]):
            return pick("control")

        if any(word in msg_lower for word in ["joint", "arm", "gripper"]):
            return pick("joints")

        if any(word in msg_lower for word in ["car", "speed", "throttle"]):
            return pick("car")

        if any(word in msg_lower for word in ["record", "episode", "training"]):
            return pick("record")

        if any(word in msg_lower for word in ["safe", "stop", "emergency"]):
            return pick("safety")

        return pick("default")

