import datetime
import io
import json
import os
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
        delegate_model_hint: Optional[str] = None,
        image_jpeg: Optional[bytes] = None,
        image_source: Optional[str] = None,
        vision_requested: bool = False,
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
            image_obj: Any = None
            if image_jpeg:
                # Best-effort decode for VLMs (Gemma 3n VLM expects PIL or numpy-like image).
                try:
                    from PIL import Image  # type: ignore

                    image_obj = Image.open(io.BytesIO(image_jpeg)).convert("RGB")
                except Exception:
                    image_obj = None
                prompt = (
                    prompt
                    + "\n\nVision:\n"
                    + f"- attached: true\n- source: {image_source or 'unknown'}\n"
                    + "- instruction: Use the image to ground your answer. If uncertain, ask a follow-up.\n"
                )
            elif vision_requested:
                prompt = (
                    prompt
                    + "\n\nVision:\n"
                    + f"- attached: false\n- source: {image_source or 'unknown'}\n"
                    + "- note: Vision was requested but no frame was available.\n"
                )

            # Optional: consult a sub-agent (e.g., Gemma 3n) and feed its answer back into the Agent Manager.
            # This is purely advisory: no auto-execution.
            subagent_info: Dict[str, Any] = {}
            if delegate_model_hint:
                try:
                    sub_raw = self._call_model(message, prompt, history, model_hint=delegate_model_hint, image=image_obj)
                    sub_text, _sub_structured = _extract_structured_block(sub_raw)
                    subagent_info = {
                        "model_hint": delegate_model_hint,
                        "response": sub_text[:2000],
                    }
                    # Feed the subagent answer into the system prompt so the Agent Manager can learn/incorporate.
                    prompt = (
                        prompt
                        + "\n\nSub-agent consult (advisory):\n"
                        + f"- model_hint: {delegate_model_hint}\n"
                        + f"- response: {sub_text[:2000]}\n"
                    )
                except Exception as exc:  # noqa: BLE001
                    # Non-fatal.
                    subagent_info = {"model_hint": delegate_model_hint, "error": str(exc)}

            raw_response = self._call_model(message, prompt, history, model_hint=model_hint, image=image_obj)
            response, structured = _extract_structured_block(raw_response)
            if delegate_model_hint:
                if not isinstance(structured, dict):
                    structured = {}
                structured["subagent"] = subagent_info
            if image_jpeg or vision_requested:
                if not isinstance(structured, dict):
                    structured = {}
                structured["vision"] = {
                    "attached": bool(image_jpeg),
                    "source": image_source or "unknown",
                    "bytes": int(len(image_jpeg)) if image_jpeg else 0,
                    "requested": bool(vision_requested),
                }
                # Also surface this in RLDS logs via structured payload (no raw pixels).
                # If you later want pixel provenance, use /api/camera/frame URIs + timestamps.
            # Optional: attach tool-router suggestions for learning (no auto-exec).
            try:
                enable_suggest = os.environ.get("CONTINUON_ENABLE_TOOL_ROUTER_SUGGEST", "1").lower() in ("1", "true", "yes", "on")
                if enable_suggest:
                    from continuonbrain.jax_models.infer.tool_router_infer import load_tool_router_bundle, predict_topk

                    export_dir = Path("/opt/continuonos/brain/model/adapters/candidate/tool_router_seed")
                    if export_dir.exists():
                        bundle = load_tool_router_bundle(export_dir)
                        preds = predict_topk(bundle, message, k=5)
                        if not isinstance(structured, dict):
                            structured = {}
                        structured["suggested_tools"] = preds
            except Exception:
                pass
            self._log_chat(message, response, status)
            self._maybe_log_chat_rlds(
                message=message,
                response=response,
                structured=structured,
                status=status,
                session_id=session_id,
                model_hint=model_hint,
            )
            status_updates = [
                f"Mode={mode}",
                f"Hardware={hardware}",
                f"MotionAllowed={bool(allow_motion)}",
            ]
            if model_hint:
                status_updates.append(f"ModelHint={model_hint}")
            if delegate_model_hint:
                status_updates.append(f"SubagentHint={delegate_model_hint}")
            if image_jpeg:
                status_updates.append(f"Vision={image_source or 'attached'}")
            elif vision_requested:
                status_updates.append("Vision=unavailable")
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
        creator_line = ""
        try:
            from continuonbrain.settings_manager import SettingsStore

            settings = SettingsStore(Path(self.config_dir)).load()
            creator = (
                ((settings or {}).get("identity") or {}).get("creator_display_name")  # type: ignore[union-attr]
                or ""
            )
            creator = str(creator).strip()
            if creator:
                creator_line = f"\nCreator alignment:\n- creator_display_name: {creator}\n"
        except Exception:
            creator_line = ""
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
            f"Current Status:\n- Mode: {mode}\n- Hardware: {hardware}\n- Motion Allowed: {allow_motion}\n"
            + creator_line
            + "\n"
            "Always answer as the Agent Manager. State when you are using the primary model, a fallback, or a sub-agent/tool. Be concise, technical, and action-oriented."
        )

    def _call_model(self, message: str, prompt: str, _history: list, model_hint: Optional[str] = None, image: Any = None) -> str:
        if self.gemma_chat:
            return self.gemma_chat.chat(message, system_context=prompt, image=image, model_hint=model_hint)

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

    def _maybe_log_chat_rlds(
        self,
        *,
        message: str,
        response: str,
        structured: Dict[str, Any],
        status: Dict[str, Any],
        session_id: Optional[str],
        model_hint: Optional[str],
    ) -> None:
        """
        Optional: log chat turns as RLDS episodes for training/eval replay.

        This is OFF by default (offline-first + explicit consent).
        Enable with:
          CONTINUON_LOG_CHAT_RLDS=1
        """
        try:
            enabled = os.environ.get("CONTINUON_LOG_CHAT_RLDS", "0").lower() in ("1", "true", "yes", "on")
            if not enabled:
                return
            from continuonbrain.rlds.chat_rlds_logger import ChatRldsLogConfig, log_chat_turn

            cfg_dir = Path(self.config_dir)
            episodes_dir = Path(os.environ.get("CONTINUON_CHAT_RLDS_DIR") or (cfg_dir / "rlds" / "episodes"))
            cfg = ChatRldsLogConfig(episodes_dir=episodes_dir, group_by_session=True)
            log_chat_turn(
                cfg,
                user_message=message,
                assistant_response=response,
                structured=structured if isinstance(structured, dict) else {},
                status_context=status if isinstance(status, dict) else {},
                session_id=session_id,
                model_hint=model_hint,
                agent_label=model_hint or "agent_manager",
            )
        except Exception as exc:  # noqa: BLE001
            # Never block chat on logging failures.
            print(f"Failed to log chat to RLDS: {exc}")

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

