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
        on_chat_event: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self.config_dir = config_dir
        self.status_provider = status_provider
        self.gemma_chat = gemma_chat
        self.on_chat_event = on_chat_event
        
        # Optional Hailo vision offload (subprocess-safe). Used only when an image is attached.
        self._hailo_vision = None
        self._hailo_vision_state_cache: Optional[dict] = None

    def get_hailo_state(self) -> Optional[dict]:
        """Expose current Hailo vision status for architecture/status endpoints."""
        try:
            if self._hailo_vision is None:
                from continuonbrain.services.hailo_vision import HailoVision

                self._hailo_vision = HailoVision()
            self._hailo_vision_state_cache = self._hailo_vision.get_state()
            return self._hailo_vision_state_cache
        except Exception:
            return self._hailo_vision_state_cache

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
        print(f"[ChatAdapter] chat() called with model_hint={model_hint} delegate={delegate_model_hint}")
        # Track current session for fallback detection
        self._current_session_id = session_id

        # Auto-detect delegate hint if passed in model_hint
        if model_hint and "consult:" in model_hint and not delegate_model_hint:
             delegate_model_hint = model_hint
        
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
                # Optional Hailo vision summary (never fatal).
                hailo_summary = None
                hailo_payload: Dict[str, Any] = {}
                try:
                    if self._hailo_vision is None:
                        from continuonbrain.services.hailo_vision import HailoVision

                        self._hailo_vision = HailoVision()
                    res = self._hailo_vision.infer_jpeg(image_jpeg)
                    self._hailo_vision_state_cache = self._hailo_vision.get_state()
                    hailo_payload = res if isinstance(res, dict) else {}
                    if hailo_payload.get("ok") and isinstance(hailo_payload.get("topk"), list):
                        tops = hailo_payload["topk"][:5]
                        idxs = ",".join(str(t.get("index")) for t in tops)
                        hailo_summary = f"hailo_topk_indices=[{idxs}]"
                    elif hailo_payload.get("error"):
                        hailo_summary = f"hailo_error={str(hailo_payload.get('error'))[:160]}"
                except Exception as exc:  # noqa: BLE001
                    hailo_summary = f"hailo_error={str(exc)[:160]}"

                prompt = (
                    prompt
                    + "\n\nVision:\n"
                    + f"- attached: true\n- source: {image_source or 'unknown'}\n"
                    + "- instruction: Use the image to ground your answer. If uncertain, ask a follow-up.\n"
                )
                if hailo_summary:
                    prompt = prompt + f"- hailo: {hailo_summary}\n"
            elif vision_requested:
                prompt = (
                    prompt
                    + "\n\nVision:\n"
                    + f"- attached: false\n- source: {image_source or 'unknown'}\n"
                    + "- note: Vision was requested but no frame was available.\n"
                )

            # Optional: consult a sub-agent (e.g., Gemma 3n) and feed its answer back into the Agent Manager.
            # This is purely advisory: no auto-execution.
            # 2. If 'consult:' prefix, we do a sub-agent "thought" turn first
            if delegate_model_hint and "consult:" in delegate_model_hint:
                print(f"[ChatAdapter] Consulting subagent: {delegate_model_hint}")
                if self.gemma_chat:
                    try:
                        # Parse out the sub-model name if needed, or just ask gemma
                        # We'll just pass the user text to the sub-agent for now
                        sub_text = self.gemma_chat.chat(
                            message=message,
                            system_context=prompt, # Pass the full prompt for context
                            model_hint=delegate_model_hint.replace("consult:", "").strip() # Extract model hint
                        )
                        
                        print(f"[ChatAdapter] Subagent response len: {len(sub_text)}")
                        if self.on_chat_event:
                            print(f"[ChatAdapter] Emitting subagent event")
                            try:
                                self.on_chat_event({
                                    "role": "subagent",
                                    "name": "Gemma 3n",  # UI label
                                    "text": sub_text[:2000],
                                    "model": delegate_model_hint,
                                    "timestamp": datetime.datetime.now().isoformat(),
                                })
                            except Exception as e:
                                print(f"[ChatAdapter] Error emitting event: {e}")
                        
                        # Feed the subagent answer into the system prompt so the Agent Manager can learn/incorporate.
                        prompt = (
                            prompt
                            + "\n\nSub-agent consult (advisory):\n"
                            + f"- model_hint: {delegate_model_hint}\n"
                            + f"- response: {sub_text[:2000]}\n"
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Non-fatal.
                        print(f"[ChatAdapter] Subagent consult failed: {exc}")
                        pass # Continue without subagent info
                else:
                    print(f"[ChatAdapter] Subagent consult skipped: gemma_chat not available")

            subagent_info: Dict[str, Any] = {}
            if delegate_model_hint and "consult:" not in delegate_model_hint: # Only process if not already handled by "consult:"
                try:
                    sub_raw = self._call_model(message, prompt, history, model_hint=delegate_model_hint, image=image_obj)
                    sub_text, _sub_structured = _extract_structured_block(sub_raw)
                    subagent_info = {
                        "model_hint": delegate_model_hint,
                        "response": sub_text[:2000],
                    }
                    if self.on_chat_event:
                        try:
                            self.on_chat_event({
                                "role": "subagent",
                                "name": "Gemma 3n",  # UI label
                                "text": sub_text[:2000],
                                "model": delegate_model_hint,
                                "timestamp": datetime.datetime.now().isoformat(),
                            })
                        except Exception:
                            pass

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

            # Call model and capture raw response
            raw_response = self._call_model(message, prompt, history, model_hint=model_hint, image=image_obj)
            
            # Check if we got a fallback response
            is_fallback = (
                raw_response and (
                    "Status snapshot" in raw_response or
                    "Robot status:" in raw_response or
                    "Ready for XR" in raw_response or
                    (raw_response.startswith("[model=") and len(raw_response) < 200 and "Status" in raw_response)
                )
            )
            
            # If fallback and this is chat learning, try to get actual model response
            if is_fallback and session_id and "chat_learn" in str(session_id):
                # Try to force model initialization if available
                if not self.gemma_chat:
                    try:
                        # Central chat backend builder (may return None when disabled).
                        from continuonbrain.gemma_chat import build_chat_service
                        self.gemma_chat = build_chat_service()
                        if self.gemma_chat and hasattr(self.gemma_chat, 'model') and self.gemma_chat.model is not None:
                            # Retry with actual model
                            raw_response = self._call_model(message, prompt, history, model_hint=model_hint, image=image_obj)
                            is_fallback = False  # Reset if we got a real response
                    except Exception:
                        pass
            
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
                # Attach hailo vision metadata when available (no raw pixels, bounded payload).
                try:
                    if image_jpeg and self._hailo_vision is not None:
                        hs = self._hailo_vision.get_state()
                        structured["vision"]["hailo"] = {
                            "enabled": bool(hs.get("enabled")) if isinstance(hs, dict) else None,
                            "available": bool(hs.get("available")) if isinstance(hs, dict) else None,
                            "hef_path": hs.get("hef_path") if isinstance(hs, dict) else None,
                            "last_ok": hs.get("last_ok") if isinstance(hs, dict) else None,
                            "last_error": hs.get("last_error") if isinstance(hs, dict) else None,
                        }
                except Exception:
                    pass
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
            
            if self.on_chat_event:
                print(f"[ChatAdapter] Emitting agent_manager event")
                try:
                    self.on_chat_event({
                        "role": "agent_manager",
                        "name": "Agent Manager",
                        "text": response,
                        "model": model_hint or "primary",
                        "timestamp": datetime.datetime.now().isoformat(),
                    })
                except Exception as e:
                     print(f"[ChatAdapter] Error emitting agent_manager event: {e}")
            
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
        # Try to use actual model first
        if self.gemma_chat:
            try:
                response = self.gemma_chat.chat(message, system_context=prompt, image=image, model_hint=model_hint)
                # Verify we got a real response, not an error
                if response and "Error" not in response and "failed" not in response.lower():
                    return response
                # If we got an error, fall through to fallback
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(f"Model call failed, using fallback: {exc}")
                # Fall through to fallback

        # Fallback: generate response when model unavailable
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
        fallback_response = self._generate_response(message, context, model_hint=model_hint)
        
        # Log warning if we're using fallback for chat learning
        if "chat_learn" in str(getattr(self, '_current_session_id', '')):
            import logging
            logging.getLogger(__name__).warning(
                f"Using fallback response for chat learning. Model unavailable. "
                f"Response will not be logged to RLDS to avoid polluting training data."
            )
        
        return fallback_response

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
            enabled_env = os.environ.get("CONTINUON_LOG_CHAT_RLDS", "0").lower() in ("1", "true", "yes", "on")
            enabled_settings = False
            try:
                from continuonbrain.settings_manager import SettingsStore

                settings = SettingsStore(Path(self.config_dir)).load()
                enabled_settings = bool(((settings or {}).get("chat") or {}).get("log_rlds", False))
            except Exception:
                enabled_settings = False
            enabled = bool(enabled_env or enabled_settings)
            if not enabled:
                return
            
            # Detect if this is a fallback response (generic status message)
            is_fallback_response = (
                response and (
                    "Status snapshot" in response or
                    "Robot status:" in response or
                    "Ready for XR" in response or
                    (response.startswith("[model=") and len(response) < 200 and "Status" in response)
                )
            )
            
            # For chat learning sessions, we want actual conversation, not fallback responses
            # Skip logging fallback responses for chat learning to avoid polluting training data
            if is_fallback_response and session_id and "chat_learn" in str(session_id):
                import logging
                logging.getLogger(__name__).warning(
                    f"Skipping RLDS log for fallback response in chat learning session {session_id}. "
                    f"Model may not be available. Enable model or fix chat adapter."
                )
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
            "hope-v1": {
                "status": f"{model_tag} [HOPE-V1 Primary] System nominal. Status: {context}. Ready for complex orchestrations.",
                "control": f"{model_tag} [HOPE-V1] Precision control active. Use sliders or keyboard. Speed governed at SLOW (0.3).",
                "joints": f"{model_tag} [HOPE-V1] 6-DOF Manipulator available. Base/Shoulder/Elbow/Wrist/Gripper. Range [-1.0, 1.0].",
                "car": f"{model_tag} [HOPE-V1] Mobility platform online. Presets: Crawl/Slow/Med/Fast. Default: SLOW.",
                "record": f"{model_tag} [HOPE-V1] Episode logging enabled in manual_training mode. Demonstrations are valuable.",
                "safety": f"{model_tag} [HOPE-V1] Safety protocols engaged. E-Stop monitoring active. Proceed with caution.",
                "default": f"{model_tag} [HOPE-V1] Agent Manager online. I am HOPE v1, ready to assist with robot operations and learning. Status={context}.",
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

