"""
Conversation Handler - The main interface for talking to your robot.

Ties together intent classification, action execution, and responses.
Includes Gemini/Claude integration for intelligent conversation.
"""

from typing import Callable, Optional, Union, List
from dataclasses import dataclass

from conversation.intents import Intent, IntentClassifier, ParsedIntent
from conversation.claude_backend import ClaudeBackend
from conversation.gemini_backend import GeminiBackend
from actor_runtime import ActorRuntime

# Tool prediction (trained model)
try:
    from trainer.predictor import ToolPredictorService, Prediction
    HAS_PREDICTOR = True
except ImportError:
    HAS_PREDICTOR = False
    Prediction = None


@dataclass
class Response:
    """A response to the user."""

    text: str
    action_taken: bool = False
    action_type: str | None = None


class ConversationHandler:
    """
    Handles conversation with the robot.

    Usage:
        handler = ConversationHandler(runtime, executor)
        response = handler.handle("drive forward")
        print(response.text)
    """

    def __init__(
        self,
        runtime: ActorRuntime,
        executor: Callable[[dict], None],
        default_speed: float = 0.5,
        use_ai: bool = True,
        prefer_gemini: bool = True,
        model_path: Optional[str] = None,
    ):
        self.runtime = runtime
        self.executor = executor
        self.classifier = IntentClassifier(runtime.teaching.list_behaviors())

        self.speed = default_speed
        self.min_speed = 0.1
        self.max_speed = 1.0

        # Tool history for predictions
        self.tool_history: List[str] = []

        # Tool predictor (trained model)
        self.predictor: Optional[ToolPredictorService] = None
        if HAS_PREDICTOR:
            self.predictor = ToolPredictorService(model_path)
            if self.predictor.is_ready:
                print("[Init] Tool predictor enabled")

        # AI backend for intelligent conversation (Gemini preferred, Claude fallback)
        self.ai_backend: Optional[Union[GeminiBackend, ClaudeBackend]] = None
        if use_ai:
            if prefer_gemini:
                self.ai_backend = GeminiBackend()
                if self.ai_backend.is_available:
                    print("[Init] Gemini integration enabled")
                else:
                    # Fallback to Claude
                    self.ai_backend = ClaudeBackend()
                    if self.ai_backend.is_available:
                        print("[Init] Claude integration enabled (Gemini unavailable)")
            else:
                self.ai_backend = ClaudeBackend()
                if self.ai_backend.is_available:
                    print("[Init] Claude integration enabled")
                else:
                    # Fallback to Gemini
                    self.ai_backend = GeminiBackend()
                    if self.ai_backend.is_available:
                        print("[Init] Gemini integration enabled (Claude unavailable)")

    def handle(self, user_input: str) -> Response:
        """
        Process user input and return a response.

        This is the main entry point for conversation.
        """
        # Update known behaviors
        self.classifier.update_behaviors(self.runtime.teaching.list_behaviors())

        # Classify intent
        parsed = self.classifier.classify(user_input)

        # Handle based on intent
        handler = self._get_handler(parsed.intent)
        return handler(parsed)

    def _get_handler(self, intent: Intent) -> Callable[[ParsedIntent], Response]:
        """Get the handler function for an intent."""
        handlers = {
            # Movement
            Intent.MOVE_FORWARD: self._handle_forward,
            Intent.MOVE_BACKWARD: self._handle_backward,
            Intent.TURN_LEFT: self._handle_left,
            Intent.TURN_RIGHT: self._handle_right,
            Intent.STOP: self._handle_stop,

            # Speed
            Intent.SPEED_UP: self._handle_speed_up,
            Intent.SLOW_DOWN: self._handle_slow_down,
            Intent.SET_SPEED: self._handle_set_speed,

            # Teaching
            Intent.START_TEACHING: self._handle_start_teaching,
            Intent.STOP_TEACHING: self._handle_stop_teaching,
            Intent.CANCEL_TEACHING: self._handle_cancel_teaching,

            # Invoke
            Intent.INVOKE_BEHAVIOR: self._handle_invoke,

            # Memory
            Intent.LIST_BEHAVIORS: self._handle_list,
            Intent.FORGET_BEHAVIOR: self._handle_forget,
            Intent.DESCRIBE_BEHAVIOR: self._handle_describe,

            # System
            Intent.STATUS: self._handle_status,
            Intent.HELP: self._handle_help,
            Intent.QUIT: self._handle_quit,

            # Unknown
            Intent.UNKNOWN: self._handle_unknown,
        }
        return handlers.get(intent, self._handle_unknown)

    # === Movement Handlers ===

    def _handle_forward(self, parsed: ParsedIntent) -> Response:
        speed = self.speed * 0.5 if parsed.params.get("slow") else self.speed
        action = {"type": "forward", "speed": speed}
        self.runtime.execute_action(action, self.executor)

        if self.runtime.teaching.is_recording:
            return Response(f"[Recording] Forward at {int(speed * 100)}%", True, "forward")
        return Response(f"Moving forward at {int(speed * 100)}%.", True, "forward")

    def _handle_backward(self, parsed: ParsedIntent) -> Response:
        speed = self.speed * 0.5 if parsed.params.get("slow") else self.speed
        action = {"type": "backward", "speed": speed}
        self.runtime.execute_action(action, self.executor)

        if self.runtime.teaching.is_recording:
            return Response(f"[Recording] Backward at {int(speed * 100)}%", True, "backward")
        return Response(f"Moving backward at {int(speed * 100)}%.", True, "backward")

    def _handle_left(self, parsed: ParsedIntent) -> Response:
        action = {"type": "left", "speed": self.speed}
        self.runtime.execute_action(action, self.executor)

        if self.runtime.teaching.is_recording:
            return Response("[Recording] Turn left", True, "left")
        return Response("Turning left.", True, "left")

    def _handle_right(self, parsed: ParsedIntent) -> Response:
        action = {"type": "right", "speed": self.speed}
        self.runtime.execute_action(action, self.executor)

        if self.runtime.teaching.is_recording:
            return Response("[Recording] Turn right", True, "right")
        return Response("Turning right.", True, "right")

    def _handle_stop(self, parsed: ParsedIntent) -> Response:
        action = {"type": "stop"}
        self.runtime.execute_action(action, self.executor)
        return Response("Stopped.", True, "stop")

    # === Speed Handlers ===

    def _handle_speed_up(self, parsed: ParsedIntent) -> Response:
        self.speed = min(self.max_speed, self.speed + 0.1)
        return Response(f"Speed increased to {int(self.speed * 100)}%.")

    def _handle_slow_down(self, parsed: ParsedIntent) -> Response:
        self.speed = max(self.min_speed, self.speed - 0.1)
        return Response(f"Speed decreased to {int(self.speed * 100)}%.")

    def _handle_set_speed(self, parsed: ParsedIntent) -> Response:
        try:
            percent = int(parsed.params.get("percent", 50))
            self.speed = max(self.min_speed, min(self.max_speed, percent / 100))
            return Response(f"Speed set to {int(self.speed * 100)}%.")
        except ValueError:
            return Response("Invalid speed. Use a number like 'speed 50'.")

    # === Teaching Handlers ===

    def _handle_start_teaching(self, parsed: ParsedIntent) -> Response:
        name = parsed.params.get("name", "").strip()
        if not name:
            return Response("What should I call this behavior? Say 'teach <name>'.")

        result = self.runtime.teach(name)
        return Response(result)

    def _handle_stop_teaching(self, parsed: ParsedIntent) -> Response:
        if not self.runtime.teaching.is_recording:
            return Response("Not currently recording.")
        result = self.runtime.done_teaching()
        return Response(result)

    def _handle_cancel_teaching(self, parsed: ParsedIntent) -> Response:
        if not self.runtime.teaching.is_recording:
            return Response("Not currently recording.")
        result = self.runtime.teaching.cancel_recording()
        return Response(result)

    # === Invoke Handler ===

    def _handle_invoke(self, parsed: ParsedIntent) -> Response:
        name = parsed.params.get("name", "").strip()
        if not name:
            return Response("What behavior should I run?")

        def on_step(step: int, total: int, action: dict):
            print(f"  [{step}/{total}] {action.get('type', '?')}")

        result = self.runtime.invoke(name, self.executor, on_step)
        return Response(result, True, "invoke")

    # === Memory Handlers ===

    def _handle_list(self, parsed: ParsedIntent) -> Response:
        behaviors = self.runtime.teaching.list_behaviors()
        if not behaviors:
            return Response("I haven't learned any behaviors yet. Teach me with 'teach <name>'.")
        return Response(f"I know: {', '.join(behaviors)}")

    def _handle_forget(self, parsed: ParsedIntent) -> Response:
        name = parsed.params.get("name", "").strip()
        if not name:
            return Response("What should I forget?")
        result = self.runtime.teaching.forget(name)
        return Response(result)

    def _handle_describe(self, parsed: ParsedIntent) -> Response:
        name = parsed.params.get("name", "").strip()
        if not name:
            return Response("What behavior should I describe?")
        result = self.runtime.teaching.describe(name)
        return Response(result)

    # === System Handlers ===

    def _handle_status(self, parsed: ParsedIntent) -> Response:
        status = self.runtime.status()
        lines = [
            f"Speed: {int(self.speed * 100)}%",
            f"Behaviors: {status['behaviors']}",
            f"Recording: {status['recording_name'] or 'No'}",
            f"Uptime: {int(status['uptime_s'])}s",
        ]
        return Response("\n".join(lines))

    def _handle_help(self, parsed: ParsedIntent) -> Response:
        help_text = """Commands:
  forward, back, left, right, stop
  faster, slower, speed <0-100>
  teach <name> - start recording
  done - save recording
  <name> - run learned behavior
  list - show behaviors
  forget <name> - delete behavior
  status - show status
  quit - exit"""
        return Response(help_text)

    def _handle_quit(self, parsed: ParsedIntent) -> Response:
        self.runtime.shutdown()
        return Response("Goodbye!")

    def _handle_unknown(self, parsed: ParsedIntent) -> Response:
        # If recording, keep strict command mode
        if self.runtime.teaching.is_recording:
            return Response(f"I don't understand '{parsed.raw_text}'. Say 'done' to save, or 'cancel' to abort.")

        # Try AI backend (Gemini or Claude) for intelligent conversation
        if self.ai_backend and self.ai_backend.is_available:
            response_text, action = self.ai_backend.process(
                parsed.raw_text,
                speed=self.speed,
                behaviors=self.runtime.teaching.list_behaviors(),
                recording=self.runtime.teaching.recording_name,
            )

            # If AI suggested an action, execute it
            if action:
                action_type = action.get("type")
                if action_type in ("forward", "backward", "left", "right", "stop"):
                    self.runtime.execute_action(action, self.executor)
                    return Response(response_text, True, action_type)
                elif action_type == "teach":
                    result = self.runtime.teach(action.get("name", ""))
                    return Response(f"{response_text}\n{result}")
                elif action_type == "invoke":
                    def on_step(step: int, total: int, action: dict):
                        print(f"  [{step}/{total}] {action.get('type', '?')}")
                    result = self.runtime.invoke(action.get("name", ""), self.executor, on_step)
                    return Response(f"{response_text}\n{result}", True, "invoke")

            return Response(response_text)

        return Response(f"I don't understand '{parsed.raw_text}'. Say 'help' for commands.")

    # === Tool Prediction ===

    def predict_tool(self, task: str = "") -> Optional[dict]:
        """
        Predict the next tool based on task and history.

        Args:
            task: Optional task description for context

        Returns:
            Dictionary with prediction info or None if predictor unavailable
        """
        if not self.predictor or not self.predictor.is_ready:
            return None

        if task:
            pred = self.predictor.predict_for_task(task, self.tool_history)
        else:
            context = {
                "current_tool": "",
                "prev_tool": self.tool_history[-1] if self.tool_history else "",
                "prev_success": True,
                "step_idx": len(self.tool_history),
            }
            pred = self.predictor.predict(context)

        if pred:
            return {
                "tool": pred.tool,
                "confidence": pred.confidence,
                "alternatives": [{"tool": t, "confidence": c} for t, c in pred.alternatives],
            }
        return None

    def record_tool_use(self, tool: str, success: bool = True):
        """Record a tool use for future predictions."""
        self.tool_history.append(tool)
        # Keep only last 20 tools
        if len(self.tool_history) > 20:
            self.tool_history = self.tool_history[-20:]

    def get_tool_suggestions(self, count: int = 3) -> List[dict]:
        """Get top tool suggestions based on current context."""
        pred = self.predict_tool()
        if not pred:
            return []

        suggestions = [{"tool": pred["tool"], "confidence": pred["confidence"]}]
        suggestions.extend(pred["alternatives"][:count - 1])
        return suggestions
