"""
Conversation Handler - The main interface for talking to your robot.

Ties together intent classification, action execution, and responses.
"""

from typing import Callable
from dataclasses import dataclass

from .intents import Intent, IntentClassifier, ParsedIntent
from ..actor_runtime import ActorRuntime


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
    ):
        self.runtime = runtime
        self.executor = executor
        self.classifier = IntentClassifier(runtime.teaching.list_behaviors())

        self.speed = default_speed
        self.min_speed = 0.1
        self.max_speed = 1.0

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
        if self.runtime.teaching.is_recording:
            return Response(f"I don't understand '{parsed.raw_text}'. Say 'done' to save, or 'cancel' to abort.")
        return Response(f"I don't understand '{parsed.raw_text}'. Say 'help' for commands.")
