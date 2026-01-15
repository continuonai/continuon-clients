"""
Game Handler - Extends conversation handler with grid game commands.

Provides:
- Game-specific intents (look, grab, where, reset, levels)
- Integration with GridWorld for movement
- Teaching behaviors that work in the grid context
- State queries and visualization
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Optional
import re

from simulator.world import GridWorld, MoveResult, load_level, LEVELS
from actor_runtime import ActorRuntime


class GameIntent(Enum):
    """Game-specific intents beyond base robot commands."""
    # Movement (maps to base)
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()

    # Game-specific
    LOOK = auto()           # What's ahead?
    WHERE = auto()          # Where am I?
    INVENTORY = auto()      # What do I have?
    GRAB = auto()           # Pick up item
    USE = auto()            # Use item

    # Level management
    LOAD_LEVEL = auto()
    LIST_LEVELS = auto()
    RESET_LEVEL = auto()

    # Teaching (maps to base)
    START_TEACHING = auto()
    STOP_TEACHING = auto()
    CANCEL_TEACHING = auto()
    INVOKE_BEHAVIOR = auto()

    # Memory
    LIST_BEHAVIORS = auto()
    FORGET_BEHAVIOR = auto()

    # System
    STATUS = auto()
    HELP = auto()

    # Unknown
    UNKNOWN = auto()


@dataclass
class GameResponse:
    """Response from game handler."""
    text: str
    action_taken: bool = False
    action_type: Optional[str] = None
    sandbox_denied: bool = False
    level_complete: bool = False
    world_state: Optional[dict] = None


class GameIntentClassifier:
    """Classify user input into game intents."""

    PATTERNS = [
        # Help/system
        (r"^(help|\?)$", GameIntent.HELP, {}),
        (r"^(status|info)$", GameIntent.STATUS, {}),

        # Level management
        (r"^(load|start|play)\s+level\s+(\w+)$", GameIntent.LOAD_LEVEL, {"level": 2}),
        (r"^(load|start|play)\s+(\w+)$", GameIntent.LOAD_LEVEL, {"level": 2}),
        (r"^(levels?|list\s+levels?)$", GameIntent.LIST_LEVELS, {}),
        (r"^(reset|restart)(\s+level)?$", GameIntent.RESET_LEVEL, {}),

        # Teaching
        (r"^(teach|learn|record)\s+(.+)$", GameIntent.START_TEACHING, {"name": 2}),
        (r"^(done|finished|save|end)$", GameIntent.STOP_TEACHING, {}),
        (r"^(cancel|abort|nevermind)$", GameIntent.CANCEL_TEACHING, {}),

        # Memory
        (r"^(list|show)\s*(behaviors?|skills?|macros?)$", GameIntent.LIST_BEHAVIORS, {}),
        (r"^(forget|delete)\s+(.+)$", GameIntent.FORGET_BEHAVIOR, {"name": 2}),

        # Game-specific queries
        (r"^(look|see|scan|ahead)$", GameIntent.LOOK, {}),
        (r"^(where|position|location)(\s+am\s+i)?$", GameIntent.WHERE, {}),
        (r"^(inventory|items?|bag|what\s+do\s+i\s+have)$", GameIntent.INVENTORY, {}),
        (r"^(grab|pick\s*up|take|collect)(\s+(.+))?$", GameIntent.GRAB, {"item": 3}),
        (r"^use\s+(.+)$", GameIntent.USE, {"item": 1}),

        # Movement
        (r"^(go\s+)?(forward|ahead|straight|f)$", GameIntent.MOVE_FORWARD, {}),
        (r"^(go\s+)?(back|backward|reverse|b)$", GameIntent.MOVE_BACKWARD, {}),
        (r"^(turn\s+)?left|l$", GameIntent.TURN_LEFT, {}),
        (r"^(turn\s+)?right|r$", GameIntent.TURN_RIGHT, {}),

        # Cardinal directions (converted to turn+forward)
        (r"^(go\s+)?north|n$", GameIntent.MOVE_FORWARD, {"cardinal": "north"}),
        (r"^(go\s+)?south|s$", GameIntent.MOVE_FORWARD, {"cardinal": "south"}),
        (r"^(go\s+)?east|e$", GameIntent.MOVE_FORWARD, {"cardinal": "east"}),
        (r"^(go\s+)?west|w$", GameIntent.MOVE_FORWARD, {"cardinal": "west"}),
    ]

    def __init__(self, behaviors: Optional[list[str]] = None):
        self.behaviors = behaviors or []

    def update_behaviors(self, behaviors: list[str]) -> None:
        self.behaviors = behaviors

    def classify(self, text: str) -> tuple[GameIntent, dict]:
        """Classify text into intent and params."""
        text = text.strip().lower()

        if not text:
            return GameIntent.UNKNOWN, {}

        for pattern, intent, param_groups in self.PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                params = {}
                for name, group_num in param_groups.items():
                    if isinstance(group_num, int) and group_num <= len(match.groups()):
                        value = match.group(group_num)
                        if value:
                            params[name] = value.strip()
                return intent, params

        # Check for behavior invocation
        for behavior in self.behaviors:
            if text == behavior.lower() or text == f"do {behavior.lower()}" or text == f"run {behavior.lower()}":
                return GameIntent.INVOKE_BEHAVIOR, {"name": behavior}

        return GameIntent.UNKNOWN, {}


class GameHandler:
    """
    Handles game commands and integrates with Brain B systems.

    Usage:
        world = load_level("tutorial")
        handler = GameHandler(world, runtime)
        response = handler.handle("forward")
    """

    def __init__(
        self,
        world: GridWorld,
        runtime: ActorRuntime,
        on_state_change: Optional[Callable[[dict], None]] = None,
    ):
        self.world = world
        self.runtime = runtime
        self.classifier = GameIntentClassifier(runtime.teaching.list_behaviors())
        self.on_state_change = on_state_change

    def handle(self, user_input: str) -> GameResponse:
        """Process user input and return response."""
        self.classifier.update_behaviors(self.runtime.teaching.list_behaviors())
        intent, params = self.classifier.classify(user_input)

        # Get handler method
        handler = getattr(self, f"_handle_{intent.name.lower()}", self._handle_unknown)
        response = handler(params, user_input)

        # Notify state change if handler provided
        if self.on_state_change and response.action_taken:
            response.world_state = self.world.to_dict()
            self.on_state_change(response.world_state)

        return response

    def _create_executor(self) -> Callable[[dict], None]:
        """Create executor function for teaching system."""
        def executor(action: dict) -> None:
            action_type = action.get("type", "")
            if action_type == "forward":
                self.world.move_forward()
            elif action_type == "backward":
                self.world.move_backward()
            elif action_type == "left":
                self.world.turn_left()
            elif action_type == "right":
                self.world.turn_right()
        return executor

    # === Movement Handlers ===

    def _handle_move_forward(self, params: dict, raw: str) -> GameResponse:
        result = self.world.move_forward()

        # Record for teaching if active
        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "forward"})

        # Log event
        self.runtime.execute_action(
            {"type": "forward", "game_result": result.message},
            lambda a: None  # No-op executor, game handles movement
        )

        prefix = "[Recording] " if self.runtime.teaching.is_recording else ""
        return GameResponse(
            text=f"{prefix}{result.message}",
            action_taken=result.success,
            action_type="forward",
            sandbox_denied=result.sandbox_denied,
            level_complete=result.level_complete,
        )

    def _handle_move_backward(self, params: dict, raw: str) -> GameResponse:
        result = self.world.move_backward()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "backward"})

        self.runtime.execute_action(
            {"type": "backward", "game_result": result.message},
            lambda a: None
        )

        prefix = "[Recording] " if self.runtime.teaching.is_recording else ""
        return GameResponse(
            text=f"{prefix}{result.message}",
            action_taken=result.success,
            action_type="backward",
            sandbox_denied=result.sandbox_denied,
            level_complete=result.level_complete,
        )

    def _handle_turn_left(self, params: dict, raw: str) -> GameResponse:
        result = self.world.turn_left()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "left"})

        self.runtime.execute_action(
            {"type": "left", "game_result": result.message},
            lambda a: None
        )

        prefix = "[Recording] " if self.runtime.teaching.is_recording else ""
        return GameResponse(
            text=f"{prefix}{result.message}",
            action_taken=True,
            action_type="left",
        )

    def _handle_turn_right(self, params: dict, raw: str) -> GameResponse:
        result = self.world.turn_right()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "right"})

        self.runtime.execute_action(
            {"type": "right", "game_result": result.message},
            lambda a: None
        )

        prefix = "[Recording] " if self.runtime.teaching.is_recording else ""
        return GameResponse(
            text=f"{prefix}{result.message}",
            action_taken=True,
            action_type="right",
        )

    # === Game Query Handlers ===

    def _handle_look(self, params: dict, raw: str) -> GameResponse:
        return GameResponse(text=self.world.look())

    def _handle_where(self, params: dict, raw: str) -> GameResponse:
        return GameResponse(text=self.world.where_am_i())

    def _handle_inventory(self, params: dict, raw: str) -> GameResponse:
        inv = self.world.robot.inventory
        if inv:
            return GameResponse(text=f"You have: {', '.join(inv)}")
        return GameResponse(text="Your inventory is empty.")

    def _handle_grab(self, params: dict, raw: str) -> GameResponse:
        # Grabbing happens automatically when moving onto key tiles
        return GameResponse(text="Move onto items to collect them automatically.")

    def _handle_use(self, params: dict, raw: str) -> GameResponse:
        item = params.get("item", "")
        if item == "key":
            return GameResponse(text="Walk into a door to use your key.")
        return GameResponse(text=f"I don't know how to use '{item}'.")

    # === Level Management ===

    def _handle_load_level(self, params: dict, raw: str) -> GameResponse:
        level_id = params.get("level", "tutorial")
        try:
            self.world = load_level(level_id)
            return GameResponse(
                text=f"Loaded level: {LEVELS[level_id]['name']}\n{LEVELS[level_id]['description']}\n\n{self.world.render_with_border()}",
                action_taken=True,
                action_type="load_level",
            )
        except ValueError as e:
            return GameResponse(text=str(e))

    def _handle_list_levels(self, params: dict, raw: str) -> GameResponse:
        lines = ["Available levels:"]
        for level_id, level in LEVELS.items():
            lines.append(f"  {level_id}: {level['name']} - {level['description']}")
        return GameResponse(text="\n".join(lines))

    def _handle_reset_level(self, params: dict, raw: str) -> GameResponse:
        self.world.reset()
        return GameResponse(
            text=f"Level reset.\n\n{self.world.render_with_border()}",
            action_taken=True,
            action_type="reset",
        )

    # === Teaching Handlers ===

    def _handle_start_teaching(self, params: dict, raw: str) -> GameResponse:
        name = params.get("name", "").strip()
        if not name:
            return GameResponse(text="What should I call this behavior? Say 'teach <name>'.")
        result = self.runtime.teach(name)
        return GameResponse(text=result)

    def _handle_stop_teaching(self, params: dict, raw: str) -> GameResponse:
        if not self.runtime.teaching.is_recording:
            return GameResponse(text="Not currently recording.")
        result = self.runtime.done_teaching()
        return GameResponse(text=result, action_taken=True)

    def _handle_cancel_teaching(self, params: dict, raw: str) -> GameResponse:
        if not self.runtime.teaching.is_recording:
            return GameResponse(text="Not currently recording.")
        result = self.runtime.teaching.cancel_recording()
        return GameResponse(text=result)

    def _handle_invoke_behavior(self, params: dict, raw: str) -> GameResponse:
        name = params.get("name", "").strip()
        if not name:
            return GameResponse(text="What behavior should I run?")

        executor = self._create_executor()

        def on_step(step: int, total: int, action: dict):
            # Could emit websocket events here
            pass

        result = self.runtime.invoke(name, executor, on_step)
        return GameResponse(text=result, action_taken=True, action_type="invoke")

    # === Memory Handlers ===

    def _handle_list_behaviors(self, params: dict, raw: str) -> GameResponse:
        behaviors = self.runtime.teaching.list_behaviors()
        if not behaviors:
            return GameResponse(text="No behaviors learned yet. Use 'teach <name>' to start.")
        return GameResponse(text=f"Known behaviors: {', '.join(behaviors)}")

    def _handle_forget_behavior(self, params: dict, raw: str) -> GameResponse:
        name = params.get("name", "").strip()
        if not name:
            return GameResponse(text="What should I forget?")
        result = self.runtime.teaching.forget(name)
        return GameResponse(text=result)

    # === System Handlers ===

    def _handle_status(self, params: dict, raw: str) -> GameResponse:
        status = self.runtime.status()
        lines = [
            f"Level: {self.world.level_name}",
            f"Robot: ({self.world.robot.x}, {self.world.robot.y}) facing {self.world.robot.direction.name}",
            f"Moves: {self.world.robot.moves}",
            f"Behaviors: {status['behaviors']}",
            f"Recording: {status['recording_name'] or 'No'}",
        ]
        return GameResponse(text="\n".join(lines))

    def _handle_help(self, params: dict, raw: str) -> GameResponse:
        help_text = """RobotGrid Commands:

Movement:
  forward (f), backward (b), left (l), right (r)
  north (n), south (s), east (e), west (w)

Queries:
  look      - What's ahead?
  where     - Current position
  inventory - What do I have?
  status    - Game status

Levels:
  levels         - List available levels
  load <name>    - Load a level
  reset          - Restart current level

Teaching:
  teach <name>   - Start recording behavior
  done           - Save recording
  cancel         - Cancel recording
  <name>         - Run learned behavior
  list behaviors - Show learned behaviors
  forget <name>  - Delete behavior

Goal: Navigate to G while avoiding ~ (lava).
      Collect K (keys) to open D (doors).
      Push B (boxes) onto O (buttons)."""
        return GameResponse(text=help_text)

    def _handle_unknown(self, params: dict, raw: str) -> GameResponse:
        if self.runtime.teaching.is_recording:
            return GameResponse(text=f"Unknown command: '{raw}'. Say 'done' to save or 'cancel' to abort.")
        return GameResponse(text=f"Unknown command: '{raw}'. Say 'help' for commands.")
