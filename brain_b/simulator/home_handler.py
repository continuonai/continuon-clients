"""
3D Home Game Handler - Extends conversation handler with 3D home exploration commands.

Provides:
- 3D movement intents (forward, backward, strafe, turn, look up/down)
- Interaction with home objects (doors, switches, drawers)
- Room navigation and object visibility
- Teaching behaviors in 3D context
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Optional, List
import re

from simulator.home_world import (
    HomeWorld,
    MoveResult3D,
    get_level,
    list_levels,
    LEVELS,
)
from actor_runtime import ActorRuntime


class Home3DIntent(Enum):
    """3D Home game intents."""
    # Movement
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    STRAFE_LEFT = auto()
    STRAFE_RIGHT = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    LOOK_UP = auto()
    LOOK_DOWN = auto()

    # Interaction
    INTERACT = auto()        # Use/toggle object in front
    PICKUP = auto()          # Collect item (auto on move)

    # Queries
    LOOK = auto()            # What's visible?
    WHERE = auto()           # Where am I?
    INVENTORY = auto()       # What do I have?
    ROOM_INFO = auto()       # What room is this?

    # Level management
    LOAD_LEVEL = auto()
    LIST_LEVELS = auto()
    RESET_LEVEL = auto()

    # Teaching
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
    MAP = auto()             # Show top-down view

    # Unknown
    UNKNOWN = auto()


@dataclass
class Home3DResponse:
    """Response from 3D home handler."""
    text: str
    action_taken: bool = False
    action_type: Optional[str] = None
    sandbox_denied: bool = False
    level_complete: bool = False
    collected_item: Optional[str] = None
    interacted_object: Optional[str] = None
    world_state: Optional[dict] = None


class Home3DIntentClassifier:
    """Classify user input into 3D home intents."""

    PATTERNS = [
        # Help/system
        (r"^(help|\?)$", Home3DIntent.HELP, {}),
        (r"^(status|info)$", Home3DIntent.STATUS, {}),
        (r"^(map|view|top)(\s+down)?$", Home3DIntent.MAP, {}),

        # Level management
        (r"^(load|start|play)\s+level\s+(\w+)$", Home3DIntent.LOAD_LEVEL, {"level": 2}),
        (r"^(load|start|play)\s+(\w+)$", Home3DIntent.LOAD_LEVEL, {"level": 2}),
        (r"^(levels?|list\s+levels?)$", Home3DIntent.LIST_LEVELS, {}),
        (r"^(reset|restart)(\s+level)?$", Home3DIntent.RESET_LEVEL, {}),

        # Teaching
        (r"^(teach|learn|record)\s+(.+)$", Home3DIntent.START_TEACHING, {"name": 2}),
        (r"^(done|finished|save|end)$", Home3DIntent.STOP_TEACHING, {}),
        (r"^(cancel|abort|nevermind)$", Home3DIntent.CANCEL_TEACHING, {}),

        # Memory
        (r"^(list|show)\s*(behaviors?|skills?|macros?)$", Home3DIntent.LIST_BEHAVIORS, {}),
        (r"^(forget|delete)\s+(.+)$", Home3DIntent.FORGET_BEHAVIOR, {"name": 2}),

        # Queries
        (r"^(look|see|scan|around|visible)$", Home3DIntent.LOOK, {}),
        (r"^(where|position|location)(\s+am\s+i)?$", Home3DIntent.WHERE, {}),
        (r"^(inventory|items?|bag|what\s+do\s+i\s+have)$", Home3DIntent.INVENTORY, {}),
        (r"^(room|room\s+info|what\s+room)$", Home3DIntent.ROOM_INFO, {}),

        # Interaction
        (r"^(interact|use|open|close|toggle|press|activate)(\s+(.+))?$", Home3DIntent.INTERACT, {"target": 3}),
        (r"^(grab|pick\s*up|take|collect)(\s+(.+))?$", Home3DIntent.PICKUP, {"item": 3}),

        # 3D Movement - Forward/Back
        (r"^(go\s+)?(forward|ahead|straight|f|w)$", Home3DIntent.MOVE_FORWARD, {}),
        (r"^(go\s+)?(back|backward|reverse|s)$", Home3DIntent.MOVE_BACKWARD, {}),

        # 3D Movement - Strafe
        (r"^strafe\s+left|strafe\s+l|q$", Home3DIntent.STRAFE_LEFT, {}),
        (r"^strafe\s+right|strafe\s+r|e$", Home3DIntent.STRAFE_RIGHT, {}),

        # 3D Movement - Turn
        (r"^(turn\s+)?left|a$", Home3DIntent.TURN_LEFT, {}),
        (r"^(turn\s+)?right|d$", Home3DIntent.TURN_RIGHT, {}),

        # 3D Movement - Look Up/Down
        (r"^look\s+up|tilt\s+up$", Home3DIntent.LOOK_UP, {}),
        (r"^look\s+down|tilt\s+down$", Home3DIntent.LOOK_DOWN, {}),
    ]

    def __init__(self, behaviors: Optional[List[str]] = None):
        self.behaviors = behaviors or []

    def update_behaviors(self, behaviors: List[str]) -> None:
        self.behaviors = behaviors

    def classify(self, text: str) -> tuple:
        """Classify text into intent and params."""
        text = text.strip().lower()

        if not text:
            return Home3DIntent.UNKNOWN, {}

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
                return Home3DIntent.INVOKE_BEHAVIOR, {"name": behavior}

        return Home3DIntent.UNKNOWN, {}


class HomeHandler:
    """
    Handles 3D home game commands and integrates with Brain B systems.

    Usage:
        world = get_level("simple_apartment")
        handler = HomeHandler(world, runtime)
        response = handler.handle("forward")
    """

    def __init__(
        self,
        world: HomeWorld,
        runtime: ActorRuntime,
        on_state_change: Optional[Callable[[dict], None]] = None,
    ):
        self.world = world
        self.runtime = runtime
        self.classifier = Home3DIntentClassifier(runtime.teaching.list_behaviors())
        self.on_state_change = on_state_change

    def handle(self, user_input: str) -> Home3DResponse:
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
            elif action_type == "strafe_left":
                self.world.strafe_left()
            elif action_type == "strafe_right":
                self.world.strafe_right()
            elif action_type == "turn_left":
                self.world.turn_left()
            elif action_type == "turn_right":
                self.world.turn_right()
            elif action_type == "look_up":
                self.world.look_up()
            elif action_type == "look_down":
                self.world.look_down()
            elif action_type == "interact":
                self.world.interact()
        return executor

    def _result_to_response(
        self,
        result: MoveResult3D,
        action_type: str,
    ) -> Home3DResponse:
        """Convert MoveResult3D to Home3DResponse."""
        prefix = "[Recording] " if self.runtime.teaching.is_recording else ""
        return Home3DResponse(
            text=f"{prefix}{result.message}",
            action_taken=result.success,
            action_type=action_type,
            sandbox_denied=result.sandbox_denied,
            level_complete=result.reached_goal,
            collected_item=result.collected_item,
            interacted_object=result.interacted_object,
        )

    # === Movement Handlers ===

    def _handle_move_forward(self, params: dict, raw: str) -> Home3DResponse:
        result = self.world.move_forward()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "forward"})

        self.runtime.execute_action(
            {"type": "forward", "game_result": result.message},
            lambda a: None
        )

        return self._result_to_response(result, "forward")

    def _handle_move_backward(self, params: dict, raw: str) -> Home3DResponse:
        result = self.world.move_backward()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "backward"})

        self.runtime.execute_action(
            {"type": "backward", "game_result": result.message},
            lambda a: None
        )

        return self._result_to_response(result, "backward")

    def _handle_strafe_left(self, params: dict, raw: str) -> Home3DResponse:
        result = self.world.strafe_left()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "strafe_left"})

        self.runtime.execute_action(
            {"type": "strafe_left", "game_result": result.message},
            lambda a: None
        )

        return self._result_to_response(result, "strafe_left")

    def _handle_strafe_right(self, params: dict, raw: str) -> Home3DResponse:
        result = self.world.strafe_right()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "strafe_right"})

        self.runtime.execute_action(
            {"type": "strafe_right", "game_result": result.message},
            lambda a: None
        )

        return self._result_to_response(result, "strafe_right")

    def _handle_turn_left(self, params: dict, raw: str) -> Home3DResponse:
        result = self.world.turn_left()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "turn_left"})

        self.runtime.execute_action(
            {"type": "turn_left", "game_result": result.message},
            lambda a: None
        )

        return self._result_to_response(result, "turn_left")

    def _handle_turn_right(self, params: dict, raw: str) -> Home3DResponse:
        result = self.world.turn_right()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "turn_right"})

        self.runtime.execute_action(
            {"type": "turn_right", "game_result": result.message},
            lambda a: None
        )

        return self._result_to_response(result, "turn_right")

    def _handle_look_up(self, params: dict, raw: str) -> Home3DResponse:
        result = self.world.look_up()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "look_up"})

        return self._result_to_response(result, "look_up")

    def _handle_look_down(self, params: dict, raw: str) -> Home3DResponse:
        result = self.world.look_down()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "look_down"})

        return self._result_to_response(result, "look_down")

    # === Interaction Handlers ===

    def _handle_interact(self, params: dict, raw: str) -> Home3DResponse:
        result = self.world.interact()

        if self.runtime.teaching.is_recording:
            self.runtime.teaching.record_action({"type": "interact"})

        self.runtime.execute_action(
            {"type": "interact", "game_result": result.message},
            lambda a: None
        )

        return self._result_to_response(result, "interact")

    def _handle_pickup(self, params: dict, raw: str) -> Home3DResponse:
        return Home3DResponse(text="Move onto items to collect them automatically.")

    # === Query Handlers ===

    def _handle_look(self, params: dict, raw: str) -> Home3DResponse:
        visible = self.world.get_visible_objects(max_distance=5.0)
        if not visible:
            return Home3DResponse(text="Nothing visible nearby.")

        lines = ["You see:"]
        for obj in visible[:10]:  # Limit to 10 objects
            dist_str = f"{obj['distance']:.1f}m"
            extra = ""
            if obj.get("interactive"):
                extra = " (interactive)"
            elif obj.get("collectible"):
                extra = " (collectible)"
            lines.append(f"  - {obj['type']}{extra} at {dist_str}")

        return Home3DResponse(text="\n".join(lines))

    def _handle_where(self, params: dict, raw: str) -> Home3DResponse:
        pos = self.world.robot.position
        rot = self.world.robot.rotation
        room = self.world.get_room_at(pos)
        room_name = room.room_type.value.replace("_", " ").title() if room else "Unknown"

        lines = [
            f"Position: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})",
            f"Facing: {rot.yaw:.0f}° (pitch: {rot.pitch:.0f}°)",
            f"Room: {room_name}",
            f"Moves: {self.world.robot.moves}",
            f"Battery: {self.world.robot.battery:.0%}",
        ]

        if self.world.goal_position:
            dist = pos.distance_to(self.world.goal_position)
            lines.append(f"Goal distance: {dist:.1f}m")

        return Home3DResponse(text="\n".join(lines))

    def _handle_inventory(self, params: dict, raw: str) -> Home3DResponse:
        inv = self.world.robot.inventory
        if inv:
            return Home3DResponse(text=f"You have: {', '.join(inv)}")
        return Home3DResponse(text="Your inventory is empty.")

    def _handle_room_info(self, params: dict, raw: str) -> Home3DResponse:
        room = self.world.get_room_at(self.world.robot.position)
        if not room:
            return Home3DResponse(text="You're not in a defined room.")

        room_name = room.room_type.value.replace("_", " ").title()
        lines = [f"Room: {room_name}"]

        # Count objects in room
        obj_counts = {}
        for obj in room.objects:
            obj_type = obj.object_type.value
            obj_counts[obj_type] = obj_counts.get(obj_type, 0) + 1

        if obj_counts:
            lines.append("Contents:")
            for obj_type, count in sorted(obj_counts.items()):
                lines.append(f"  - {obj_type}: {count}")

        if room.connected_rooms:
            lines.append(f"Connected to: {', '.join(room.connected_rooms)}")

        return Home3DResponse(text="\n".join(lines))

    # === Level Management ===

    def _handle_load_level(self, params: dict, raw: str) -> Home3DResponse:
        level_id = params.get("level", "simple_apartment")
        try:
            self.world = get_level(level_id)
            return Home3DResponse(
                text=f"Loaded: {level_id}\nGoal: {self.world.goal_description}\n\n{self.world.render_top_down()}",
                action_taken=True,
                action_type="load_level",
            )
        except ValueError as e:
            return Home3DResponse(text=str(e))

    def _handle_list_levels(self, params: dict, raw: str) -> Home3DResponse:
        level_ids = list_levels()
        lines = ["Available 3D Home levels:"]
        for level_id in level_ids:
            # Create level to get description
            level = get_level(level_id)
            lines.append(f"  {level_id}: {level.goal_description}")
        return Home3DResponse(text="\n".join(lines))

    def _handle_reset_level(self, params: dict, raw: str) -> Home3DResponse:
        level_id = self.world.level_id
        self.world = get_level(level_id)
        return Home3DResponse(
            text=f"Level reset.\n\n{self.world.render_top_down()}",
            action_taken=True,
            action_type="reset",
        )

    # === Teaching Handlers ===

    def _handle_start_teaching(self, params: dict, raw: str) -> Home3DResponse:
        name = params.get("name", "").strip()
        if not name:
            return Home3DResponse(text="What should I call this behavior? Say 'teach <name>'.")
        result = self.runtime.teach(name)
        return Home3DResponse(text=result)

    def _handle_stop_teaching(self, params: dict, raw: str) -> Home3DResponse:
        if not self.runtime.teaching.is_recording:
            return Home3DResponse(text="Not currently recording.")
        result = self.runtime.done_teaching()
        return Home3DResponse(text=result, action_taken=True)

    def _handle_cancel_teaching(self, params: dict, raw: str) -> Home3DResponse:
        if not self.runtime.teaching.is_recording:
            return Home3DResponse(text="Not currently recording.")
        result = self.runtime.teaching.cancel_recording()
        return Home3DResponse(text=result)

    def _handle_invoke_behavior(self, params: dict, raw: str) -> Home3DResponse:
        name = params.get("name", "").strip()
        if not name:
            return Home3DResponse(text="What behavior should I run?")

        executor = self._create_executor()

        def on_step(step: int, total: int, action: dict):
            pass

        result = self.runtime.invoke(name, executor, on_step)
        return Home3DResponse(text=result, action_taken=True, action_type="invoke")

    # === Memory Handlers ===

    def _handle_list_behaviors(self, params: dict, raw: str) -> Home3DResponse:
        behaviors = self.runtime.teaching.list_behaviors()
        if not behaviors:
            return Home3DResponse(text="No behaviors learned yet. Use 'teach <name>' to start.")
        return Home3DResponse(text=f"Known behaviors: {', '.join(behaviors)}")

    def _handle_forget_behavior(self, params: dict, raw: str) -> Home3DResponse:
        name = params.get("name", "").strip()
        if not name:
            return Home3DResponse(text="What should I forget?")
        result = self.runtime.teaching.forget(name)
        return Home3DResponse(text=result)

    # === System Handlers ===

    def _handle_status(self, params: dict, raw: str) -> Home3DResponse:
        status = self.runtime.status()
        pos = self.world.robot.position
        room = self.world.get_room_at(pos)

        lines = [
            f"Level: {self.world.level_id}",
            f"Goal: {self.world.goal_description}",
            f"Room: {room.room_type.value if room else 'unknown'}",
            f"Position: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})",
            f"Moves: {self.world.robot.moves}",
            f"Battery: {self.world.robot.battery:.0%}",
            f"Inventory: {len(self.world.robot.inventory)} items",
            f"Behaviors: {status['behaviors']}",
            f"Recording: {status['recording_name'] or 'No'}",
            f"Complete: {self.world.level_complete}",
        ]
        return Home3DResponse(text="\n".join(lines))

    def _handle_map(self, params: dict, raw: str) -> Home3DResponse:
        return Home3DResponse(text=self.world.render_top_down())

    def _handle_help(self, params: dict, raw: str) -> Home3DResponse:
        help_text = """3D Home Exploration Commands:

Movement:
  forward (f/w)    - Move forward
  backward (b/s)   - Move backward
  strafe left (q)  - Strafe left
  strafe right (e) - Strafe right
  turn left (a)    - Turn left 45 degrees
  turn right (d)   - Turn right 45 degrees
  look up          - Tilt camera up
  look down        - Tilt camera down

Interaction:
  interact         - Use object in front (doors, switches)
  grab             - Items auto-collect on contact

Queries:
  look             - List visible objects
  where            - Current position and room
  room             - Room details
  inventory        - Show collected items
  map              - Top-down view
  status           - Full status

Levels:
  levels           - List available levels
  load <name>      - Load a level
  reset            - Restart current level

Teaching:
  teach <name>     - Start recording behavior
  done             - Save recording
  cancel           - Cancel recording
  <name>           - Run learned behavior
  list behaviors   - Show learned behaviors
  forget <name>    - Delete behavior

Symbols: ^ v < > = robot direction, # = wall, D = door, G = goal"""
        return Home3DResponse(text=help_text)

    def _handle_unknown(self, params: dict, raw: str) -> Home3DResponse:
        if self.runtime.teaching.is_recording:
            return Home3DResponse(
                text=f"Unknown command: '{raw}'. Say 'done' to save or 'cancel' to abort."
            )
        return Home3DResponse(text=f"Unknown command: '{raw}'. Say 'help' for commands.")
