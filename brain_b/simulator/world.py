"""
Grid World - The game environment for RobotGrid.

Provides:
- Tile-based grid with various cell types
- Robot with position, direction, inventory
- Game state serialization for event sourcing
- Puzzle mechanics (keys, doors, pushable boxes)
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
import json
import copy


class Direction(Enum):
    """Robot facing direction."""
    NORTH = auto()  # Up (y decreases)
    EAST = auto()   # Right (x increases)
    SOUTH = auto()  # Down (y increases)
    WEST = auto()   # Left (x decreases)

    def turn_left(self) -> "Direction":
        return {
            Direction.NORTH: Direction.WEST,
            Direction.WEST: Direction.SOUTH,
            Direction.SOUTH: Direction.EAST,
            Direction.EAST: Direction.NORTH,
        }[self]

    def turn_right(self) -> "Direction":
        return {
            Direction.NORTH: Direction.EAST,
            Direction.EAST: Direction.SOUTH,
            Direction.SOUTH: Direction.WEST,
            Direction.WEST: Direction.NORTH,
        }[self]

    def delta(self) -> tuple[int, int]:
        """Get (dx, dy) for moving in this direction."""
        return {
            Direction.NORTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0),
        }[self]

    @property
    def symbol(self) -> str:
        return {
            Direction.NORTH: "^",
            Direction.EAST: ">",
            Direction.SOUTH: "v",
            Direction.WEST: "<",
        }[self]


class Tile(Enum):
    """Types of tiles in the grid."""
    FLOOR = "."       # Empty floor - robot can move here
    WALL = "#"        # Solid wall - blocks movement
    LAVA = "~"        # Restricted zone - sandbox denies entry
    KEY = "K"         # Collectible key
    DOOR = "D"        # Door - requires key to open
    GOAL = "G"        # Level goal
    BOX = "B"         # Pushable box
    BUTTON = "O"      # Pressure button (activates when box/robot on it)
    START = "S"       # Starting position (becomes floor)

    @property
    def walkable(self) -> bool:
        """Can the robot walk on this tile?"""
        return self in (Tile.FLOOR, Tile.KEY, Tile.GOAL, Tile.BUTTON, Tile.START)

    @property
    def restricted(self) -> bool:
        """Is this tile restricted by sandbox?"""
        return self == Tile.LAVA

    @property
    def collectible(self) -> bool:
        """Can this tile be picked up?"""
        return self == Tile.KEY

    @property
    def pushable(self) -> bool:
        """Can this tile be pushed?"""
        return self == Tile.BOX


@dataclass
class Robot:
    """The robot navigating the grid."""
    x: int
    y: int
    direction: Direction = Direction.NORTH
    inventory: list[str] = field(default_factory=list)
    moves: int = 0

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "direction": self.direction.name,
            "inventory": self.inventory.copy(),
            "moves": self.moves,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Robot":
        return cls(
            x=data["x"],
            y=data["y"],
            direction=Direction[data["direction"]],
            inventory=data.get("inventory", []),
            moves=data.get("moves", 0),
        )


@dataclass
class MoveResult:
    """Result of attempting a move."""
    success: bool
    message: str
    action_type: str
    tile_entered: Optional[Tile] = None
    item_collected: Optional[str] = None
    sandbox_denied: bool = False
    level_complete: bool = False


class GridWorld:
    """
    The game world - a 2D grid with robot navigation.

    Supports:
    - Loading levels from string or dict
    - Robot movement with collision detection
    - Item collection and door mechanics
    - State serialization for checkpointing
    """

    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.grid: list[list[Tile]] = [[Tile.FLOOR] * width for _ in range(height)]
        self.robot = Robot(0, 0)
        self.level_name = "empty"
        self.buttons_pressed: set[tuple[int, int]] = set()
        self._initial_state: Optional[dict] = None

    def load_level(self, level_str: str, name: str = "custom") -> None:
        """
        Load a level from a string representation.

        Example:
            ###########
            #S..K..D.G#
            #.##~~~##.#
            #.........#
            ###########
        """
        lines = [line for line in level_str.strip().split("\n") if line]
        self.height = len(lines)
        self.width = max(len(line) for line in lines)
        self.grid = []
        self.level_name = name
        self.robot = Robot(0, 0)
        self.buttons_pressed = set()

        char_to_tile = {t.value: t for t in Tile}

        for y, line in enumerate(lines):
            row = []
            for x, char in enumerate(line):
                tile = char_to_tile.get(char, Tile.FLOOR)
                if tile == Tile.START:
                    self.robot = Robot(x, y, Direction.EAST)
                    tile = Tile.FLOOR
                row.append(tile)
            # Pad row to full width
            row.extend([Tile.WALL] * (self.width - len(row)))
            self.grid.append(row)

        self._initial_state = self.to_dict()

    def reset(self) -> None:
        """Reset level to initial state."""
        if self._initial_state:
            self.from_dict(self._initial_state)

    def get_tile(self, x: int, y: int) -> Tile:
        """Get tile at position, WALL if out of bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return Tile.WALL

    def set_tile(self, x: int, y: int, tile: Tile) -> None:
        """Set tile at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = tile

    def move_forward(self) -> MoveResult:
        """Move robot one tile in facing direction."""
        dx, dy = self.robot.direction.delta()
        new_x = self.robot.x + dx
        new_y = self.robot.y + dy

        target_tile = self.get_tile(new_x, new_y)

        # Check wall collision
        if target_tile == Tile.WALL:
            return MoveResult(False, "Blocked by wall.", "forward")

        # Check lava (sandbox restriction)
        if target_tile.restricted:
            return MoveResult(
                False,
                "SANDBOX DENIED: Restricted zone (lava).",
                "forward",
                sandbox_denied=True
            )

        # Check door without key
        if target_tile == Tile.DOOR:
            if "key" not in self.robot.inventory:
                return MoveResult(False, "Door is locked. Need a key.", "forward")
            # Use key to open door
            self.robot.inventory.remove("key")
            self.set_tile(new_x, new_y, Tile.FLOOR)

        # Check pushable box
        if target_tile == Tile.BOX:
            push_result = self._push_box(new_x, new_y, dx, dy)
            if not push_result.success:
                return push_result

        # Move robot
        self.robot.x = new_x
        self.robot.y = new_y
        self.robot.moves += 1

        # Check for item collection
        target_tile = self.get_tile(new_x, new_y)  # Re-get in case box was pushed
        item_collected = None
        if target_tile == Tile.KEY:
            self.robot.inventory.append("key")
            self.set_tile(new_x, new_y, Tile.FLOOR)
            item_collected = "key"

        # Check for goal
        if target_tile == Tile.GOAL:
            return MoveResult(
                True,
                "LEVEL COMPLETE! You reached the goal!",
                "forward",
                tile_entered=target_tile,
                item_collected=item_collected,
                level_complete=True
            )

        # Check button press
        if target_tile == Tile.BUTTON:
            self.buttons_pressed.add((new_x, new_y))

        msg = "Moved forward."
        if item_collected:
            msg = f"Moved forward. Collected {item_collected}!"

        return MoveResult(True, msg, "forward", target_tile, item_collected)

    def _push_box(self, box_x: int, box_y: int, dx: int, dy: int) -> MoveResult:
        """Try to push a box."""
        new_box_x = box_x + dx
        new_box_y = box_y + dy

        target = self.get_tile(new_box_x, new_box_y)

        if not target.walkable and target != Tile.BUTTON:
            return MoveResult(False, "Can't push box - blocked.", "forward")

        if target.restricted:
            return MoveResult(
                False,
                "SANDBOX DENIED: Can't push box into restricted zone.",
                "forward",
                sandbox_denied=True
            )

        # Push the box
        self.set_tile(box_x, box_y, Tile.FLOOR)
        self.set_tile(new_box_x, new_box_y, Tile.BOX)

        return MoveResult(True, "Pushed box.", "forward")

    def move_backward(self) -> MoveResult:
        """Move robot one tile backward (without turning)."""
        # Temporarily reverse direction
        opposite = self.robot.direction.turn_left().turn_left()
        dx, dy = opposite.delta()
        new_x = self.robot.x + dx
        new_y = self.robot.y + dy

        target_tile = self.get_tile(new_x, new_y)

        if target_tile == Tile.WALL:
            return MoveResult(False, "Blocked by wall.", "backward")

        if target_tile.restricted:
            return MoveResult(
                False,
                "SANDBOX DENIED: Restricted zone (lava).",
                "backward",
                sandbox_denied=True
            )

        if target_tile == Tile.DOOR:
            if "key" not in self.robot.inventory:
                return MoveResult(False, "Door is locked. Need a key.", "backward")
            self.robot.inventory.remove("key")
            self.set_tile(new_x, new_y, Tile.FLOOR)

        if target_tile == Tile.BOX:
            return MoveResult(False, "Can't push box backward.", "backward")

        self.robot.x = new_x
        self.robot.y = new_y
        self.robot.moves += 1

        target_tile = self.get_tile(new_x, new_y)
        item_collected = None
        if target_tile == Tile.KEY:
            self.robot.inventory.append("key")
            self.set_tile(new_x, new_y, Tile.FLOOR)
            item_collected = "key"

        if target_tile == Tile.GOAL:
            return MoveResult(
                True,
                "LEVEL COMPLETE! You reached the goal!",
                "backward",
                tile_entered=target_tile,
                level_complete=True
            )

        msg = "Moved backward."
        if item_collected:
            msg = f"Moved backward. Collected {item_collected}!"

        return MoveResult(True, msg, "backward", target_tile, item_collected)

    def turn_left(self) -> MoveResult:
        """Turn robot 90 degrees left."""
        self.robot.direction = self.robot.direction.turn_left()
        return MoveResult(True, f"Turned left. Now facing {self.robot.direction.name}.", "left")

    def turn_right(self) -> MoveResult:
        """Turn robot 90 degrees right."""
        self.robot.direction = self.robot.direction.turn_right()
        return MoveResult(True, f"Turned right. Now facing {self.robot.direction.name}.", "right")

    def look(self) -> str:
        """Describe what the robot sees ahead."""
        dx, dy = self.robot.direction.delta()
        look_x = self.robot.x + dx
        look_y = self.robot.y + dy

        tile = self.get_tile(look_x, look_y)
        descriptions = {
            Tile.FLOOR: "empty floor",
            Tile.WALL: "a wall",
            Tile.LAVA: "dangerous lava (restricted zone)",
            Tile.KEY: "a shiny key",
            Tile.DOOR: "a locked door",
            Tile.GOAL: "the goal!",
            Tile.BOX: "a pushable box",
            Tile.BUTTON: "a pressure button",
        }
        return f"I see {descriptions.get(tile, 'something unknown')} ahead."

    def where_am_i(self) -> str:
        """Describe robot's current position and state."""
        lines = [
            f"Position: ({self.robot.x}, {self.robot.y})",
            f"Facing: {self.robot.direction.name}",
            f"Moves: {self.robot.moves}",
        ]
        if self.robot.inventory:
            lines.append(f"Inventory: {', '.join(self.robot.inventory)}")
        else:
            lines.append("Inventory: empty")

        current_tile = self.get_tile(self.robot.x, self.robot.y)
        lines.append(f"Standing on: {current_tile.name}")

        return "\n".join(lines)

    def render(self) -> str:
        """Render the grid as ASCII art."""
        lines = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if x == self.robot.x and y == self.robot.y:
                    row += self.robot.direction.symbol
                else:
                    row += self.grid[y][x].value
            lines.append(row)
        return "\n".join(lines)

    def render_with_border(self) -> str:
        """Render with a nice border for display."""
        grid = self.render()
        lines = grid.split("\n")
        bordered = ["┌" + "─" * self.width + "┐"]
        for line in lines:
            bordered.append("│" + line + "│")
        bordered.append("└" + "─" * self.width + "┘")
        return "\n".join(bordered)

    def to_dict(self) -> dict:
        """Serialize world state for checkpointing."""
        return {
            "width": self.width,
            "height": self.height,
            "grid": [[t.value for t in row] for row in self.grid],
            "robot": self.robot.to_dict(),
            "level_name": self.level_name,
            "buttons_pressed": list(self.buttons_pressed),
        }

    def from_dict(self, data: dict) -> None:
        """Restore world state from checkpoint."""
        self.width = data["width"]
        self.height = data["height"]
        char_to_tile = {t.value: t for t in Tile}
        self.grid = [[char_to_tile.get(c, Tile.FLOOR) for c in row] for row in data["grid"]]
        self.robot = Robot.from_dict(data["robot"])
        self.level_name = data.get("level_name", "unknown")
        self.buttons_pressed = set(tuple(b) for b in data.get("buttons_pressed", []))
        self._initial_state = copy.deepcopy(data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "GridWorld":
        """Create world from JSON string."""
        data = json.loads(json_str)
        world = cls()
        world.from_dict(data)
        return world


# === Built-in Levels ===

LEVELS = {
    "tutorial": {
        "name": "Tutorial",
        "description": "Learn the basics. Get to the goal!",
        "map": """
#########
#S......G#
#########
""",
    },
    "key_door": {
        "name": "Key & Door",
        "description": "Find the key to unlock the door.",
        "map": """
###########
#S...K....#
#.#######.#
#.........#
#.######D##
#........G#
###########
""",
    },
    "lava_maze": {
        "name": "Lava Maze",
        "description": "Navigate around restricted zones.",
        "map": """
#############
#S..........#
#.~~~.~~~.#.#
#.~~~.~~~.#.#
#.....~~~.#.#
#####.~~~.#.#
#G....~~~...#
#############
""",
    },
    "box_puzzle": {
        "name": "Box Puzzle",
        "description": "Push the box onto the button to open the path.",
        "map": """
###########
#S........#
#.######..#
#.#....#..#
#.#.B..O..#
#.#....####
#.#......G#
###########
""",
    },
    "challenge": {
        "name": "Challenge",
        "description": "Combine all skills: keys, doors, boxes, and lava.",
        "map": """
###############
#S....~~~.....#
#.###.~~~.###.#
#.#K#.~~~.#D#.#
#.###.~~~.###.#
#.....~~~.....#
#.###.~~~.###.#
#.#B#.....#O#.#
#.###.###.###.#
#.....#G#.....#
###############
""",
    },
}


def load_level(level_id: str) -> GridWorld:
    """Load a built-in level by ID."""
    if level_id not in LEVELS:
        raise ValueError(f"Unknown level: {level_id}. Available: {list(LEVELS.keys())}")

    level = LEVELS[level_id]
    world = GridWorld()
    world.load_level(level["map"], level["name"])
    return world
