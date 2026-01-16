"""
3D Home Exploration Game - World Model

A voxel-based 3D environment for robot home exploration training.
Robot navigates rooms, interacts with objects, and learns home tasks.
"""

import json
import random
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set


class RoomType(Enum):
    """Types of rooms in a home."""
    LIVING_ROOM = "living_room"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    HALLWAY = "hallway"
    GARAGE = "garage"
    OFFICE = "office"
    DINING_ROOM = "dining_room"


class ObjectType(Enum):
    """Types of objects in the home."""
    # Furniture
    COUCH = "couch"
    TABLE = "table"
    CHAIR = "chair"
    BED = "bed"
    DESK = "desk"
    SHELF = "shelf"

    # Appliances
    FRIDGE = "fridge"
    STOVE = "stove"
    SINK = "sink"
    TV = "tv"
    LAMP = "lamp"

    # Interactive
    DOOR = "door"
    DRAWER = "drawer"
    SWITCH = "switch"

    # Collectibles
    KEY = "key"
    REMOTE = "remote"
    PHONE = "phone"
    BOOK = "book"
    CUP = "cup"

    # Structure
    WALL = "wall"
    FLOOR = "floor"
    STAIRS = "stairs"
    WINDOW = "window"


class Direction3D(Enum):
    """3D directions for movement."""
    NORTH = (0, -1, 0)
    SOUTH = (0, 1, 0)
    EAST = (1, 0, 0)
    WEST = (-1, 0, 0)
    UP = (0, 0, 1)
    DOWN = (0, 0, -1)

    def delta(self) -> Tuple[int, int, int]:
        return self.value

    @staticmethod
    def from_angle(yaw: float) -> "Direction3D":
        """Get cardinal direction from yaw angle (degrees)."""
        yaw = yaw % 360
        if 315 <= yaw or yaw < 45:
            return Direction3D.NORTH
        elif 45 <= yaw < 135:
            return Direction3D.EAST
        elif 135 <= yaw < 225:
            return Direction3D.SOUTH
        else:
            return Direction3D.WEST


@dataclass
class Position3D:
    """3D position in the world."""
    x: float
    y: float
    z: float  # Height/floor level

    def __add__(self, other: "Position3D") -> "Position3D":
        return Position3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def distance_to(self, other: "Position3D") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def to_grid(self) -> Tuple[int, int, int]:
        """Convert to integer grid coordinates."""
        return (int(self.x), int(self.y), int(self.z))

    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(d: Dict) -> "Position3D":
        return Position3D(d["x"], d["y"], d["z"])


@dataclass
class Rotation3D:
    """3D rotation (Euler angles in degrees)."""
    pitch: float = 0.0  # Up/down look (-90 to 90)
    yaw: float = 0.0    # Left/right rotation (0-360)
    roll: float = 0.0   # Tilt (usually 0)

    def forward_vector(self) -> Tuple[float, float, float]:
        """Get unit vector pointing forward."""
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)

        x = math.cos(pitch_rad) * math.sin(yaw_rad)
        y = -math.cos(pitch_rad) * math.cos(yaw_rad)
        z = math.sin(pitch_rad)

        return (x, y, z)

    def to_dict(self) -> Dict:
        return {"pitch": self.pitch, "yaw": self.yaw, "roll": self.roll}

    @staticmethod
    def from_dict(d: Dict) -> "Rotation3D":
        return Rotation3D(d.get("pitch", 0), d.get("yaw", 0), d.get("roll", 0))


@dataclass
class WorldObject:
    """An object in the 3D world."""
    object_type: ObjectType
    position: Position3D
    size: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # width, depth, height
    is_solid: bool = True
    is_interactive: bool = False
    is_collectible: bool = False
    state: Dict = field(default_factory=dict)  # e.g., {"open": False} for doors

    def to_dict(self) -> Dict:
        return {
            "type": self.object_type.value,
            "position": self.position.to_dict(),
            "size": list(self.size),
            "is_solid": self.is_solid,
            "is_interactive": self.is_interactive,
            "is_collectible": self.is_collectible,
            "state": self.state,
        }

    @staticmethod
    def from_dict(d: Dict) -> "WorldObject":
        return WorldObject(
            object_type=ObjectType(d["type"]),
            position=Position3D.from_dict(d["position"]),
            size=tuple(d.get("size", [1, 1, 1])),
            is_solid=d.get("is_solid", True),
            is_interactive=d.get("is_interactive", False),
            is_collectible=d.get("is_collectible", False),
            state=d.get("state", {}),
        )

    def contains_point(self, pos: Position3D) -> bool:
        """Check if a point is inside this object's bounding box."""
        return (
            self.position.x <= pos.x < self.position.x + self.size[0] and
            self.position.y <= pos.y < self.position.y + self.size[1] and
            self.position.z <= pos.z < self.position.z + self.size[2]
        )


@dataclass
class Room:
    """A room in the home."""
    room_type: RoomType
    bounds: Tuple[Position3D, Position3D]  # min corner, max corner
    objects: List[WorldObject] = field(default_factory=list)
    connected_rooms: List[str] = field(default_factory=list)  # Room IDs

    def contains_point(self, pos: Position3D) -> bool:
        """Check if a point is inside this room."""
        return (
            self.bounds[0].x <= pos.x < self.bounds[1].x and
            self.bounds[0].y <= pos.y < self.bounds[1].y and
            self.bounds[0].z <= pos.z < self.bounds[1].z
        )

    def to_dict(self) -> Dict:
        return {
            "type": self.room_type.value,
            "bounds": [self.bounds[0].to_dict(), self.bounds[1].to_dict()],
            "objects": [obj.to_dict() for obj in self.objects],
            "connected_rooms": self.connected_rooms,
        }


@dataclass
class Robot3D:
    """The robot exploring the home."""
    position: Position3D = field(default_factory=lambda: Position3D(0, 0, 0))
    rotation: Rotation3D = field(default_factory=Rotation3D)
    height: float = 1.0  # Robot height
    radius: float = 0.3  # Robot collision radius
    inventory: List[str] = field(default_factory=list)
    battery: float = 1.0  # 0.0 to 1.0
    moves: int = 0

    def to_dict(self) -> Dict:
        return {
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "height": self.height,
            "radius": self.radius,
            "inventory": self.inventory.copy(),
            "battery": self.battery,
            "moves": self.moves,
        }

    @staticmethod
    def from_dict(d: Dict) -> "Robot3D":
        robot = Robot3D(
            position=Position3D.from_dict(d["position"]),
            rotation=Rotation3D.from_dict(d["rotation"]),
            height=d.get("height", 1.0),
            radius=d.get("radius", 0.3),
            inventory=d.get("inventory", []).copy(),
            battery=d.get("battery", 1.0),
            moves=d.get("moves", 0),
        )
        return robot


@dataclass
class MoveResult3D:
    """Result of a movement action."""
    success: bool
    message: str
    collected_item: Optional[str] = None
    interacted_object: Optional[str] = None
    reached_goal: bool = False
    sandbox_denied: bool = False


class HomeWorld:
    """
    3D Home environment for robot exploration.

    Features:
    - Voxel-based collision detection
    - Multiple rooms with different types
    - Interactive objects (doors, drawers, switches)
    - Collectible items
    - First-person view model
    """

    def __init__(
        self,
        width: int = 20,
        depth: int = 20,
        height: int = 3,  # Usually 1 floor = 3 units high
    ):
        self.width = width
        self.depth = depth
        self.height = height

        self.rooms: Dict[str, Room] = {}
        self.objects: List[WorldObject] = []
        self.robot = Robot3D()

        self.goal_position: Optional[Position3D] = None
        self.goal_description: str = "Explore the home"

        self.level_id: str = "default"
        self.level_complete: bool = False

    def add_room(self, room_id: str, room: Room):
        """Add a room to the world."""
        self.rooms[room_id] = room
        # Add room objects to global list
        for obj in room.objects:
            if obj not in self.objects:
                self.objects.append(obj)

    def add_object(self, obj: WorldObject):
        """Add an object to the world."""
        self.objects.append(obj)

    def get_room_at(self, pos: Position3D) -> Optional[Room]:
        """Get the room containing a position."""
        for room in self.rooms.values():
            if room.contains_point(pos):
                return room
        return None

    def get_object_at(self, pos: Position3D) -> Optional[WorldObject]:
        """Get the object at a position."""
        for obj in self.objects:
            if obj.contains_point(pos):
                return obj
        return None

    def is_walkable(self, pos: Position3D) -> bool:
        """Check if a position is walkable (no solid objects)."""
        # Check bounds
        if not (0 <= pos.x < self.width and 0 <= pos.y < self.depth and 0 <= pos.z < self.height):
            return False

        # Check solid objects
        for obj in self.objects:
            if obj.is_solid and obj.contains_point(pos):
                # Special case: doors can be opened
                if obj.object_type == ObjectType.DOOR and obj.state.get("open", False):
                    continue
                return False

        return True

    def move_forward(self, distance: float = 1.0) -> MoveResult3D:
        """Move the robot forward in its facing direction."""
        dx, dy, dz = self.robot.rotation.forward_vector()

        # Only move horizontally (ignore pitch for walking)
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx, dy = dx / length * distance, dy / length * distance

        new_pos = Position3D(
            self.robot.position.x + dx,
            self.robot.position.y + dy,
            self.robot.position.z,
        )

        return self._try_move(new_pos)

    def move_backward(self, distance: float = 1.0) -> MoveResult3D:
        """Move the robot backward."""
        dx, dy, dz = self.robot.rotation.forward_vector()
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx, dy = -dx / length * distance, -dy / length * distance

        new_pos = Position3D(
            self.robot.position.x + dx,
            self.robot.position.y + dy,
            self.robot.position.z,
        )

        return self._try_move(new_pos)

    def strafe_left(self, distance: float = 1.0) -> MoveResult3D:
        """Strafe left (perpendicular to facing)."""
        yaw_rad = math.radians(self.robot.rotation.yaw - 90)
        dx = math.sin(yaw_rad) * distance
        dy = -math.cos(yaw_rad) * distance

        new_pos = Position3D(
            self.robot.position.x + dx,
            self.robot.position.y + dy,
            self.robot.position.z,
        )

        return self._try_move(new_pos)

    def strafe_right(self, distance: float = 1.0) -> MoveResult3D:
        """Strafe right (perpendicular to facing)."""
        yaw_rad = math.radians(self.robot.rotation.yaw + 90)
        dx = math.sin(yaw_rad) * distance
        dy = -math.cos(yaw_rad) * distance

        new_pos = Position3D(
            self.robot.position.x + dx,
            self.robot.position.y + dy,
            self.robot.position.z,
        )

        return self._try_move(new_pos)

    def _try_move(self, new_pos: Position3D) -> MoveResult3D:
        """Attempt to move to a new position."""
        if not self.is_walkable(new_pos):
            return MoveResult3D(False, "Path blocked")

        self.robot.position = new_pos
        self.robot.moves += 1
        self.robot.battery = max(0.0, self.robot.battery - 0.001)

        # Check for collectibles
        collected = None
        for obj in self.objects:
            if obj.is_collectible and obj.contains_point(new_pos):
                self.robot.inventory.append(obj.object_type.value)
                self.objects.remove(obj)
                collected = obj.object_type.value
                break

        # Check for goal
        reached_goal = False
        if self.goal_position and new_pos.distance_to(self.goal_position) < 1.0:
            reached_goal = True
            self.level_complete = True

        return MoveResult3D(
            True,
            "Moved successfully",
            collected_item=collected,
            reached_goal=reached_goal,
        )

    def turn_left(self, degrees: float = 45.0) -> MoveResult3D:
        """Turn the robot left."""
        self.robot.rotation.yaw = (self.robot.rotation.yaw - degrees) % 360
        return MoveResult3D(True, f"Turned left {degrees} degrees")

    def turn_right(self, degrees: float = 45.0) -> MoveResult3D:
        """Turn the robot right."""
        self.robot.rotation.yaw = (self.robot.rotation.yaw + degrees) % 360
        return MoveResult3D(True, f"Turned right {degrees} degrees")

    def look_up(self, degrees: float = 15.0) -> MoveResult3D:
        """Tilt camera up."""
        self.robot.rotation.pitch = min(90, self.robot.rotation.pitch + degrees)
        return MoveResult3D(True, f"Looking up")

    def look_down(self, degrees: float = 15.0) -> MoveResult3D:
        """Tilt camera down."""
        self.robot.rotation.pitch = max(-90, self.robot.rotation.pitch - degrees)
        return MoveResult3D(True, f"Looking down")

    def interact(self) -> MoveResult3D:
        """Interact with object in front of robot."""
        # Check objects in front of robot
        dx, dy, _ = self.robot.rotation.forward_vector()
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx, dy = dx / length, dy / length

        check_pos = Position3D(
            self.robot.position.x + dx,
            self.robot.position.y + dy,
            self.robot.position.z,
        )

        for obj in self.objects:
            if obj.is_interactive and obj.contains_point(check_pos):
                return self._interact_with(obj)

        return MoveResult3D(False, "Nothing to interact with")

    def _interact_with(self, obj: WorldObject) -> MoveResult3D:
        """Interact with a specific object."""
        if obj.object_type == ObjectType.DOOR:
            obj.state["open"] = not obj.state.get("open", False)
            state = "opened" if obj.state["open"] else "closed"
            return MoveResult3D(True, f"Door {state}", interacted_object="door")

        elif obj.object_type == ObjectType.SWITCH:
            obj.state["on"] = not obj.state.get("on", False)
            state = "on" if obj.state["on"] else "off"
            return MoveResult3D(True, f"Switch turned {state}", interacted_object="switch")

        elif obj.object_type == ObjectType.DRAWER:
            obj.state["open"] = not obj.state.get("open", False)
            state = "opened" if obj.state["open"] else "closed"
            return MoveResult3D(True, f"Drawer {state}", interacted_object="drawer")

        return MoveResult3D(True, f"Interacted with {obj.object_type.value}")

    def get_visible_objects(self, max_distance: float = 10.0) -> List[Dict]:
        """Get objects visible from robot's position."""
        visible = []

        for obj in self.objects:
            dist = self.robot.position.distance_to(obj.position)
            if dist <= max_distance:
                # Simple visibility check (could add ray casting)
                visible.append({
                    "type": obj.object_type.value,
                    "distance": dist,
                    "position": obj.position.to_dict(),
                    "interactive": obj.is_interactive,
                    "collectible": obj.is_collectible,
                })

        return sorted(visible, key=lambda x: x["distance"])

    def get_observation(self) -> Dict:
        """Get full observation for RLDS logging."""
        current_room = self.get_room_at(self.robot.position)

        return {
            "robot_state": self.robot.to_dict(),
            "current_room": current_room.room_type.value if current_room else "unknown",
            "visible_objects": self.get_visible_objects(),
            "goal_distance": self.robot.position.distance_to(self.goal_position) if self.goal_position else -1,
            "level_complete": self.level_complete,
        }

    def to_dict(self) -> Dict:
        """Serialize world state."""
        return {
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "level_id": self.level_id,
            "robot": self.robot.to_dict(),
            "rooms": {rid: room.to_dict() for rid, room in self.rooms.items()},
            "objects": [obj.to_dict() for obj in self.objects],
            "goal_position": self.goal_position.to_dict() if self.goal_position else None,
            "goal_description": self.goal_description,
            "level_complete": self.level_complete,
        }

    @staticmethod
    def from_dict(d: Dict) -> "HomeWorld":
        """Deserialize world state."""
        world = HomeWorld(d["width"], d["depth"], d["height"])
        world.level_id = d.get("level_id", "default")
        world.robot = Robot3D.from_dict(d["robot"])
        world.goal_description = d.get("goal_description", "")
        world.level_complete = d.get("level_complete", False)

        if d.get("goal_position"):
            world.goal_position = Position3D.from_dict(d["goal_position"])

        for obj_data in d.get("objects", []):
            world.objects.append(WorldObject.from_dict(obj_data))

        return world

    def render_top_down(self) -> str:
        """Render a top-down ASCII view of the current floor."""
        z = int(self.robot.position.z)
        grid = [["." for _ in range(self.width)] for _ in range(self.depth)]

        # Draw objects
        for obj in self.objects:
            if int(obj.position.z) == z:
                x, y = int(obj.position.x), int(obj.position.y)
                if 0 <= x < self.width and 0 <= y < self.depth:
                    char = self._object_char(obj.object_type)
                    grid[y][x] = char

        # Draw robot
        rx, ry = int(self.robot.position.x), int(self.robot.position.y)
        if 0 <= rx < self.width and 0 <= ry < self.depth:
            grid[ry][rx] = self._robot_char()

        # Draw goal
        if self.goal_position and int(self.goal_position.z) == z:
            gx, gy = int(self.goal_position.x), int(self.goal_position.y)
            if 0 <= gx < self.width and 0 <= gy < self.depth:
                grid[gy][gx] = "G"

        return "\n".join("".join(row) for row in grid)

    def _object_char(self, obj_type: ObjectType) -> str:
        """Get ASCII character for object type."""
        chars = {
            ObjectType.WALL: "#",
            ObjectType.DOOR: "D",
            ObjectType.COUCH: "C",
            ObjectType.TABLE: "T",
            ObjectType.BED: "B",
            ObjectType.FRIDGE: "F",
            ObjectType.STOVE: "S",
            ObjectType.KEY: "K",
            ObjectType.STAIRS: "^",
        }
        return chars.get(obj_type, "?")

    def _robot_char(self) -> str:
        """Get ASCII character for robot direction."""
        yaw = self.robot.rotation.yaw % 360
        if 315 <= yaw or yaw < 45:
            return "^"  # North
        elif 45 <= yaw < 135:
            return ">"  # East
        elif 135 <= yaw < 225:
            return "v"  # South
        else:
            return "<"  # West


# ============================================================================
# Level Definitions
# ============================================================================

def create_simple_apartment() -> HomeWorld:
    """Create a simple one-room apartment."""
    world = HomeWorld(width=10, depth=10, height=3)
    world.level_id = "simple_apartment"
    world.goal_description = "Find the key and reach the door"

    # Walls around the room
    for x in range(10):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 9, 0)))
    for y in range(1, 9):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(9, y, 0)))

    # Furniture
    world.add_object(WorldObject(ObjectType.COUCH, Position3D(2, 2, 0), size=(2, 1, 1)))
    world.add_object(WorldObject(ObjectType.TABLE, Position3D(5, 5, 0)))
    world.add_object(WorldObject(ObjectType.BED, Position3D(7, 2, 0), size=(2, 2, 1)))

    # Key (collectible)
    world.add_object(WorldObject(
        ObjectType.KEY,
        Position3D(8, 7, 0),
        is_solid=False,
        is_collectible=True,
    ))

    # Door (goal)
    world.add_object(WorldObject(
        ObjectType.DOOR,
        Position3D(5, 0, 0),
        is_interactive=True,
        state={"open": False},
    ))

    # Robot start
    world.robot.position = Position3D(1, 5, 0)
    world.robot.rotation.yaw = 0  # Facing north

    # Goal
    world.goal_position = Position3D(5, 1, 0)

    return world


def create_two_room_house() -> HomeWorld:
    """Create a two-room house with a hallway."""
    world = HomeWorld(width=15, depth=10, height=3)
    world.level_id = "two_room_house"
    world.goal_description = "Navigate from bedroom to kitchen"

    # Outer walls
    for x in range(15):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 9, 0)))
    for y in range(1, 9):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(14, y, 0)))

    # Dividing wall with door
    for y in range(1, 9):
        if y != 5:  # Leave doorway
            world.add_object(WorldObject(ObjectType.WALL, Position3D(7, y, 0)))

    # Door between rooms
    world.add_object(WorldObject(
        ObjectType.DOOR,
        Position3D(7, 5, 0),
        is_interactive=True,
        state={"open": False},
    ))

    # Bedroom (left side)
    world.add_object(WorldObject(ObjectType.BED, Position3D(1, 1, 0), size=(2, 2, 1)))
    world.add_object(WorldObject(ObjectType.DESK, Position3D(1, 7, 0)))
    world.add_object(WorldObject(ObjectType.LAMP, Position3D(3, 1, 0), is_solid=False))

    # Kitchen (right side)
    world.add_object(WorldObject(ObjectType.FRIDGE, Position3D(12, 1, 0), size=(1, 1, 2)))
    world.add_object(WorldObject(ObjectType.STOVE, Position3D(10, 1, 0)))
    world.add_object(WorldObject(ObjectType.SINK, Position3D(12, 3, 0)))
    world.add_object(WorldObject(ObjectType.TABLE, Position3D(10, 6, 0)))

    # Robot starts in bedroom
    world.robot.position = Position3D(3, 5, 0)
    world.robot.rotation.yaw = 90  # Facing east

    # Goal in kitchen
    world.goal_position = Position3D(11, 5, 0)

    return world


def create_multi_floor_house() -> HomeWorld:
    """Create a house with stairs between floors."""
    world = HomeWorld(width=12, depth=12, height=6)
    world.level_id = "multi_floor"
    world.goal_description = "Go upstairs and find the book"

    # Ground floor (z=0)
    for x in range(12):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 11, 0)))
    for y in range(1, 11):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(11, y, 0)))

    # Stairs (connects floor 0 to floor 3)
    for i in range(3):
        world.add_object(WorldObject(
            ObjectType.STAIRS,
            Position3D(10, 5, i),
            is_solid=False,
            is_interactive=True,
            state={"connects_to": i + 1},
        ))

    # Ground floor furniture
    world.add_object(WorldObject(ObjectType.COUCH, Position3D(2, 2, 0), size=(2, 1, 1)))
    world.add_object(WorldObject(ObjectType.TV, Position3D(2, 5, 0), is_solid=False))

    # Upper floor (z=3)
    for x in range(12):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 3)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 11, 3)))
    for y in range(1, 11):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 3)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(11, y, 3)))

    # Upper floor bedroom
    world.add_object(WorldObject(ObjectType.BED, Position3D(2, 2, 3), size=(2, 2, 1)))
    world.add_object(WorldObject(ObjectType.SHELF, Position3D(2, 8, 3)))

    # Book (goal item)
    world.add_object(WorldObject(
        ObjectType.BOOK,
        Position3D(3, 8, 3),
        is_solid=False,
        is_collectible=True,
    ))

    # Robot starts on ground floor
    world.robot.position = Position3D(5, 5, 0)
    world.robot.rotation.yaw = 0

    # Goal is the book
    world.goal_position = Position3D(3, 8, 3)

    return world


# ============================================================================
# Curriculum Levels (Increasing Difficulty)
# ============================================================================

def create_empty_room() -> HomeWorld:
    """Level 0: Empty room - learn basic navigation."""
    world = HomeWorld(width=8, depth=8, height=3)
    world.level_id = "empty_room"
    world.goal_description = "Walk to the goal marker"

    # Walls
    for x in range(8):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 7, 0)))
    for y in range(1, 7):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(7, y, 0)))

    world.robot.position = Position3D(2, 5, 0)
    world.robot.rotation.yaw = 0
    world.goal_position = Position3D(5, 2, 0)

    return world


def create_obstacle_course() -> HomeWorld:
    """Level 1: Navigate around furniture."""
    world = HomeWorld(width=10, depth=10, height=3)
    world.level_id = "obstacle_course"
    world.goal_description = "Navigate around furniture to reach the goal"

    # Walls
    for x in range(10):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 9, 0)))
    for y in range(1, 9):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(9, y, 0)))

    # Obstacles
    world.add_object(WorldObject(ObjectType.TABLE, Position3D(3, 3, 0)))
    world.add_object(WorldObject(ObjectType.COUCH, Position3D(5, 5, 0), size=(2, 1, 1)))
    world.add_object(WorldObject(ObjectType.CHAIR, Position3D(7, 3, 0)))
    world.add_object(WorldObject(ObjectType.DESK, Position3D(2, 6, 0)))

    world.robot.position = Position3D(1, 1, 0)
    world.robot.rotation.yaw = 0
    world.goal_position = Position3D(8, 8, 0)

    return world


def create_door_puzzle() -> HomeWorld:
    """Level 2: Open a door to proceed."""
    world = HomeWorld(width=12, depth=8, height=3)
    world.level_id = "door_puzzle"
    world.goal_description = "Open the door and reach the other side"

    # Walls
    for x in range(12):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 7, 0)))
    for y in range(1, 7):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(11, y, 0)))

    # Dividing wall with door
    for y in range(1, 7):
        if y != 4:
            world.add_object(WorldObject(ObjectType.WALL, Position3D(6, y, 0)))

    # Door
    world.add_object(WorldObject(
        ObjectType.DOOR,
        Position3D(6, 4, 0),
        is_interactive=True,
        state={"open": False},
    ))

    world.robot.position = Position3D(2, 4, 0)
    world.robot.rotation.yaw = 90
    world.goal_position = Position3D(9, 4, 0)

    return world


def create_key_hunt() -> HomeWorld:
    """Level 3: Find a key to unlock the door."""
    world = HomeWorld(width=12, depth=12, height=3)
    world.level_id = "key_hunt"
    world.goal_description = "Find the key and unlock the door to escape"

    # Walls
    for x in range(12):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 11, 0)))
    for y in range(1, 11):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(11, y, 0)))

    # Internal walls creating L-shape
    for x in range(1, 7):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 6, 0)))
    for y in range(2, 6):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(6, y, 0)))

    # Furniture
    world.add_object(WorldObject(ObjectType.TABLE, Position3D(9, 3, 0)))
    world.add_object(WorldObject(ObjectType.COUCH, Position3D(2, 2, 0), size=(2, 1, 1)))

    # Key hidden in corner
    world.add_object(WorldObject(
        ObjectType.KEY,
        Position3D(10, 9, 0),
        is_solid=False,
        is_collectible=True,
    ))

    # Locked door to goal
    world.add_object(WorldObject(
        ObjectType.DOOR,
        Position3D(3, 0, 0),
        is_interactive=True,
        state={"open": False, "locked": True},
    ))

    world.robot.position = Position3D(9, 5, 0)
    world.robot.rotation.yaw = 180
    world.goal_position = Position3D(3, 1, 0)

    return world


def create_office_layout() -> HomeWorld:
    """Level 4: Navigate an office with desks and switches."""
    world = HomeWorld(width=15, depth=12, height=3)
    world.level_id = "office_layout"
    world.goal_description = "Navigate the office and reach the exit"

    # Walls
    for x in range(15):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 11, 0)))
    for y in range(1, 11):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(14, y, 0)))

    # Cubicle walls
    for x in range(3, 12, 4):
        for y in range(2, 9, 3):
            world.add_object(WorldObject(ObjectType.DESK, Position3D(x, y, 0)))
            world.add_object(WorldObject(ObjectType.CHAIR, Position3D(x + 1, y, 0)))

    # Light switches
    world.add_object(WorldObject(
        ObjectType.SWITCH,
        Position3D(1, 1, 0),
        is_solid=False,
        is_interactive=True,
        state={"on": True},
    ))

    # Some lamps
    world.add_object(WorldObject(ObjectType.LAMP, Position3D(7, 5, 0), is_solid=False))

    world.robot.position = Position3D(1, 5, 0)
    world.robot.rotation.yaw = 90
    world.goal_position = Position3D(13, 5, 0)

    return world


def create_living_room_kitchen() -> HomeWorld:
    """Level 5: Living room connected to kitchen."""
    world = HomeWorld(width=16, depth=12, height=3)
    world.level_id = "living_kitchen"
    world.goal_description = "Get the cup from the kitchen and bring it to the living room"

    # Outer walls
    for x in range(16):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 11, 0)))
    for y in range(1, 11):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(15, y, 0)))

    # Dividing wall with opening
    for y in range(1, 11):
        if y < 4 or y > 7:
            world.add_object(WorldObject(ObjectType.WALL, Position3D(8, y, 0)))

    # Living room (left)
    world.add_object(WorldObject(ObjectType.COUCH, Position3D(2, 2, 0), size=(2, 1, 1)))
    world.add_object(WorldObject(ObjectType.TV, Position3D(2, 8, 0), is_solid=False))
    world.add_object(WorldObject(ObjectType.TABLE, Position3D(5, 5, 0)))

    # Kitchen (right)
    world.add_object(WorldObject(ObjectType.FRIDGE, Position3D(12, 2, 0), size=(1, 1, 2)))
    world.add_object(WorldObject(ObjectType.STOVE, Position3D(10, 2, 0)))
    world.add_object(WorldObject(ObjectType.SINK, Position3D(14, 5, 0)))
    world.add_object(WorldObject(ObjectType.TABLE, Position3D(11, 8, 0)))

    # Cup to collect
    world.add_object(WorldObject(
        ObjectType.CUP,
        Position3D(11, 8, 0),
        is_solid=False,
        is_collectible=True,
    ))

    world.robot.position = Position3D(4, 5, 0)
    world.robot.rotation.yaw = 90
    world.goal_position = Position3D(5, 5, 0)  # Return to table

    return world


def create_bathroom_search() -> HomeWorld:
    """Level 6: Find the phone in the bathroom."""
    world = HomeWorld(width=10, depth=14, height=3)
    world.level_id = "bathroom_search"
    world.goal_description = "Find your phone that you left in the bathroom"

    # Outer walls
    for x in range(10):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 13, 0)))
    for y in range(1, 13):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(9, y, 0)))

    # Hallway and bathroom walls
    for x in range(1, 9):
        if x != 4:
            world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 8, 0)))

    # Door to bathroom
    world.add_object(WorldObject(
        ObjectType.DOOR,
        Position3D(4, 8, 0),
        is_interactive=True,
        state={"open": False},
    ))

    # Hallway furniture
    world.add_object(WorldObject(ObjectType.SHELF, Position3D(2, 3, 0)))
    world.add_object(WorldObject(ObjectType.TABLE, Position3D(6, 3, 0)))

    # Bathroom fixtures
    world.add_object(WorldObject(ObjectType.SINK, Position3D(2, 10, 0)))
    world.add_object(WorldObject(ObjectType.SHELF, Position3D(7, 11, 0)))

    # Phone to find
    world.add_object(WorldObject(
        ObjectType.PHONE,
        Position3D(7, 11, 0),
        is_solid=False,
        is_collectible=True,
    ))

    world.robot.position = Position3D(5, 2, 0)
    world.robot.rotation.yaw = 0
    world.goal_position = Position3D(5, 2, 0)  # Return to start with phone

    return world


def create_full_house() -> HomeWorld:
    """Level 7: Complete house with multiple rooms."""
    world = HomeWorld(width=20, depth=16, height=3)
    world.level_id = "full_house"
    world.goal_description = "Explore all rooms and find the remote control"

    # Outer walls
    for x in range(20):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 15, 0)))
    for y in range(1, 15):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(19, y, 0)))

    # Vertical divider (living/dining | bedrooms)
    for y in range(1, 15):
        if y not in (5, 10):
            world.add_object(WorldObject(ObjectType.WALL, Position3D(10, y, 0)))

    # Doors
    world.add_object(WorldObject(ObjectType.DOOR, Position3D(10, 5, 0),
                                 is_interactive=True, state={"open": False}))
    world.add_object(WorldObject(ObjectType.DOOR, Position3D(10, 10, 0),
                                 is_interactive=True, state={"open": False}))

    # Horizontal divider in right side (bedrooms)
    for x in range(11, 19):
        if x != 15:
            world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 8, 0)))
    world.add_object(WorldObject(ObjectType.DOOR, Position3D(15, 8, 0),
                                 is_interactive=True, state={"open": False}))

    # Living room (bottom left)
    world.add_object(WorldObject(ObjectType.COUCH, Position3D(2, 2, 0), size=(2, 1, 1)))
    world.add_object(WorldObject(ObjectType.TV, Position3D(2, 5, 0), is_solid=False))
    world.add_object(WorldObject(ObjectType.TABLE, Position3D(6, 3, 0)))

    # Dining room (top left)
    world.add_object(WorldObject(ObjectType.TABLE, Position3D(4, 10, 0), size=(2, 2, 1)))

    # Bedroom 1 (bottom right)
    world.add_object(WorldObject(ObjectType.BED, Position3D(14, 2, 0), size=(2, 2, 1)))
    world.add_object(WorldObject(ObjectType.DESK, Position3D(17, 2, 0)))

    # Bedroom 2 (top right)
    world.add_object(WorldObject(ObjectType.BED, Position3D(14, 11, 0), size=(2, 2, 1)))
    world.add_object(WorldObject(ObjectType.SHELF, Position3D(17, 13, 0)))

    # Remote to find (hidden in bedroom 2)
    world.add_object(WorldObject(
        ObjectType.REMOTE,
        Position3D(17, 13, 0),
        is_solid=False,
        is_collectible=True,
    ))

    world.robot.position = Position3D(5, 3, 0)
    world.robot.rotation.yaw = 0
    world.goal_position = Position3D(2, 5, 0)  # Return to TV area

    return world


# Factory function to create world from room scanner data
def create_from_scan(scan_data: Dict) -> "HomeWorld":
    """
    Create a HomeWorld from room scanner data.

    Args:
        scan_data: Dict with:
            - dimensions: {width, height, depth} in meters
            - room_type: Optional room type string
            - objects: List of detected objects with positions
            - textures: Optional dict of wall textures (base64)
            - coverage: Coverage data from scanner

    Returns:
        HomeWorld configured as a real2sim training environment
    """
    # Extract dimensions (convert from meters to grid units, 1 unit = 0.5m)
    dims = scan_data.get("dimensions", {"width": 5, "height": 3, "depth": 5})
    width = int(dims.get("width", 5) * 2)   # Convert to grid units
    depth = int(dims.get("depth", 5) * 2)
    height = int(dims.get("height", 3) * 2)

    # Clamp to reasonable sizes
    width = max(4, min(width, 40))
    depth = max(4, min(depth, 40))
    height = max(2, min(height, 10))

    # Determine room type
    room_type_str = scan_data.get("room_type", "living_room")
    try:
        room_type = RoomType(room_type_str)
    except ValueError:
        room_type = RoomType.LIVING_ROOM

    # Create world
    world = HomeWorld(
        width=width,
        depth=depth,
        height=height,
    )
    world.level_id = "scanned_room"
    world.goal_description = f"Explore scanned {room_type.value} and learn the layout"

    # Create the scanned room
    room = Room(
        room_type=room_type,
        bounds=(Position3D(0, 0, 0), Position3D(width, depth, height)),
    )
    world.add_room("main_room", room)

    # Add walls around the perimeter
    for x in range(width):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, 0, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(x, depth - 1, 0)))
    for y in range(depth):
        world.add_object(WorldObject(ObjectType.WALL, Position3D(0, y, 0)))
        world.add_object(WorldObject(ObjectType.WALL, Position3D(width - 1, y, 0)))

    # Add detected objects from scan
    detected_objects = scan_data.get("objects", [])
    for obj in detected_objects:
        obj_type_str = obj.get("type", "").lower()
        obj_pos = obj.get("position", {})

        # Map common object names to ObjectType
        type_mapping = {
            "couch": ObjectType.COUCH,
            "sofa": ObjectType.COUCH,
            "table": ObjectType.TABLE,
            "desk": ObjectType.DESK,
            "chair": ObjectType.CHAIR,
            "bed": ObjectType.BED,
            "tv": ObjectType.TV,
            "television": ObjectType.TV,
            "lamp": ObjectType.LAMP,
            "light": ObjectType.LAMP,
            "fridge": ObjectType.FRIDGE,
            "refrigerator": ObjectType.FRIDGE,
            "sink": ObjectType.SINK,
            "stove": ObjectType.STOVE,
            "oven": ObjectType.STOVE,
            "shelf": ObjectType.SHELF,
            "bookshelf": ObjectType.SHELF,
            "door": ObjectType.DOOR,
            "window": ObjectType.WINDOW,
        }

        if obj_type_str in type_mapping:
            # Convert position from meters to grid units
            pos_x = int(obj_pos.get("x", width // 2) * 2) % width
            pos_y = int(obj_pos.get("y", depth // 2) * 2) % depth
            pos_z = int(obj_pos.get("z", 0) * 2)

            world.add_object(WorldObject(
                type_mapping[obj_type_str],
                Position3D(pos_x, pos_y, pos_z)
            ))

    # Add some default furniture if no objects detected
    if not detected_objects:
        # Add a few items based on room type
        if room_type == RoomType.LIVING_ROOM:
            world.add_object(WorldObject(ObjectType.COUCH, Position3D(width // 2, depth // 2, 0)))
            world.add_object(WorldObject(ObjectType.TV, Position3D(width // 2, 1, 0)))
            world.add_object(WorldObject(ObjectType.LAMP, Position3D(2, 2, 0)))
        elif room_type == RoomType.KITCHEN:
            world.add_object(WorldObject(ObjectType.FRIDGE, Position3D(1, 1, 0)))
            world.add_object(WorldObject(ObjectType.STOVE, Position3D(3, 1, 0)))
            world.add_object(WorldObject(ObjectType.SINK, Position3D(5, 1, 0)))
        elif room_type == RoomType.BEDROOM:
            world.add_object(WorldObject(ObjectType.BED, Position3D(width // 2, depth // 2, 0)))
            world.add_object(WorldObject(ObjectType.DESK, Position3D(2, 2, 0)))

    # Store scan metadata for later use
    world.metadata = {
        "scan_data": scan_data,
        "real2sim": True,
        "original_dimensions_m": dims,
    }

    # Set robot start position (center of room)
    world.robot.position = Position3D(width // 2, depth // 2, 0)
    world.goal_position = Position3D(width - 2, depth - 2, 0)

    return world


# Dynamic level storage for scanned rooms
SCANNED_LEVELS: Dict[str, HomeWorld] = {}


def register_scanned_level(level_id: str, world: HomeWorld) -> str:
    """Register a scanned level for use."""
    SCANNED_LEVELS[level_id] = world
    return level_id


def get_scanned_level(level_id: str) -> Optional[HomeWorld]:
    """Get a registered scanned level."""
    return SCANNED_LEVELS.get(level_id)


# Built-in levels
LEVELS = {
    # Curriculum levels (0-7)
    "empty_room": create_empty_room,
    "obstacle_course": create_obstacle_course,
    "door_puzzle": create_door_puzzle,
    "key_hunt": create_key_hunt,
    "office_layout": create_office_layout,
    "living_kitchen": create_living_room_kitchen,
    "bathroom_search": create_bathroom_search,
    "full_house": create_full_house,
    # Original levels
    "simple_apartment": create_simple_apartment,
    "two_room_house": create_two_room_house,
    "multi_floor": create_multi_floor_house,
}


# Curriculum order for training
CURRICULUM_ORDER = [
    "empty_room",        # Level 0: Basic navigation
    "obstacle_course",   # Level 1: Avoid obstacles
    "door_puzzle",       # Level 2: Open doors
    "key_hunt",          # Level 3: Find items + doors
    "simple_apartment",  # Level 4: Combined skills
    "office_layout",     # Level 5: Complex navigation
    "living_kitchen",    # Level 6: Multi-room
    "bathroom_search",   # Level 7: Search and return
    "two_room_house",    # Level 8: Two rooms
    "full_house",        # Level 9: Full house
    "multi_floor",       # Level 10: Multi-floor
]


def get_level(level_id: str) -> HomeWorld:
    """Get a level by ID."""
    if level_id not in LEVELS:
        raise ValueError(f"Unknown level: {level_id}")
    return LEVELS[level_id]()


def list_levels() -> List[str]:
    """List available level IDs."""
    return list(LEVELS.keys())


if __name__ == "__main__":
    # Demo
    print("=== 3D Home Exploration Demo ===\n")

    world = create_two_room_house()
    print(f"Level: {world.level_id}")
    print(f"Goal: {world.goal_description}")
    print(f"\nTop-down view:\n")
    print(world.render_top_down())

    print(f"\nRobot at: ({world.robot.position.x}, {world.robot.position.y})")
    print(f"Facing: {world.robot.rotation.yaw}°")

    # Test movement
    print("\n=== Testing Movement ===")
    result = world.move_forward()
    print(f"Move forward: {result.message}")
    print(f"New position: ({world.robot.position.x:.1f}, {world.robot.position.y:.1f})")

    result = world.turn_right()
    print(f"Turn right: {result.message}")

    print(f"\nVisible objects:")
    for obj in world.get_visible_objects()[:5]:
        print(f"  - {obj['type']} at distance {obj['distance']:.1f}")
