#!/usr/bin/env python3
"""
Home Robot Training Games - Progressive Skill Development

Creates increasingly complex scenarios for training a general home robot:

TIER 1 - Basic Movement (Cycles 1-10)
  - Navigate empty rooms
  - Avoid static obstacles
  - Reach goal locations

TIER 2 - Object Interaction (Cycles 11-25)
  - Pick up items
  - Deliver objects to locations
  - Open doors

TIER 3 - Multi-Room Navigation (Cycles 26-50)
  - Navigate between rooms
  - Remember room layouts
  - Find specific rooms (kitchen, bedroom, etc.)

TIER 4 - Task Completion (Cycles 51-100)
  - "Get me a glass from the kitchen"
  - "Clean up the toys in the living room"
  - "Check if the front door is locked"

TIER 5 - Complex Scenarios (Cycles 100+)
  - Multi-step tasks with dependencies
  - Time-sensitive tasks
  - Handling interruptions
  - Human interaction scenarios

Usage:
    python brain_b/simulator/home_robot_games.py --tier 1
    python brain_b/simulator/home_robot_games.py --generate 20
    python brain_b/simulator/home_robot_games.py --play
"""

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any


class ObjectType(Enum):
    """Objects a home robot might interact with."""
    # Containers
    CUP = "cup"
    GLASS = "glass"
    BOWL = "bowl"
    PLATE = "plate"
    BOX = "box"
    BAG = "bag"

    # Furniture
    CHAIR = "chair"
    TABLE = "table"
    COUCH = "couch"
    BED = "bed"
    DESK = "desk"

    # Appliances
    TV = "tv"
    LAMP = "lamp"
    FRIDGE = "fridge"
    MICROWAVE = "microwave"

    # Items
    BOOK = "book"
    REMOTE = "remote"
    PHONE = "phone"
    KEYS = "keys"
    TOY = "toy"
    PILLOW = "pillow"
    BLANKET = "blanket"
    TRASH = "trash"

    # Fixtures
    DOOR = "door"
    WINDOW = "window"
    LIGHT_SWITCH = "light_switch"


class RoomType(Enum):
    """Types of rooms in a home."""
    LIVING_ROOM = "living_room"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    HALLWAY = "hallway"
    DINING_ROOM = "dining_room"
    OFFICE = "office"
    GARAGE = "garage"
    ENTRANCE = "entrance"


class ActionType(Enum):
    """Actions the robot can take."""
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    PICK_UP = "pick_up"
    PUT_DOWN = "put_down"
    OPEN = "open"
    CLOSE = "close"
    PUSH = "push"
    TOGGLE = "toggle"  # For switches/buttons
    SPEAK = "speak"
    WAIT = "wait"
    SCAN = "scan"  # Look around


class TaskType(Enum):
    """Types of tasks for the robot."""
    # Tier 1
    NAVIGATE_TO_POINT = "navigate_to_point"
    AVOID_OBSTACLES = "avoid_obstacles"
    EXPLORE_ROOM = "explore_room"

    # Tier 2
    FETCH_OBJECT = "fetch_object"
    DELIVER_OBJECT = "deliver_object"
    OPEN_DOOR = "open_door"
    TOGGLE_SWITCH = "toggle_switch"

    # Tier 3
    FIND_ROOM = "find_room"
    PATROL_ROOMS = "patrol_rooms"
    SEARCH_FOR_OBJECT = "search_for_object"

    # Tier 4
    CLEAN_ROOM = "clean_room"
    SET_TABLE = "set_table"
    TIDY_UP = "tidy_up"
    CHECK_SECURITY = "check_security"

    # Tier 5
    ASSIST_HUMAN = "assist_human"
    HANDLE_EMERGENCY = "handle_emergency"
    MULTI_STEP_CHORE = "multi_step_chore"


@dataclass
class GameObject:
    """An object in the game world."""
    id: str
    type: ObjectType
    x: int
    y: int
    room: str
    pickable: bool = True
    interactable: bool = True
    state: Dict = field(default_factory=dict)  # e.g., {"open": False}


@dataclass
class Room:
    """A room in the home."""
    name: str
    type: RoomType
    x: int  # Top-left corner
    y: int
    width: int
    height: int
    objects: List[GameObject] = field(default_factory=list)
    doors: List[Tuple[int, int, str]] = field(default_factory=list)  # (x, y, connects_to)


@dataclass
class Task:
    """A task for the robot to complete."""
    id: str
    type: TaskType
    tier: int
    description: str
    objectives: List[Dict]  # List of sub-objectives
    hints: List[str] = field(default_factory=list)
    time_limit: Optional[int] = None  # seconds
    reward: int = 10

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "tier": self.tier,
            "description": self.description,
            "objectives": self.objectives,
            "hints": self.hints,
            "time_limit": self.time_limit,
            "reward": self.reward,
        }


@dataclass
class GameState:
    """Current state of a game."""
    rooms: List[Room]
    robot_x: int
    robot_y: int
    robot_dir: str  # "north", "south", "east", "west"
    robot_room: str
    inventory: List[GameObject]
    task: Task
    score: int = 0
    steps: int = 0
    completed_objectives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "robot": {
                "x": self.robot_x,
                "y": self.robot_y,
                "dir": self.robot_dir,
                "room": self.robot_room,
                "inventory": [o.type.value for o in self.inventory],
            },
            "rooms": [
                {
                    "name": r.name,
                    "type": r.type.value,
                    "bounds": [r.x, r.y, r.width, r.height],
                    "objects": [{"type": o.type.value, "x": o.x, "y": o.y} for o in r.objects],
                }
                for r in self.rooms
            ],
            "task": self.task.to_dict(),
            "score": self.score,
            "steps": self.steps,
            "completed": self.completed_objectives,
        }


class HomeRobotGameGenerator:
    """Generates progressive training games for home robots."""

    # Room templates with typical objects
    ROOM_TEMPLATES = {
        RoomType.LIVING_ROOM: {
            "size": (8, 6),
            "objects": [ObjectType.COUCH, ObjectType.TV, ObjectType.LAMP, ObjectType.REMOTE, ObjectType.PILLOW],
        },
        RoomType.KITCHEN: {
            "size": (6, 5),
            "objects": [ObjectType.FRIDGE, ObjectType.TABLE, ObjectType.CUP, ObjectType.PLATE, ObjectType.BOWL],
        },
        RoomType.BEDROOM: {
            "size": (6, 5),
            "objects": [ObjectType.BED, ObjectType.LAMP, ObjectType.PILLOW, ObjectType.BLANKET, ObjectType.PHONE],
        },
        RoomType.BATHROOM: {
            "size": (4, 4),
            "objects": [ObjectType.LIGHT_SWITCH],
        },
        RoomType.HALLWAY: {
            "size": (3, 8),
            "objects": [ObjectType.LIGHT_SWITCH],
        },
        RoomType.OFFICE: {
            "size": (5, 5),
            "objects": [ObjectType.DESK, ObjectType.CHAIR, ObjectType.LAMP, ObjectType.BOOK, ObjectType.PHONE],
        },
        RoomType.ENTRANCE: {
            "size": (4, 4),
            "objects": [ObjectType.DOOR, ObjectType.KEYS, ObjectType.LIGHT_SWITCH],
        },
    }

    def __init__(self):
        self.game_count = 0

    def generate_tier1_game(self, difficulty: int = 1) -> GameState:
        """
        Tier 1: Basic Movement
        - Navigate to goal
        - Avoid obstacles
        - Simple room exploration
        """
        # Single room
        room_type = random.choice([RoomType.LIVING_ROOM, RoomType.KITCHEN, RoomType.BEDROOM])
        template = self.ROOM_TEMPLATES[room_type]

        width, height = template["size"]
        width += difficulty * 2  # Larger rooms at higher difficulty
        height += difficulty

        room = Room(
            name=room_type.value,
            type=room_type,
            x=0, y=0,
            width=width, height=height,
        )

        # Add obstacles based on difficulty
        num_obstacles = difficulty * 2
        for i in range(num_obstacles):
            obj_type = random.choice([ObjectType.CHAIR, ObjectType.BOX, ObjectType.TABLE])
            obj = GameObject(
                id=f"obstacle_{i}",
                type=obj_type,
                x=random.randint(2, width-2),
                y=random.randint(2, height-2),
                room=room.name,
                pickable=False,
            )
            room.objects.append(obj)

        # Robot starts in corner
        robot_x, robot_y = 1, 1

        # Goal is opposite corner
        goal_x, goal_y = width - 2, height - 2

        # Create task
        task_type = random.choice([
            TaskType.NAVIGATE_TO_POINT,
            TaskType.AVOID_OBSTACLES,
            TaskType.EXPLORE_ROOM,
        ])

        if task_type == TaskType.NAVIGATE_TO_POINT:
            task = Task(
                id=f"tier1_{self.game_count}",
                type=task_type,
                tier=1,
                description=f"Navigate to position ({goal_x}, {goal_y})",
                objectives=[{"type": "reach", "x": goal_x, "y": goal_y}],
                hints=["Move forward to advance", "Turn to change direction"],
                reward=10 + difficulty * 5,
            )
        elif task_type == TaskType.AVOID_OBSTACLES:
            task = Task(
                id=f"tier1_{self.game_count}",
                type=task_type,
                tier=1,
                description=f"Reach the goal without hitting obstacles",
                objectives=[
                    {"type": "reach", "x": goal_x, "y": goal_y},
                    {"type": "no_collisions"},
                ],
                hints=["Watch for furniture", "Plan your path"],
                reward=15 + difficulty * 5,
            )
        else:
            task = Task(
                id=f"tier1_{self.game_count}",
                type=task_type,
                tier=1,
                description=f"Explore the entire {room_type.value}",
                objectives=[{"type": "visit_all_tiles"}],
                hints=["Cover the whole room", "Don't miss corners"],
                reward=20 + difficulty * 5,
            )

        self.game_count += 1

        return GameState(
            rooms=[room],
            robot_x=robot_x,
            robot_y=robot_y,
            robot_dir="east",
            robot_room=room.name,
            inventory=[],
            task=task,
        )

    def generate_tier2_game(self, difficulty: int = 1) -> GameState:
        """
        Tier 2: Object Interaction
        - Pick up and deliver objects
        - Open doors
        - Toggle switches
        """
        room_type = random.choice([RoomType.LIVING_ROOM, RoomType.KITCHEN, RoomType.BEDROOM])
        template = self.ROOM_TEMPLATES[room_type]

        width, height = template["size"]

        room = Room(
            name=room_type.value,
            type=room_type,
            x=0, y=0,
            width=width, height=height,
        )

        # Add interactable objects
        target_objects = random.sample(template["objects"], min(3, len(template["objects"])))
        for i, obj_type in enumerate(target_objects):
            obj = GameObject(
                id=f"obj_{i}",
                type=obj_type,
                x=random.randint(2, width-2),
                y=random.randint(2, height-2),
                room=room.name,
                pickable=obj_type not in [ObjectType.COUCH, ObjectType.BED, ObjectType.FRIDGE, ObjectType.TV],
            )
            room.objects.append(obj)

        # Create task
        task_type = random.choice([
            TaskType.FETCH_OBJECT,
            TaskType.DELIVER_OBJECT,
            TaskType.TOGGLE_SWITCH,
        ])

        pickable_objects = [o for o in room.objects if o.pickable]

        if task_type == TaskType.FETCH_OBJECT and pickable_objects:
            target = random.choice(pickable_objects)
            task = Task(
                id=f"tier2_{self.game_count}",
                type=task_type,
                tier=2,
                description=f"Pick up the {target.type.value}",
                objectives=[{"type": "pick_up", "object": target.type.value}],
                hints=["Navigate to the object", "Use pick_up action when close"],
                reward=20 + difficulty * 10,
            )
        elif task_type == TaskType.DELIVER_OBJECT and pickable_objects:
            target = random.choice(pickable_objects)
            dest_x, dest_y = random.randint(1, width-2), random.randint(1, height-2)
            task = Task(
                id=f"tier2_{self.game_count}",
                type=task_type,
                tier=2,
                description=f"Bring the {target.type.value} to ({dest_x}, {dest_y})",
                objectives=[
                    {"type": "pick_up", "object": target.type.value},
                    {"type": "deliver_to", "x": dest_x, "y": dest_y},
                ],
                hints=["First pick up the object", "Then carry it to the destination"],
                reward=30 + difficulty * 10,
            )
        else:
            # Toggle switch
            switch = GameObject(
                id="switch_1",
                type=ObjectType.LIGHT_SWITCH,
                x=1, y=height//2,
                room=room.name,
                pickable=False,
                state={"on": False},
            )
            room.objects.append(switch)

            task = Task(
                id=f"tier2_{self.game_count}",
                type=TaskType.TOGGLE_SWITCH,
                tier=2,
                description="Turn on the light switch",
                objectives=[{"type": "toggle", "object": "light_switch", "target_state": True}],
                hints=["Find the switch on the wall", "Use toggle action"],
                reward=15 + difficulty * 10,
            )

        self.game_count += 1

        return GameState(
            rooms=[room],
            robot_x=1,
            robot_y=1,
            robot_dir="east",
            robot_room=room.name,
            inventory=[],
            task=task,
        )

    def generate_tier3_game(self, difficulty: int = 1) -> GameState:
        """
        Tier 3: Multi-Room Navigation
        - Navigate between connected rooms
        - Find specific rooms
        - Search for objects across rooms
        """
        # Generate 2-4 connected rooms
        num_rooms = 2 + difficulty
        rooms = []

        # Layout: linear for now
        x_offset = 0
        room_types = random.sample(list(RoomType), min(num_rooms, len(RoomType)))

        for i, room_type in enumerate(room_types):
            template = self.ROOM_TEMPLATES.get(room_type, {"size": (5, 5), "objects": []})
            width, height = template["size"]

            room = Room(
                name=f"{room_type.value}_{i}",
                type=room_type,
                x=x_offset, y=0,
                width=width, height=height,
            )

            # Add typical objects
            for j, obj_type in enumerate(template.get("objects", [])[:3]):
                obj = GameObject(
                    id=f"{room.name}_obj_{j}",
                    type=obj_type,
                    x=x_offset + random.randint(1, width-2),
                    y=random.randint(1, height-2),
                    room=room.name,
                    pickable=obj_type not in [ObjectType.COUCH, ObjectType.BED, ObjectType.FRIDGE, ObjectType.TV],
                )
                room.objects.append(obj)

            # Add door to next room
            if i < num_rooms - 1:
                room.doors.append((x_offset + width - 1, height // 2, f"{room_types[i+1].value}_{i+1}"))

            rooms.append(room)
            x_offset += width

        # Create task
        task_type = random.choice([
            TaskType.FIND_ROOM,
            TaskType.SEARCH_FOR_OBJECT,
            TaskType.PATROL_ROOMS,
        ])

        target_room = random.choice(rooms[1:])  # Not starting room

        if task_type == TaskType.FIND_ROOM:
            task = Task(
                id=f"tier3_{self.game_count}",
                type=task_type,
                tier=3,
                description=f"Find and enter the {target_room.type.value}",
                objectives=[{"type": "enter_room", "room_type": target_room.type.value}],
                hints=["Navigate through doors", "Look for room signs"],
                reward=40 + difficulty * 15,
            )
        elif task_type == TaskType.SEARCH_FOR_OBJECT:
            # Find an object in another room
            target_objs = [o for r in rooms[1:] for o in r.objects if o.pickable]
            if target_objs:
                target = random.choice(target_objs)
                task = Task(
                    id=f"tier3_{self.game_count}",
                    type=task_type,
                    tier=3,
                    description=f"Find and retrieve the {target.type.value}",
                    objectives=[
                        {"type": "find", "object": target.type.value},
                        {"type": "pick_up", "object": target.type.value},
                    ],
                    hints=["Search multiple rooms", "The object might be anywhere"],
                    reward=50 + difficulty * 15,
                )
            else:
                task = Task(
                    id=f"tier3_{self.game_count}",
                    type=TaskType.PATROL_ROOMS,
                    tier=3,
                    description="Visit all rooms in the house",
                    objectives=[{"type": "visit_room", "room": r.name} for r in rooms],
                    hints=["Go through each doorway", "Make sure to enter each room"],
                    reward=45 + difficulty * 15,
                )
        else:
            task = Task(
                id=f"tier3_{self.game_count}",
                type=TaskType.PATROL_ROOMS,
                tier=3,
                description="Visit all rooms in the house",
                objectives=[{"type": "visit_room", "room": r.name} for r in rooms],
                hints=["Go through each doorway", "Make sure to enter each room"],
                reward=45 + difficulty * 15,
            )

        self.game_count += 1

        return GameState(
            rooms=rooms,
            robot_x=rooms[0].x + 1,
            robot_y=1,
            robot_dir="east",
            robot_room=rooms[0].name,
            inventory=[],
            task=task,
        )

    def generate_tier4_game(self, difficulty: int = 1) -> GameState:
        """
        Tier 4: Task Completion
        - Complex household tasks
        - Multiple steps required
        - Real-world scenarios
        """
        # Generate a home layout
        rooms = self._generate_home_layout()

        # Create task
        task_type = random.choice([
            TaskType.CLEAN_ROOM,
            TaskType.TIDY_UP,
            TaskType.CHECK_SECURITY,
            TaskType.SET_TABLE,
        ])

        if task_type == TaskType.CLEAN_ROOM:
            # Add trash to a room
            target_room = random.choice([r for r in rooms if r.type in [RoomType.LIVING_ROOM, RoomType.BEDROOM]])
            num_trash = 2 + difficulty
            for i in range(num_trash):
                trash = GameObject(
                    id=f"trash_{i}",
                    type=ObjectType.TRASH,
                    x=target_room.x + random.randint(1, target_room.width-2),
                    y=random.randint(1, target_room.height-2),
                    room=target_room.name,
                )
                target_room.objects.append(trash)

            task = Task(
                id=f"tier4_{self.game_count}",
                type=task_type,
                tier=4,
                description=f"Clean up all the trash in the {target_room.type.value}",
                objectives=[{"type": "collect_all", "object": "trash", "room": target_room.name}],
                hints=["Find all trash items", "Pick them up one by one"],
                reward=60 + difficulty * 20,
                time_limit=120 + difficulty * 30,
            )

        elif task_type == TaskType.TIDY_UP:
            # Toys scattered, need to put in a box
            living = next((r for r in rooms if r.type == RoomType.LIVING_ROOM), rooms[0])

            # Add toys
            for i in range(3):
                toy = GameObject(
                    id=f"toy_{i}",
                    type=ObjectType.TOY,
                    x=living.x + random.randint(1, living.width-2),
                    y=random.randint(1, living.height-2),
                    room=living.name,
                )
                living.objects.append(toy)

            # Add toy box
            box = GameObject(
                id="toy_box",
                type=ObjectType.BOX,
                x=living.x + living.width - 2,
                y=living.height - 2,
                room=living.name,
                pickable=False,
            )
            living.objects.append(box)

            task = Task(
                id=f"tier4_{self.game_count}",
                type=task_type,
                tier=4,
                description="Put all the toys in the toy box",
                objectives=[
                    {"type": "pick_up", "object": "toy"},
                    {"type": "deliver_to_object", "target": "box"},
                ] * 3,
                hints=["Pick up toys one at a time", "Bring each to the box"],
                reward=70 + difficulty * 20,
            )

        elif task_type == TaskType.CHECK_SECURITY:
            # Check all doors are closed
            entrance = next((r for r in rooms if r.type == RoomType.ENTRANCE), rooms[0])

            # Add doors
            doors = []
            for i, r in enumerate(rooms[:3]):
                door = GameObject(
                    id=f"door_{i}",
                    type=ObjectType.DOOR,
                    x=r.x + r.width - 1,
                    y=r.height // 2,
                    room=r.name,
                    pickable=False,
                    state={"open": random.choice([True, False])},
                )
                r.objects.append(door)
                doors.append(door)

            task = Task(
                id=f"tier4_{self.game_count}",
                type=task_type,
                tier=4,
                description="Check all doors and close any that are open",
                objectives=[
                    {"type": "check_and_close", "object": "door", "id": d.id}
                    for d in doors
                ],
                hints=["Visit each door", "Close it if open"],
                reward=50 + difficulty * 20,
            )

        else:  # SET_TABLE
            kitchen = next((r for r in rooms if r.type == RoomType.KITCHEN), rooms[0])
            dining = next((r for r in rooms if r.type == RoomType.DINING_ROOM), kitchen)

            # Items to move
            items = [ObjectType.PLATE, ObjectType.CUP, ObjectType.BOWL]
            for i, item_type in enumerate(items):
                item = GameObject(
                    id=f"table_item_{i}",
                    type=item_type,
                    x=kitchen.x + random.randint(1, kitchen.width-2),
                    y=random.randint(1, kitchen.height-2),
                    room=kitchen.name,
                )
                kitchen.objects.append(item)

            task = Task(
                id=f"tier4_{self.game_count}",
                type=task_type,
                tier=4,
                description="Set the table: bring plates, cups, and bowls from the kitchen",
                objectives=[
                    {"type": "transport", "object": item_type.value, "from": kitchen.name, "to": dining.name}
                    for item_type in items
                ],
                hints=["Get items from kitchen", "Bring to dining area"],
                reward=80 + difficulty * 20,
                time_limit=180,
            )

        self.game_count += 1

        return GameState(
            rooms=rooms,
            robot_x=rooms[0].x + 1,
            robot_y=1,
            robot_dir="east",
            robot_room=rooms[0].name,
            inventory=[],
            task=task,
        )

    def generate_tier5_game(self, difficulty: int = 1) -> GameState:
        """
        Tier 5: Complex Scenarios
        - Multi-step tasks with dependencies
        - Human interaction
        - Emergency response
        """
        rooms = self._generate_home_layout()

        task_type = random.choice([
            TaskType.ASSIST_HUMAN,
            TaskType.MULTI_STEP_CHORE,
            TaskType.HANDLE_EMERGENCY,
        ])

        if task_type == TaskType.ASSIST_HUMAN:
            # Human requests something
            requests = [
                ("bring_drink", "Please bring me a glass of water", [
                    {"type": "go_to", "room": "kitchen"},
                    {"type": "pick_up", "object": "glass"},
                    {"type": "go_to", "room": "living_room"},
                    {"type": "deliver_to", "target": "human"},
                ]),
                ("find_remote", "I can't find the TV remote", [
                    {"type": "search", "object": "remote"},
                    {"type": "pick_up", "object": "remote"},
                    {"type": "deliver_to", "target": "human"},
                ]),
                ("get_blanket", "I'm cold, can you get me a blanket?", [
                    {"type": "go_to", "room": "bedroom"},
                    {"type": "pick_up", "object": "blanket"},
                    {"type": "go_to", "room": "living_room"},
                    {"type": "deliver_to", "target": "human"},
                ]),
            ]

            req_id, description, objectives = random.choice(requests)

            task = Task(
                id=f"tier5_{self.game_count}",
                type=task_type,
                tier=5,
                description=description,
                objectives=objectives,
                hints=["Listen to the human's request", "Complete each step in order"],
                reward=100 + difficulty * 30,
                time_limit=300,
            )

        elif task_type == TaskType.MULTI_STEP_CHORE:
            task = Task(
                id=f"tier5_{self.game_count}",
                type=task_type,
                tier=5,
                description="Morning routine: open curtains, turn on lights, check doors",
                objectives=[
                    {"type": "toggle", "object": "window", "target_state": "open"},
                    {"type": "toggle", "object": "light_switch", "target_state": True},
                    {"type": "check_and_close", "object": "door"},
                    {"type": "report", "to": "human", "message": "Morning check complete"},
                ],
                hints=["Do each task in a logical order", "Report when done"],
                reward=120 + difficulty * 30,
                time_limit=240,
            )

        else:  # HANDLE_EMERGENCY
            emergencies = [
                ("spill", "There's a spill in the kitchen!", [
                    {"type": "go_to", "room": "kitchen"},
                    {"type": "locate", "object": "spill"},
                    {"type": "clean", "object": "spill"},
                    {"type": "report", "message": "Spill cleaned up"},
                ]),
                ("door_alert", "Front door left open!", [
                    {"type": "go_to", "room": "entrance"},
                    {"type": "close", "object": "door"},
                    {"type": "report", "message": "Door secured"},
                ]),
            ]

            emg_id, description, objectives = random.choice(emergencies)

            task = Task(
                id=f"tier5_{self.game_count}",
                type=task_type,
                tier=5,
                description=description,
                objectives=objectives,
                hints=["Respond quickly", "Safety first"],
                reward=150 + difficulty * 30,
                time_limit=60,
            )

        self.game_count += 1

        return GameState(
            rooms=rooms,
            robot_x=rooms[0].x + 1,
            robot_y=1,
            robot_dir="east",
            robot_room=rooms[0].name,
            inventory=[],
            task=task,
        )

    def _generate_home_layout(self) -> List[Room]:
        """Generate a realistic home layout."""
        rooms = []

        # Standard home: entrance -> hallway -> living/kitchen/bedroom
        layouts = [
            [RoomType.ENTRANCE, RoomType.HALLWAY, RoomType.LIVING_ROOM, RoomType.KITCHEN],
            [RoomType.ENTRANCE, RoomType.LIVING_ROOM, RoomType.KITCHEN, RoomType.BEDROOM],
            [RoomType.HALLWAY, RoomType.LIVING_ROOM, RoomType.DINING_ROOM, RoomType.KITCHEN],
        ]

        layout = random.choice(layouts)
        x_offset = 0

        for i, room_type in enumerate(layout):
            template = self.ROOM_TEMPLATES.get(room_type, {"size": (5, 5), "objects": []})
            width, height = template["size"]

            room = Room(
                name=f"{room_type.value}",
                type=room_type,
                x=x_offset, y=0,
                width=width, height=height,
            )

            # Add objects
            for j, obj_type in enumerate(template.get("objects", [])[:4]):
                obj = GameObject(
                    id=f"{room.name}_obj_{j}",
                    type=obj_type,
                    x=x_offset + random.randint(1, max(1, width-2)),
                    y=random.randint(1, max(1, height-2)),
                    room=room.name,
                    pickable=obj_type not in [ObjectType.COUCH, ObjectType.BED, ObjectType.FRIDGE, ObjectType.TV, ObjectType.TABLE],
                )
                room.objects.append(obj)

            rooms.append(room)
            x_offset += width

        return rooms

    def generate_game(self, tier: int, difficulty: int = 1) -> GameState:
        """Generate a game for the specified tier."""
        if tier == 1:
            return self.generate_tier1_game(difficulty)
        elif tier == 2:
            return self.generate_tier2_game(difficulty)
        elif tier == 3:
            return self.generate_tier3_game(difficulty)
        elif tier == 4:
            return self.generate_tier4_game(difficulty)
        else:
            return self.generate_tier5_game(difficulty)

    def generate_progressive_curriculum(self, total_games: int = 100) -> List[GameState]:
        """Generate a curriculum of progressively harder games."""
        games = []

        # Distribution across tiers
        tier_distribution = [
            (1, 0.3),   # 30% tier 1
            (2, 0.25),  # 25% tier 2
            (3, 0.20),  # 20% tier 3
            (4, 0.15),  # 15% tier 4
            (5, 0.10),  # 10% tier 5
        ]

        for tier, ratio in tier_distribution:
            num_games = int(total_games * ratio)
            for i in range(num_games):
                # Difficulty increases within tier
                difficulty = 1 + (i * 3 // num_games)
                game = self.generate_game(tier, difficulty)
                games.append(game)

        # Shuffle to mix tiers (but keep overall progression)
        # Group into chunks and shuffle within chunks
        chunk_size = 10
        for i in range(0, len(games), chunk_size):
            chunk = games[i:i+chunk_size]
            random.shuffle(chunk)
            games[i:i+chunk_size] = chunk

        return games


def save_games_as_rlds(games: List[GameState], output_dir: str = "continuonbrain/rlds/episodes"):
    """Save games as RLDS episodes for training."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, game in enumerate(games):
        episode_dir = output_path / f"home_robot_game_{i:04d}"
        episode_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata = {
            "source": "home_robot_games",
            "tier": game.task.tier,
            "task_type": game.task.type.value,
            "description": game.task.description,
            "timestamp": datetime.now().isoformat(),
        }

        with open(episode_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save initial state
        with open(episode_dir / "initial_state.json", 'w') as f:
            json.dump(game.to_dict(), f, indent=2)

        # Save task details
        with open(episode_dir / "task.json", 'w') as f:
            json.dump(game.task.to_dict(), f, indent=2)

    print(f"Saved {len(games)} games to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Home Robot Training Games")
    parser.add_argument("--tier", type=int, default=0, help="Generate games for specific tier (1-5)")
    parser.add_argument("--generate", type=int, default=0, help="Generate N games")
    parser.add_argument("--difficulty", type=int, default=1, help="Difficulty level (1-3)")
    parser.add_argument("--save", action="store_true", help="Save as RLDS episodes")
    parser.add_argument("--show", action="store_true", help="Show game details")

    args = parser.parse_args()

    generator = HomeRobotGameGenerator()

    if args.generate > 0:
        print(f"Generating {args.generate} progressive games...")
        games = generator.generate_progressive_curriculum(args.generate)

        # Summary
        tier_counts = {}
        for g in games:
            tier = g.task.tier
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        print(f"\nGenerated {len(games)} games:")
        for tier in sorted(tier_counts.keys()):
            print(f"  Tier {tier}: {tier_counts[tier]} games")

        if args.save:
            save_games_as_rlds(games)

        if args.show:
            for g in games[:5]:
                print(f"\n--- {g.task.type.value} (Tier {g.task.tier}) ---")
                print(f"Description: {g.task.description}")
                print(f"Objectives: {len(g.task.objectives)}")
                print(f"Reward: {g.task.reward}")

    elif args.tier > 0:
        print(f"Generating tier {args.tier} game (difficulty {args.difficulty})...")
        game = generator.generate_game(args.tier, args.difficulty)

        print(f"\n{'='*50}")
        print(f"Task: {game.task.description}")
        print(f"Type: {game.task.type.value}")
        print(f"Tier: {game.task.tier}")
        print(f"Reward: {game.task.reward}")
        print(f"Objectives: {len(game.task.objectives)}")
        for i, obj in enumerate(game.task.objectives):
            print(f"  {i+1}. {obj}")
        print(f"Hints: {game.task.hints}")
        print(f"\nRooms: {[r.name for r in game.rooms]}")
        print(f"Robot at: ({game.robot_x}, {game.robot_y}) facing {game.robot_dir}")

        if args.save:
            save_games_as_rlds([game])

    else:
        # Demo: generate one of each tier
        print("Demo: Generating one game per tier\n")

        for tier in range(1, 6):
            game = generator.generate_game(tier, difficulty=2)
            print(f"Tier {tier}: {game.task.description}")
            print(f"  Type: {game.task.type.value}")
            print(f"  Reward: {game.task.reward}")
            print()


if __name__ == "__main__":
    main()
