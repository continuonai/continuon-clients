"""
Tests for RobotGrid world and game mechanics.
"""

import pytest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulator.world import GridWorld, Robot, Tile, Direction, load_level, LEVELS


class TestDirection:
    """Test Direction enum."""

    def test_turn_left(self):
        assert Direction.NORTH.turn_left() == Direction.WEST
        assert Direction.WEST.turn_left() == Direction.SOUTH
        assert Direction.SOUTH.turn_left() == Direction.EAST
        assert Direction.EAST.turn_left() == Direction.NORTH

    def test_turn_right(self):
        assert Direction.NORTH.turn_right() == Direction.EAST
        assert Direction.EAST.turn_right() == Direction.SOUTH
        assert Direction.SOUTH.turn_right() == Direction.WEST
        assert Direction.WEST.turn_right() == Direction.NORTH

    def test_delta(self):
        assert Direction.NORTH.delta() == (0, -1)
        assert Direction.SOUTH.delta() == (0, 1)
        assert Direction.EAST.delta() == (1, 0)
        assert Direction.WEST.delta() == (-1, 0)


class TestTile:
    """Test Tile enum properties."""

    def test_walkable(self):
        assert Tile.FLOOR.walkable
        assert Tile.KEY.walkable
        assert Tile.GOAL.walkable
        assert not Tile.WALL.walkable
        assert not Tile.LAVA.walkable

    def test_restricted(self):
        assert Tile.LAVA.restricted
        assert not Tile.FLOOR.restricted
        assert not Tile.WALL.restricted

    def test_collectible(self):
        assert Tile.KEY.collectible
        assert not Tile.FLOOR.collectible


class TestGridWorld:
    """Test GridWorld game mechanics."""

    def test_load_level(self):
        world = load_level("tutorial")
        assert world.level_name == "Tutorial"
        assert world.width > 0
        assert world.height > 0

    def test_all_levels_load(self):
        for level_id in LEVELS:
            world = load_level(level_id)
            assert world is not None
            assert world.robot is not None

    def test_move_forward(self):
        world = GridWorld()
        world.load_level("""
            ....
            .S..
            ....
        """)
        # Robot starts facing EAST
        result = world.move_forward()
        assert result.success
        assert world.robot.x == 2  # Moved right

    def test_wall_collision(self):
        world = GridWorld()
        world.load_level("""
            ###
            #S#
            ###
        """)
        result = world.move_forward()
        assert not result.success
        assert "wall" in result.message.lower()

    def test_lava_restriction(self):
        world = GridWorld()
        world.load_level("""
            .~..
            .S..
            ....
        """)
        world.robot.direction = Direction.NORTH
        result = world.move_forward()
        assert not result.success
        assert result.sandbox_denied
        assert "SANDBOX DENIED" in result.message

    def test_key_collection(self):
        world = GridWorld()
        world.load_level("""
            .K..
            .S..
            ....
        """)
        world.robot.direction = Direction.NORTH
        result = world.move_forward()
        assert result.success
        assert "key" in world.robot.inventory
        assert result.item_collected == "key"

    def test_door_without_key(self):
        world = GridWorld()
        world.load_level("""
            .D..
            .S..
            ....
        """)
        world.robot.direction = Direction.NORTH
        result = world.move_forward()
        assert not result.success
        assert "locked" in result.message.lower()

    def test_door_with_key(self):
        world = GridWorld()
        world.load_level("""
            .D..
            .S..
            ....
        """)
        world.robot.inventory.append("key")
        world.robot.direction = Direction.NORTH
        result = world.move_forward()
        assert result.success
        assert "key" not in world.robot.inventory  # Key consumed

    def test_goal_completion(self):
        world = GridWorld()
        world.load_level("""
            .G..
            .S..
            ....
        """)
        world.robot.direction = Direction.NORTH
        result = world.move_forward()
        assert result.success
        assert result.level_complete

    def test_turn_left(self):
        world = GridWorld()
        world.robot = Robot(0, 0, Direction.NORTH)
        world.turn_left()
        assert world.robot.direction == Direction.WEST

    def test_turn_right(self):
        world = GridWorld()
        world.robot = Robot(0, 0, Direction.NORTH)
        world.turn_right()
        assert world.robot.direction == Direction.EAST

    def test_look(self):
        world = GridWorld()
        world.load_level("""
            .K..
            .S..
            ....
        """)
        world.robot.direction = Direction.NORTH
        desc = world.look()
        assert "key" in desc.lower()

    def test_where_am_i(self):
        world = GridWorld()
        world.robot = Robot(5, 3, Direction.EAST)
        world.robot.inventory.append("key")
        info = world.where_am_i()
        assert "5" in info
        assert "3" in info
        assert "EAST" in info
        assert "key" in info

    def test_serialization(self):
        world = load_level("tutorial")
        world.robot.inventory.append("key")
        world.move_forward()

        data = world.to_dict()
        assert "grid" in data
        assert "robot" in data

        world2 = GridWorld()
        world2.from_dict(data)
        assert world2.robot.x == world.robot.x
        assert world2.robot.inventory == world.robot.inventory

    def test_reset(self):
        world = load_level("tutorial")
        initial_x = world.robot.x
        world.move_forward()
        world.move_forward()
        assert world.robot.x != initial_x

        world.reset()
        assert world.robot.x == initial_x

    def test_render(self):
        world = load_level("tutorial")
        rendered = world.render()
        assert ">" in rendered or "^" in rendered  # Robot symbol

    def test_box_push(self):
        world = GridWorld()
        world.load_level("""
            ....
            .SB.
            ....
        """)
        # Robot at (1,1) facing east, box at (2,1)
        result = world.move_forward()
        assert result.success
        assert world.robot.x == 2
        # Box should have moved to (3,1)
        assert world.get_tile(3, 1) == Tile.BOX


class TestRobot:
    """Test Robot dataclass."""

    def test_to_dict(self):
        robot = Robot(5, 3, Direction.SOUTH, ["key"], 10)
        data = robot.to_dict()
        assert data["x"] == 5
        assert data["y"] == 3
        assert data["direction"] == "SOUTH"
        assert "key" in data["inventory"]
        assert data["moves"] == 10

    def test_from_dict(self):
        data = {
            "x": 5,
            "y": 3,
            "direction": "SOUTH",
            "inventory": ["key"],
            "moves": 10
        }
        robot = Robot.from_dict(data)
        assert robot.x == 5
        assert robot.y == 3
        assert robot.direction == Direction.SOUTH
        assert "key" in robot.inventory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
