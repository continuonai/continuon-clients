#!/usr/bin/env python3
"""
Spatial Memory System

Provides spatial reasoning and memory capabilities:
1. Occupancy Grid - Mental map of environment
2. Object Memory - Remember object locations
3. Landmark Memory - Key navigation points
4. Path Memory - Remember successful routes
5. Semantic Map - Room/area labels

This enables:
- Navigation with memory (avoid re-exploring)
- Object search with memory ("where did I last see X?")
- Multi-step planning with spatial reasoning

Usage:
    from spatial_memory import SpatialMemory

    memory = SpatialMemory()
    memory.update_from_perception(perception_data)
    path = memory.plan_path(start, goal)
    objects = memory.find_object("cup")
"""

import json
import math
import heapq
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np


@dataclass
class ObjectMemory:
    """Memory of an observed object."""
    object_type: str
    position: Tuple[float, float, float]  # x, y, z
    confidence: float  # 0-1
    last_seen: str  # timestamp
    times_seen: int = 1
    room: str = ""
    graspable: bool = False
    properties: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "object_type": self.object_type,
            "position": self.position,
            "confidence": self.confidence,
            "last_seen": self.last_seen,
            "times_seen": self.times_seen,
            "room": self.room,
            "graspable": self.graspable,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'ObjectMemory':
        return cls(
            object_type=d["object_type"],
            position=tuple(d["position"]),
            confidence=d["confidence"],
            last_seen=d["last_seen"],
            times_seen=d.get("times_seen", 1),
            room=d.get("room", ""),
            graspable=d.get("graspable", False),
            properties=d.get("properties", {}),
        )


@dataclass
class Landmark:
    """A memorable landmark for navigation."""
    name: str
    position: Tuple[float, float]
    landmark_type: str  # "door", "corner", "object", "room_center"
    description: str = ""
    connected_to: List[str] = field(default_factory=list)  # Other landmark names


@dataclass
class PathMemory:
    """Memory of a successful path."""
    start: Tuple[float, float]
    goal: Tuple[float, float]
    waypoints: List[Tuple[float, float]]
    success: bool
    timestamp: str
    total_distance: float
    obstacles_avoided: int = 0


class OccupancyGrid:
    """
    2D occupancy grid representing environment.

    Cell values:
    - -1: Unknown
    -  0: Free
    -  1: Occupied (obstacle)
    -  2: Visited
    """

    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1
    VISITED = 2

    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        resolution: float = 0.1,  # meters per cell
        origin: Tuple[float, float] = (0, 0),
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin

        # Initialize as unknown
        self.grid = np.full((height, width), self.UNKNOWN, dtype=np.int8)

        # Visit counts for exploration
        self.visit_counts = np.zeros((height, width), dtype=np.int32)

        # Confidence in each cell (0-1)
        self.confidence = np.zeros((height, width), dtype=np.float32)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        x = gx * self.resolution + self.origin[0]
        y = gy * self.resolution + self.origin[1]
        return x, y

    def is_valid(self, gx: int, gy: int) -> bool:
        """Check if grid coordinates are valid."""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def get_cell(self, gx: int, gy: int) -> int:
        """Get cell value."""
        if self.is_valid(gx, gy):
            return self.grid[gy, gx]
        return self.OCCUPIED  # Out of bounds = occupied

    def set_cell(self, gx: int, gy: int, value: int, confidence: float = 1.0):
        """Set cell value."""
        if self.is_valid(gx, gy):
            self.grid[gy, gx] = value
            self.confidence[gy, gx] = confidence

    def mark_free(self, x: float, y: float, confidence: float = 0.9):
        """Mark world position as free."""
        gx, gy = self.world_to_grid(x, y)
        self.set_cell(gx, gy, self.FREE, confidence)

    def mark_occupied(self, x: float, y: float, confidence: float = 0.9):
        """Mark world position as occupied."""
        gx, gy = self.world_to_grid(x, y)
        self.set_cell(gx, gy, self.OCCUPIED, confidence)

    def mark_visited(self, x: float, y: float):
        """Mark world position as visited."""
        gx, gy = self.world_to_grid(x, y)
        if self.is_valid(gx, gy):
            self.grid[gy, gx] = self.VISITED
            self.visit_counts[gy, gx] += 1

    def is_free(self, x: float, y: float) -> bool:
        """Check if world position is free."""
        gx, gy = self.world_to_grid(x, y)
        cell = self.get_cell(gx, gy)
        return cell == self.FREE or cell == self.VISITED

    def update_from_lidar(
        self,
        robot_x: float,
        robot_y: float,
        robot_angle: float,
        ranges: List[float],
        angles: List[float],
        max_range: float = 10.0,
    ):
        """Update grid from LiDAR scan."""
        for angle, dist in zip(angles, ranges):
            # Ray angle in world frame
            world_angle = robot_angle + angle

            if dist < max_range:
                # Mark endpoint as occupied
                end_x = robot_x + dist * math.cos(world_angle)
                end_y = robot_y + dist * math.sin(world_angle)
                self.mark_occupied(end_x, end_y)

                # Mark ray as free (ray casting)
                self._trace_ray(robot_x, robot_y, end_x, end_y)
            else:
                # Mark ray as free up to max range
                end_x = robot_x + max_range * math.cos(world_angle)
                end_y = robot_y + max_range * math.sin(world_angle)
                self._trace_ray(robot_x, robot_y, end_x, end_y)

    def _trace_ray(self, x0: float, y0: float, x1: float, y1: float):
        """Trace ray and mark cells as free (Bresenham's algorithm)."""
        gx0, gy0 = self.world_to_grid(x0, y0)
        gx1, gy1 = self.world_to_grid(x1, y1)

        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy

        gx, gy = gx0, gy0

        while True:
            if self.is_valid(gx, gy) and self.grid[gy, gx] != self.OCCUPIED:
                self.grid[gy, gx] = self.FREE

            if gx == gx1 and gy == gy1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                gx += sx
            if e2 < dx:
                err += dx
                gy += sy

    def get_frontier_cells(self) -> List[Tuple[int, int]]:
        """Get frontier cells (free cells adjacent to unknown)."""
        frontiers = []

        for gy in range(1, self.height - 1):
            for gx in range(1, self.width - 1):
                if self.grid[gy, gx] == self.FREE:
                    # Check if adjacent to unknown
                    neighbors = [
                        (gx-1, gy), (gx+1, gy),
                        (gx, gy-1), (gx, gy+1),
                    ]
                    for nx, ny in neighbors:
                        if self.is_valid(nx, ny) and self.grid[ny, nx] == self.UNKNOWN:
                            frontiers.append((gx, gy))
                            break

        return frontiers

    def get_exploration_score(self, x: float, y: float) -> float:
        """Get exploration score (higher = more unexplored nearby)."""
        gx, gy = self.world_to_grid(x, y)

        if not self.is_valid(gx, gy):
            return 0.0

        # Count unknown cells in neighborhood
        unknown_count = 0
        radius = 5

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = gx + dx, gy + dy
                if self.is_valid(nx, ny) and self.grid[ny, nx] == self.UNKNOWN:
                    unknown_count += 1

        # Penalize heavily visited areas
        visit_penalty = min(self.visit_counts[gy, gx] * 0.1, 0.5)

        return unknown_count / ((2 * radius + 1) ** 2) - visit_penalty

    def to_dict(self) -> Dict:
        """Serialize grid."""
        return {
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin": self.origin,
            "grid": self.grid.tolist(),
            "visit_counts": self.visit_counts.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'OccupancyGrid':
        """Deserialize grid."""
        grid = cls(
            width=d["width"],
            height=d["height"],
            resolution=d["resolution"],
            origin=tuple(d["origin"]),
        )
        grid.grid = np.array(d["grid"], dtype=np.int8)
        grid.visit_counts = np.array(d.get("visit_counts", np.zeros_like(grid.grid)), dtype=np.int32)
        return grid


class PathPlanner:
    """A* path planner on occupancy grid."""

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using A*.

        Returns list of waypoints in world coordinates, or None if no path.
        """
        start_grid = self.grid.world_to_grid(start[0], start[1])
        goal_grid = self.grid.world_to_grid(goal[0], goal[1])

        # Check if start/goal are valid
        if not self.grid.is_valid(*start_grid) or not self.grid.is_valid(*goal_grid):
            return None

        if self.grid.get_cell(*goal_grid) == OccupancyGrid.OCCUPIED:
            return None

        # A* search
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_grid:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current)
                # Convert to world coordinates
                return [self.grid.grid_to_world(gx, gy) for gx, gy in path]

            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + self._edge_cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_grid)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        return None  # No path found

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors (8-connected)."""
        gx, gy = pos
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = gx + dx, gy + dy

                if self.grid.is_valid(nx, ny):
                    cell = self.grid.get_cell(nx, ny)
                    if cell != OccupancyGrid.OCCUPIED:
                        neighbors.append((nx, ny))

        return neighbors

    def _edge_cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Cost of moving from a to b."""
        # Diagonal moves cost more
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])

        if dx + dy == 2:
            return 1.414  # Diagonal
        return 1.0

    def _reconstruct_path(
        self,
        came_from: Dict,
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from A* search."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class SpatialMemory:
    """
    Complete spatial memory system.

    Combines:
    - Occupancy grid for navigation
    - Object memory for finding things
    - Landmark memory for high-level navigation
    - Path memory for route planning
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (100, 100),
        resolution: float = 0.1,
        data_dir: str = "brain_b_data/spatial_memory",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Occupancy grid
        self.grid = OccupancyGrid(
            width=grid_size[0],
            height=grid_size[1],
            resolution=resolution,
        )

        # Path planner
        self.planner = PathPlanner(self.grid)

        # Object memory
        self.objects: Dict[str, List[ObjectMemory]] = {}

        # Landmark memory
        self.landmarks: Dict[str, Landmark] = {}

        # Path memory
        self.paths: List[PathMemory] = []

        # Current room estimate
        self.current_room: str = ""

        # Robot position history
        self.position_history: List[Tuple[float, float, str]] = []  # (x, y, timestamp)

    def update_robot_position(self, x: float, y: float):
        """Update robot position and mark as visited."""
        self.grid.mark_visited(x, y)
        self.position_history.append((x, y, datetime.now().isoformat()))

        # Keep history limited
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-500:]

    def update_from_perception(
        self,
        robot_x: float,
        robot_y: float,
        robot_angle: float,
        lidar_ranges: List[float] = None,
        lidar_angles: List[float] = None,
        detected_objects: List[Dict] = None,
    ):
        """Update memory from perception data."""
        # Update position
        self.update_robot_position(robot_x, robot_y)

        # Update occupancy grid from LiDAR
        if lidar_ranges and lidar_angles:
            self.grid.update_from_lidar(
                robot_x, robot_y, robot_angle,
                lidar_ranges, lidar_angles,
            )

        # Update object memory
        if detected_objects:
            for obj in detected_objects:
                self.remember_object(
                    obj_type=obj.get("class_name", "unknown"),
                    position=obj.get("position_3d", (robot_x, robot_y, 0)),
                    confidence=obj.get("confidence", 0.5),
                    graspable=obj.get("graspable", False),
                )

    def remember_object(
        self,
        obj_type: str,
        position: Tuple[float, float, float],
        confidence: float = 0.8,
        graspable: bool = False,
        room: str = "",
    ):
        """Remember seeing an object at a location."""
        now = datetime.now().isoformat()

        if obj_type not in self.objects:
            self.objects[obj_type] = []

        # Check if we've seen this object nearby before
        for existing in self.objects[obj_type]:
            dist = math.sqrt(
                (existing.position[0] - position[0])**2 +
                (existing.position[1] - position[1])**2
            )

            if dist < 0.5:  # Same object (within 0.5m)
                # Update existing memory
                existing.last_seen = now
                existing.times_seen += 1
                existing.confidence = max(existing.confidence, confidence)
                # Weighted average position
                alpha = 0.3
                existing.position = (
                    existing.position[0] * (1-alpha) + position[0] * alpha,
                    existing.position[1] * (1-alpha) + position[1] * alpha,
                    existing.position[2] * (1-alpha) + position[2] * alpha,
                )
                return

        # New object
        memory = ObjectMemory(
            object_type=obj_type,
            position=position,
            confidence=confidence,
            last_seen=now,
            room=room or self.current_room,
            graspable=graspable,
        )
        self.objects[obj_type].append(memory)

    def find_object(self, obj_type: str) -> Optional[ObjectMemory]:
        """
        Find remembered location of object type.

        Returns most confident/recent memory, or None.
        """
        if obj_type not in self.objects:
            return None

        memories = self.objects[obj_type]
        if not memories:
            return None

        # Sort by confidence * recency
        def score(m: ObjectMemory) -> float:
            return m.confidence * m.times_seen

        return max(memories, key=score)

    def find_objects_in_room(self, room: str) -> List[ObjectMemory]:
        """Find all objects remembered in a room."""
        result = []
        for obj_list in self.objects.values():
            for obj in obj_list:
                if obj.room == room:
                    result.append(obj)
        return result

    def add_landmark(
        self,
        name: str,
        position: Tuple[float, float],
        landmark_type: str,
        description: str = "",
    ):
        """Add a landmark for navigation."""
        self.landmarks[name] = Landmark(
            name=name,
            position=position,
            landmark_type=landmark_type,
            description=description,
        )

    def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """Plan path from start to goal."""
        path = self.planner.plan(start, goal)

        if path:
            # Remember successful path
            self.paths.append(PathMemory(
                start=start,
                goal=goal,
                waypoints=path,
                success=True,
                timestamp=datetime.now().isoformat(),
                total_distance=self._path_length(path),
            ))

        return path

    def _path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total path length."""
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            total += math.sqrt(dx*dx + dy*dy)
        return total

    def get_exploration_target(self, robot_x: float, robot_y: float) -> Optional[Tuple[float, float]]:
        """Get best position to explore next."""
        frontiers = self.grid.get_frontier_cells()

        if not frontiers:
            return None

        # Score frontiers by exploration value and distance
        def frontier_score(f: Tuple[int, int]) -> float:
            fx, fy = self.grid.grid_to_world(f[0], f[1])
            dist = math.sqrt((fx - robot_x)**2 + (fy - robot_y)**2)

            exploration = self.grid.get_exploration_score(fx, fy)
            distance_penalty = dist * 0.1

            return exploration - distance_penalty

        best_frontier = max(frontiers, key=frontier_score)
        return self.grid.grid_to_world(best_frontier[0], best_frontier[1])

    def get_memory_summary(self) -> Dict:
        """Get summary of spatial memory."""
        total_objects = sum(len(objs) for objs in self.objects.values())

        grid_stats = {
            "free_cells": int(np.sum(self.grid.grid == OccupancyGrid.FREE)),
            "occupied_cells": int(np.sum(self.grid.grid == OccupancyGrid.OCCUPIED)),
            "unknown_cells": int(np.sum(self.grid.grid == OccupancyGrid.UNKNOWN)),
            "visited_cells": int(np.sum(self.grid.grid == OccupancyGrid.VISITED)),
        }

        return {
            "objects_remembered": total_objects,
            "object_types": len(self.objects),
            "landmarks": len(self.landmarks),
            "paths_recorded": len(self.paths),
            "position_history_size": len(self.position_history),
            "grid_stats": grid_stats,
        }

    def save(self):
        """Save spatial memory to disk."""
        data = {
            "grid": self.grid.to_dict(),
            "objects": {
                k: [o.to_dict() for o in v]
                for k, v in self.objects.items()
            },
            "landmarks": {
                k: {
                    "name": v.name,
                    "position": v.position,
                    "landmark_type": v.landmark_type,
                    "description": v.description,
                    "connected_to": v.connected_to,
                }
                for k, v in self.landmarks.items()
            },
            "current_room": self.current_room,
        }

        with open(self.data_dir / "spatial_memory.json", 'w') as f:
            json.dump(data, f)

        print(f"Saved spatial memory: {self.get_memory_summary()}")

    def load(self):
        """Load spatial memory from disk."""
        filepath = self.data_dir / "spatial_memory.json"
        if not filepath.exists():
            return

        with open(filepath) as f:
            data = json.load(f)

        self.grid = OccupancyGrid.from_dict(data["grid"])
        self.planner = PathPlanner(self.grid)

        self.objects = {
            k: [ObjectMemory.from_dict(o) for o in v]
            for k, v in data.get("objects", {}).items()
        }

        for k, v in data.get("landmarks", {}).items():
            self.landmarks[k] = Landmark(
                name=v["name"],
                position=tuple(v["position"]),
                landmark_type=v["landmark_type"],
                description=v.get("description", ""),
                connected_to=v.get("connected_to", []),
            )

        self.current_room = data.get("current_room", "")

        print(f"Loaded spatial memory: {self.get_memory_summary()}")


def demo():
    """Demo the spatial memory system."""
    print("=" * 60)
    print("Spatial Memory System Demo")
    print("=" * 60)

    memory = SpatialMemory(grid_size=(50, 50), resolution=0.2)

    # Simulate robot moving and sensing
    print("\nSimulating robot exploration...")

    # Robot path
    positions = [
        (5.0, 5.0, 0.0),
        (6.0, 5.0, 0.0),
        (7.0, 5.0, math.pi/4),
        (7.5, 5.5, math.pi/2),
        (7.5, 6.5, math.pi/2),
    ]

    # Simulate LiDAR
    lidar_angles = [math.radians(a) for a in range(-180, 180, 5)]

    for x, y, angle in positions:
        # Fake LiDAR ranges (walls at edges)
        ranges = []
        for la in lidar_angles:
            world_angle = angle + la
            # Distance to boundary (simplified)
            dist = min(
                abs(10 - x) / max(0.01, abs(math.cos(world_angle))),
                abs(10 - y) / max(0.01, abs(math.sin(world_angle))),
                abs(x) / max(0.01, abs(math.cos(world_angle + math.pi))),
                abs(y) / max(0.01, abs(math.sin(world_angle + math.pi))),
            )
            ranges.append(min(dist, 10.0) + np.random.normal(0, 0.05))

        memory.update_from_perception(
            robot_x=x,
            robot_y=y,
            robot_angle=angle,
            lidar_ranges=ranges,
            lidar_angles=lidar_angles,
        )

    # Remember some objects
    print("\nRemembering objects...")
    memory.remember_object("cup", (6.0, 5.5, 0.1), confidence=0.9, graspable=True, room="kitchen")
    memory.remember_object("chair", (5.0, 6.0, 0.4), confidence=0.8, room="kitchen")
    memory.remember_object("cup", (6.1, 5.6, 0.1), confidence=0.85, graspable=True)  # Same cup

    # Find objects
    print("\nFinding objects...")
    cup = memory.find_object("cup")
    if cup:
        print(f"  Cup found at {cup.position}, seen {cup.times_seen} times")

    # Add landmarks
    memory.add_landmark("kitchen_door", (5.0, 7.0), "door", "Door to kitchen")
    memory.add_landmark("living_room_center", (3.0, 3.0), "room_center")

    # Plan path
    print("\nPlanning path...")
    path = memory.plan_path((5.0, 5.0), (7.0, 7.0))
    if path:
        print(f"  Path found with {len(path)} waypoints")
        print(f"  Waypoints: {path[:3]}...")

    # Get exploration target
    target = memory.get_exploration_target(7.5, 6.5)
    if target:
        print(f"\nNext exploration target: {target}")

    # Summary
    print(f"\nMemory Summary: {memory.get_memory_summary()}")

    # Save
    memory.save()


if __name__ == "__main__":
    demo()
