#!/usr/bin/env python3
"""
High-Fidelity Perception System for Home Robot Training

Generates realistic sensory data that approaches human perception:
- RGB camera images (simulated photorealistic views)
- Depth maps (stereo/LiDAR simulation)
- Semantic segmentation
- Object detection with 3D bounding boxes
- Audio/sound events
- Haptic/force feedback
- Proprioception (joint positions, velocities)

This enables training robots to perceive the world with human-like complexity.

Usage:
    from perception_system import PerceptionEngine

    engine = PerceptionEngine(resolution=(640, 480))
    perception = engine.generate_perception(game_state)
"""

import json
import math
import random
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class SensorType(Enum):
    """Types of sensors available to the robot."""
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR = "lidar"
    MICROPHONE = "microphone"
    TOUCH_SENSOR = "touch_sensor"
    IMU = "imu"  # Inertial measurement unit
    JOINT_ENCODER = "joint_encoder"
    TEMPERATURE = "temperature"
    PROXIMITY = "proximity"


class MaterialType(Enum):
    """Material types affecting visual/physical properties."""
    WOOD = "wood"
    METAL = "metal"
    FABRIC = "fabric"
    PLASTIC = "plastic"
    GLASS = "glass"
    CERAMIC = "ceramic"
    CARPET = "carpet"
    TILE = "tile"
    CONCRETE = "concrete"
    LEATHER = "leather"
    PAPER = "paper"
    FOOD = "food"


class LightingCondition(Enum):
    """Ambient lighting conditions."""
    BRIGHT_DAYLIGHT = "bright_daylight"
    OVERCAST = "overcast"
    INDOOR_BRIGHT = "indoor_bright"
    INDOOR_DIM = "indoor_dim"
    NIGHT_WITH_LIGHTS = "night_with_lights"
    DARK = "dark"
    MIXED_LIGHTING = "mixed_lighting"


@dataclass
class MaterialProperties:
    """Physical and visual properties of a material."""
    reflectivity: float = 0.5  # 0-1, how shiny
    roughness: float = 0.5  # 0-1, surface texture
    transparency: float = 0.0  # 0-1, see-through
    color_rgb: Tuple[int, int, int] = (128, 128, 128)
    weight_density: float = 1.0  # kg/m^3 relative
    friction: float = 0.5  # 0-1
    temperature: float = 20.0  # Celsius


# Material database
MATERIAL_DB = {
    MaterialType.WOOD: MaterialProperties(
        reflectivity=0.2, roughness=0.6, color_rgb=(139, 90, 43),
        weight_density=0.7, friction=0.6
    ),
    MaterialType.METAL: MaterialProperties(
        reflectivity=0.8, roughness=0.2, color_rgb=(180, 180, 190),
        weight_density=7.8, friction=0.4
    ),
    MaterialType.FABRIC: MaterialProperties(
        reflectivity=0.1, roughness=0.9, color_rgb=(100, 100, 120),
        weight_density=0.3, friction=0.8
    ),
    MaterialType.PLASTIC: MaterialProperties(
        reflectivity=0.4, roughness=0.3, color_rgb=(200, 200, 200),
        weight_density=1.2, friction=0.5
    ),
    MaterialType.GLASS: MaterialProperties(
        reflectivity=0.9, roughness=0.1, transparency=0.8,
        color_rgb=(240, 250, 255), weight_density=2.5, friction=0.3
    ),
    MaterialType.CERAMIC: MaterialProperties(
        reflectivity=0.5, roughness=0.2, color_rgb=(255, 250, 245),
        weight_density=2.4, friction=0.5
    ),
    MaterialType.CARPET: MaterialProperties(
        reflectivity=0.05, roughness=0.95, color_rgb=(80, 70, 60),
        weight_density=0.5, friction=0.9
    ),
    MaterialType.TILE: MaterialProperties(
        reflectivity=0.6, roughness=0.2, color_rgb=(220, 220, 210),
        weight_density=2.0, friction=0.4
    ),
}


@dataclass
class RGBImage:
    """Simulated RGB camera image."""
    width: int
    height: int
    data: np.ndarray  # HxWx3 uint8
    fov_horizontal: float = 90.0  # degrees
    fov_vertical: float = 60.0
    exposure: float = 1.0
    noise_level: float = 0.02

    def to_bytes(self) -> bytes:
        """Convert to raw bytes for transmission."""
        return self.data.tobytes()

    def add_realistic_noise(self):
        """Add camera sensor noise."""
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * 255, self.data.shape)
            self.data = np.clip(self.data.astype(float) + noise, 0, 255).astype(np.uint8)


@dataclass
class DepthMap:
    """Depth/distance measurements."""
    width: int
    height: int
    data: np.ndarray  # HxW float32 in meters
    min_range: float = 0.1  # meters
    max_range: float = 10.0
    noise_sigma: float = 0.01  # meters

    def to_point_cloud(self, camera_matrix: np.ndarray = None) -> np.ndarray:
        """Convert depth map to 3D point cloud."""
        if camera_matrix is None:
            # Default pinhole camera
            fx = self.width / 2
            fy = self.height / 2
            cx = self.width / 2
            cy = self.height / 2
        else:
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        points = []
        for v in range(self.height):
            for u in range(self.width):
                z = self.data[v, u]
                if z > self.min_range and z < self.max_range:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append([x, y, z])

        return np.array(points)


@dataclass
class SemanticSegmentation:
    """Pixel-wise semantic labels."""
    width: int
    height: int
    labels: np.ndarray  # HxW int32 class IDs
    class_names: Dict[int, str] = field(default_factory=dict)
    confidence: np.ndarray = None  # HxW float32 confidence scores

    # Standard semantic classes
    CLASSES = {
        0: "background",
        1: "floor",
        2: "wall",
        3: "ceiling",
        4: "door",
        5: "window",
        6: "furniture",
        7: "table",
        8: "chair",
        9: "sofa",
        10: "bed",
        11: "appliance",
        12: "light",
        13: "person",
        14: "pet",
        15: "object_graspable",
        16: "object_static",
        17: "obstacle",
        18: "goal",
        19: "robot_arm",
        20: "robot_body",
    }


@dataclass
class ObjectDetection:
    """Detected objects with 3D bounding boxes."""
    class_id: int
    class_name: str
    confidence: float
    # 2D bounding box in image
    bbox_2d: Tuple[int, int, int, int]  # x, y, width, height
    # 3D bounding box in world coordinates
    position_3d: Tuple[float, float, float]  # x, y, z center
    dimensions_3d: Tuple[float, float, float]  # width, height, depth
    orientation: float  # yaw in radians
    # Additional properties
    velocity: Optional[Tuple[float, float, float]] = None
    material: Optional[MaterialType] = None
    graspable: bool = False
    weight_estimate: float = 0.0  # kg


@dataclass
class AudioEvent:
    """Sound/audio event."""
    source_position: Tuple[float, float, float]
    volume_db: float  # decibels
    frequency_hz: float  # dominant frequency
    duration_ms: float
    event_type: str  # "speech", "impact", "ambient", etc.
    confidence: float = 1.0


@dataclass
class ProprioceptionState:
    """Robot's internal body state."""
    joint_positions: List[float]  # radians
    joint_velocities: List[float]  # rad/s
    joint_torques: List[float]  # Nm
    base_position: Tuple[float, float, float]  # x, y, z
    base_orientation: Tuple[float, float, float, float]  # quaternion
    base_velocity: Tuple[float, float, float]  # m/s
    base_angular_velocity: Tuple[float, float, float]  # rad/s
    battery_level: float = 1.0  # 0-1
    temperature: float = 25.0  # Celsius


@dataclass
class HapticFeedback:
    """Touch/force sensor feedback."""
    contact_points: List[Tuple[float, float, float]]  # positions
    forces: List[Tuple[float, float, float]]  # force vectors
    pressure: List[float]  # pressure at each contact
    surface_texture: Optional[float] = None  # roughness estimate


@dataclass
class LidarScan:
    """360-degree LiDAR scan."""
    angles: np.ndarray  # radians, -pi to pi
    ranges: np.ndarray  # meters
    intensities: np.ndarray  # reflectivity 0-1
    min_range: float = 0.1
    max_range: float = 30.0
    angular_resolution: float = 0.5  # degrees

    def to_cartesian(self) -> np.ndarray:
        """Convert to (x, y) points."""
        valid = (self.ranges > self.min_range) & (self.ranges < self.max_range)
        x = self.ranges[valid] * np.cos(self.angles[valid])
        y = self.ranges[valid] * np.sin(self.angles[valid])
        return np.column_stack([x, y])


@dataclass
class PerceptionFrame:
    """Complete perception frame from all sensors."""
    timestamp: float  # seconds

    # Visual
    rgb_image: Optional[RGBImage] = None
    depth_map: Optional[DepthMap] = None
    semantic_seg: Optional[SemanticSegmentation] = None
    object_detections: List[ObjectDetection] = field(default_factory=list)

    # Range sensing
    lidar_scan: Optional[LidarScan] = None

    # Audio
    audio_events: List[AudioEvent] = field(default_factory=list)
    ambient_noise_level: float = 30.0  # dB

    # Body state
    proprioception: Optional[ProprioceptionState] = None
    haptic: Optional[HapticFeedback] = None

    # Environment
    lighting: LightingCondition = LightingCondition.INDOOR_BRIGHT
    temperature: float = 22.0  # Celsius

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp,
            "lighting": self.lighting.value,
            "temperature": self.temperature,
            "ambient_noise": self.ambient_noise_level,
            "has_rgb": self.rgb_image is not None,
            "has_depth": self.depth_map is not None,
            "has_semantic": self.semantic_seg is not None,
            "num_detections": len(self.object_detections),
            "has_lidar": self.lidar_scan is not None,
            "num_audio_events": len(self.audio_events),
        }


class PerceptionEngine:
    """Generates high-fidelity perception data from game states."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        enable_noise: bool = True,
        lighting: LightingCondition = LightingCondition.INDOOR_BRIGHT,
    ):
        self.width, self.height = resolution
        self.enable_noise = enable_noise
        self.lighting = lighting
        self.frame_count = 0

        # Camera intrinsics
        self.focal_length = self.width / (2 * math.tan(math.radians(45)))

        # Object appearance database
        self.object_colors = self._init_object_colors()
        self.object_materials = self._init_object_materials()
        self.object_dimensions = self._init_object_dimensions()

    def _init_object_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """RGB colors for object types."""
        return {
            "cup": (200, 180, 160),
            "glass": (230, 240, 250),
            "bowl": (180, 170, 165),
            "plate": (245, 245, 240),
            "box": (160, 120, 80),
            "chair": (120, 100, 80),
            "table": (140, 110, 70),
            "couch": (100, 90, 85),
            "bed": (200, 190, 185),
            "desk": (130, 100, 70),
            "tv": (30, 30, 35),
            "lamp": (255, 240, 200),
            "fridge": (220, 220, 225),
            "microwave": (200, 200, 205),
            "book": (150, 130, 100),
            "remote": (40, 40, 45),
            "phone": (50, 50, 55),
            "keys": (180, 160, 100),
            "toy": (255, 100, 100),
            "pillow": (180, 175, 170),
            "blanket": (160, 150, 145),
            "trash": (80, 75, 70),
            "door": (150, 120, 90),
            "window": (200, 220, 240),
            "light_switch": (240, 240, 235),
            "floor": (180, 170, 160),
            "wall": (230, 225, 220),
            "obstacle": (100, 100, 110),
            "robot": (80, 120, 180),
            "goal": (100, 200, 100),
        }

    def _init_object_materials(self) -> Dict[str, MaterialType]:
        """Material types for objects."""
        return {
            "cup": MaterialType.CERAMIC,
            "glass": MaterialType.GLASS,
            "bowl": MaterialType.CERAMIC,
            "plate": MaterialType.CERAMIC,
            "box": MaterialType.PLASTIC,
            "chair": MaterialType.WOOD,
            "table": MaterialType.WOOD,
            "couch": MaterialType.FABRIC,
            "bed": MaterialType.FABRIC,
            "desk": MaterialType.WOOD,
            "tv": MaterialType.PLASTIC,
            "lamp": MaterialType.METAL,
            "fridge": MaterialType.METAL,
            "microwave": MaterialType.METAL,
            "book": MaterialType.PAPER,
            "remote": MaterialType.PLASTIC,
            "phone": MaterialType.GLASS,
            "keys": MaterialType.METAL,
            "toy": MaterialType.PLASTIC,
            "pillow": MaterialType.FABRIC,
            "blanket": MaterialType.FABRIC,
            "trash": MaterialType.PAPER,
            "door": MaterialType.WOOD,
            "window": MaterialType.GLASS,
            "light_switch": MaterialType.PLASTIC,
        }

    def _init_object_dimensions(self) -> Dict[str, Tuple[float, float, float]]:
        """Typical dimensions (width, height, depth) in meters."""
        return {
            "cup": (0.08, 0.10, 0.08),
            "glass": (0.07, 0.12, 0.07),
            "bowl": (0.15, 0.08, 0.15),
            "plate": (0.25, 0.02, 0.25),
            "box": (0.30, 0.25, 0.30),
            "chair": (0.50, 0.90, 0.50),
            "table": (1.20, 0.75, 0.80),
            "couch": (2.00, 0.85, 0.90),
            "bed": (2.00, 0.50, 1.50),
            "desk": (1.40, 0.75, 0.70),
            "tv": (1.00, 0.60, 0.08),
            "lamp": (0.20, 0.50, 0.20),
            "fridge": (0.70, 1.80, 0.70),
            "microwave": (0.50, 0.30, 0.40),
            "book": (0.15, 0.22, 0.03),
            "remote": (0.05, 0.15, 0.02),
            "phone": (0.08, 0.15, 0.01),
            "keys": (0.05, 0.08, 0.02),
            "toy": (0.15, 0.15, 0.15),
            "pillow": (0.50, 0.15, 0.40),
            "blanket": (1.50, 0.05, 1.20),
            "trash": (0.10, 0.08, 0.08),
            "door": (0.90, 2.10, 0.05),
            "window": (1.00, 1.20, 0.05),
            "light_switch": (0.08, 0.12, 0.02),
        }

    def generate_perception(self, game_state: Dict) -> PerceptionFrame:
        """Generate complete perception frame from game state."""
        self.frame_count += 1
        timestamp = self.frame_count / 30.0  # Assume 30 FPS

        robot = game_state.get("robot", {})
        rooms = game_state.get("rooms", [])

        # Generate all sensor data
        rgb_image = self._generate_rgb_image(robot, rooms)
        depth_map = self._generate_depth_map(robot, rooms)
        semantic_seg = self._generate_semantic_segmentation(robot, rooms)
        detections = self._generate_object_detections(robot, rooms)
        lidar_scan = self._generate_lidar_scan(robot, rooms)
        audio_events = self._generate_audio_events(robot, rooms)
        proprioception = self._generate_proprioception(robot)
        haptic = self._generate_haptic_feedback(robot, rooms)

        return PerceptionFrame(
            timestamp=timestamp,
            rgb_image=rgb_image,
            depth_map=depth_map,
            semantic_seg=semantic_seg,
            object_detections=detections,
            lidar_scan=lidar_scan,
            audio_events=audio_events,
            proprioception=proprioception,
            haptic=haptic,
            lighting=self.lighting,
        )

    def _generate_rgb_image(self, robot: Dict, rooms: List[Dict]) -> RGBImage:
        """Generate simulated RGB camera image."""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        robot_x = robot.get("x", 0)
        robot_y = robot.get("y", 0)
        robot_dir = robot.get("dir", "east")
        current_room = robot.get("room", "")

        # Direction to angle
        dir_to_angle = {"north": 90, "south": 270, "east": 0, "west": 180}
        robot_angle = dir_to_angle.get(robot_dir, 0)

        # Fill with ambient/wall color based on lighting
        ambient = self._get_ambient_color()
        image[:, :] = ambient

        # Draw floor in lower portion
        floor_color = self._apply_lighting(self.object_colors["floor"])
        floor_start = int(self.height * 0.6)
        image[floor_start:, :] = floor_color

        # Find current room
        room_data = None
        for r in rooms:
            if r.get("name") == current_room:
                room_data = r
                break

        if room_data:
            objects = room_data.get("objects", [])

            # Sort objects by distance for proper occlusion
            def distance_to_robot(obj):
                ox, oy = obj.get("x", 0), obj.get("y", 0)
                return math.sqrt((ox - robot_x)**2 + (oy - robot_y)**2)

            objects_sorted = sorted(objects, key=distance_to_robot, reverse=True)

            # Render each object
            for obj in objects_sorted:
                obj_type = obj.get("type", "obstacle")
                obj_x, obj_y = obj.get("x", 0), obj.get("y", 0)

                # Calculate relative position to robot
                dx = obj_x - robot_x
                dy = obj_y - robot_y

                # Rotate based on robot direction
                angle_rad = math.radians(robot_angle)
                rel_x = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
                rel_y = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

                # Only render objects in front of robot
                if rel_x > 0.1:
                    # Project to image coordinates
                    distance = math.sqrt(rel_x**2 + rel_y**2)

                    # Screen position
                    screen_x = int(self.width/2 + (rel_y / rel_x) * self.focal_length)

                    # Object size based on distance
                    obj_dims = self.object_dimensions.get(obj_type, (0.3, 0.3, 0.3))
                    apparent_size = int(obj_dims[1] * self.focal_length / (distance + 0.1))
                    apparent_width = int(obj_dims[0] * self.focal_length / (distance + 0.1))

                    # Screen Y based on object height
                    screen_y = int(self.height * 0.5 + apparent_size/2)

                    # Get object color
                    base_color = self.object_colors.get(obj_type, (128, 128, 128))
                    lit_color = self._apply_lighting(base_color, distance)

                    # Draw object as rectangle
                    x1 = max(0, screen_x - apparent_width//2)
                    x2 = min(self.width, screen_x + apparent_width//2)
                    y1 = max(0, screen_y - apparent_size)
                    y2 = min(self.height, screen_y)

                    if x2 > x1 and y2 > y1:
                        image[y1:y2, x1:x2] = lit_color

                        # Add shading for 3D effect
                        if apparent_width > 4:
                            shade = np.array(lit_color) * 0.8
                            image[y1:y2, x1:x1+2] = shade.astype(np.uint8)

        # Add noise if enabled
        rgb = RGBImage(
            width=self.width,
            height=self.height,
            data=image,
            noise_level=0.02 if self.enable_noise else 0.0,
        )

        if self.enable_noise:
            rgb.add_realistic_noise()

        return rgb

    def _generate_depth_map(self, robot: Dict, rooms: List[Dict]) -> DepthMap:
        """Generate simulated depth map."""
        depth = np.full((self.height, self.width), 10.0, dtype=np.float32)

        robot_x = robot.get("x", 0)
        robot_y = robot.get("y", 0)
        robot_dir = robot.get("dir", "east")
        current_room = robot.get("room", "")

        dir_to_angle = {"north": 90, "south": 270, "east": 0, "west": 180}
        robot_angle = dir_to_angle.get(robot_dir, 0)

        # Find current room
        room_data = None
        for r in rooms:
            if r.get("name") == current_room:
                room_data = r
                break

        if room_data:
            # Room bounds create walls
            bounds = room_data.get("bounds", [0, 0, 10, 10])
            room_x, room_y, room_w, room_h = bounds

            # Calculate distances to walls in view direction
            objects = room_data.get("objects", [])

            for obj in objects:
                obj_type = obj.get("type", "obstacle")
                obj_x, obj_y = obj.get("x", 0), obj.get("y", 0)

                dx = obj_x - robot_x
                dy = obj_y - robot_y

                angle_rad = math.radians(robot_angle)
                rel_x = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
                rel_y = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

                if rel_x > 0.1:
                    distance = math.sqrt(rel_x**2 + rel_y**2)

                    screen_x = int(self.width/2 + (rel_y / rel_x) * self.focal_length)

                    obj_dims = self.object_dimensions.get(obj_type, (0.3, 0.3, 0.3))
                    apparent_size = int(obj_dims[1] * self.focal_length / (distance + 0.1))
                    apparent_width = int(obj_dims[0] * self.focal_length / (distance + 0.1))

                    screen_y = int(self.height * 0.5 + apparent_size/2)

                    x1 = max(0, screen_x - apparent_width//2)
                    x2 = min(self.width, screen_x + apparent_width//2)
                    y1 = max(0, screen_y - apparent_size)
                    y2 = min(self.height, screen_y)

                    if x2 > x1 and y2 > y1:
                        depth[y1:y2, x1:x2] = distance

        # Add noise
        if self.enable_noise:
            noise = np.random.normal(0, 0.01, depth.shape)
            depth = depth + noise.astype(np.float32)

        return DepthMap(
            width=self.width,
            height=self.height,
            data=depth,
        )

    def _generate_semantic_segmentation(self, robot: Dict, rooms: List[Dict]) -> SemanticSegmentation:
        """Generate semantic segmentation labels."""
        labels = np.zeros((self.height, self.width), dtype=np.int32)

        # Background/wall = 2, floor = 1
        labels[:int(self.height * 0.6), :] = 2  # wall
        labels[int(self.height * 0.6):, :] = 1  # floor

        robot_x = robot.get("x", 0)
        robot_y = robot.get("y", 0)
        robot_dir = robot.get("dir", "east")
        current_room = robot.get("room", "")

        dir_to_angle = {"north": 90, "south": 270, "east": 0, "west": 180}
        robot_angle = dir_to_angle.get(robot_dir, 0)

        # Class mapping
        type_to_class = {
            "door": 4, "window": 5, "table": 7, "chair": 8, "couch": 9,
            "bed": 10, "fridge": 11, "microwave": 11, "lamp": 12,
            "cup": 15, "glass": 15, "bowl": 15, "plate": 15, "book": 15,
            "remote": 15, "phone": 15, "keys": 15, "toy": 15,
            "pillow": 15, "blanket": 15, "trash": 15, "box": 16,
            "tv": 16, "desk": 7, "light_switch": 16,
        }

        room_data = None
        for r in rooms:
            if r.get("name") == current_room:
                room_data = r
                break

        if room_data:
            for obj in room_data.get("objects", []):
                obj_type = obj.get("type", "obstacle")
                obj_x, obj_y = obj.get("x", 0), obj.get("y", 0)

                dx = obj_x - robot_x
                dy = obj_y - robot_y

                angle_rad = math.radians(robot_angle)
                rel_x = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
                rel_y = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

                if rel_x > 0.1:
                    distance = math.sqrt(rel_x**2 + rel_y**2)

                    screen_x = int(self.width/2 + (rel_y / rel_x) * self.focal_length)

                    obj_dims = self.object_dimensions.get(obj_type, (0.3, 0.3, 0.3))
                    apparent_size = int(obj_dims[1] * self.focal_length / (distance + 0.1))
                    apparent_width = int(obj_dims[0] * self.focal_length / (distance + 0.1))

                    screen_y = int(self.height * 0.5 + apparent_size/2)

                    x1 = max(0, screen_x - apparent_width//2)
                    x2 = min(self.width, screen_x + apparent_width//2)
                    y1 = max(0, screen_y - apparent_size)
                    y2 = min(self.height, screen_y)

                    class_id = type_to_class.get(obj_type, 17)
                    if x2 > x1 and y2 > y1:
                        labels[y1:y2, x1:x2] = class_id

        return SemanticSegmentation(
            width=self.width,
            height=self.height,
            labels=labels,
            class_names=SemanticSegmentation.CLASSES,
        )

    def _generate_object_detections(self, robot: Dict, rooms: List[Dict]) -> List[ObjectDetection]:
        """Generate object detection results with 3D boxes."""
        detections = []

        robot_x = robot.get("x", 0)
        robot_y = robot.get("y", 0)
        robot_dir = robot.get("dir", "east")
        current_room = robot.get("room", "")

        dir_to_angle = {"north": 90, "south": 270, "east": 0, "west": 180}
        robot_angle = dir_to_angle.get(robot_dir, 0)

        type_to_class = {
            "cup": 1, "glass": 2, "bowl": 3, "plate": 4, "box": 5,
            "chair": 6, "table": 7, "couch": 8, "bed": 9, "desk": 10,
            "tv": 11, "lamp": 12, "fridge": 13, "microwave": 14,
            "book": 15, "remote": 16, "phone": 17, "keys": 18,
            "toy": 19, "pillow": 20, "blanket": 21, "trash": 22,
            "door": 23, "window": 24, "light_switch": 25,
        }

        graspable_types = {"cup", "glass", "bowl", "plate", "book", "remote",
                          "phone", "keys", "toy", "pillow", "blanket", "trash"}

        room_data = None
        for r in rooms:
            if r.get("name") == current_room:
                room_data = r
                break

        if room_data:
            for obj in room_data.get("objects", []):
                obj_type = obj.get("type", "obstacle")
                obj_x, obj_y = obj.get("x", 0), obj.get("y", 0)

                dx = obj_x - robot_x
                dy = obj_y - robot_y

                angle_rad = math.radians(robot_angle)
                rel_x = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
                rel_y = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

                if rel_x > 0.1:  # In front of robot
                    distance = math.sqrt(rel_x**2 + rel_y**2)

                    # 2D bbox
                    screen_x = int(self.width/2 + (rel_y / rel_x) * self.focal_length)
                    obj_dims = self.object_dimensions.get(obj_type, (0.3, 0.3, 0.3))
                    apparent_size = int(obj_dims[1] * self.focal_length / (distance + 0.1))
                    apparent_width = int(obj_dims[0] * self.focal_length / (distance + 0.1))
                    screen_y = int(self.height * 0.5 + apparent_size/2)

                    bbox_x = max(0, screen_x - apparent_width//2)
                    bbox_y = max(0, screen_y - apparent_size)

                    # Confidence decreases with distance
                    confidence = max(0.3, 1.0 - distance / 10.0)

                    # Add noise to confidence
                    if self.enable_noise:
                        confidence += random.gauss(0, 0.05)
                        confidence = max(0.1, min(1.0, confidence))

                    material = self.object_materials.get(obj_type)
                    weight = obj_dims[0] * obj_dims[1] * obj_dims[2] * 1000  # crude estimate

                    detection = ObjectDetection(
                        class_id=type_to_class.get(obj_type, 0),
                        class_name=obj_type,
                        confidence=confidence,
                        bbox_2d=(bbox_x, bbox_y, apparent_width, apparent_size),
                        position_3d=(float(obj_x), float(obj_y), obj_dims[1]/2),
                        dimensions_3d=obj_dims,
                        orientation=0.0,
                        material=material,
                        graspable=obj_type in graspable_types,
                        weight_estimate=weight,
                    )
                    detections.append(detection)

        return detections

    def _generate_lidar_scan(self, robot: Dict, rooms: List[Dict]) -> LidarScan:
        """Generate 360-degree LiDAR scan."""
        num_rays = 360
        angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)
        ranges = np.full(num_rays, 30.0, dtype=np.float32)  # max range
        intensities = np.zeros(num_rays, dtype=np.float32)

        robot_x = robot.get("x", 0)
        robot_y = robot.get("y", 0)
        current_room = robot.get("room", "")

        room_data = None
        for r in rooms:
            if r.get("name") == current_room:
                room_data = r
                break

        if room_data:
            bounds = room_data.get("bounds", [0, 0, 10, 10])
            room_x, room_y, room_w, room_h = bounds

            # Ray casting to walls and objects
            for i, angle in enumerate(angles):
                ray_dx = math.cos(angle)
                ray_dy = math.sin(angle)

                min_dist = 30.0

                # Check room boundaries
                if abs(ray_dx) > 0.001:
                    # Left wall
                    t = (room_x - robot_x) / ray_dx
                    if t > 0:
                        min_dist = min(min_dist, t)
                    # Right wall
                    t = (room_x + room_w - robot_x) / ray_dx
                    if t > 0:
                        min_dist = min(min_dist, t)

                if abs(ray_dy) > 0.001:
                    # Bottom wall
                    t = (room_y - robot_y) / ray_dy
                    if t > 0:
                        min_dist = min(min_dist, t)
                    # Top wall
                    t = (room_y + room_h - robot_y) / ray_dy
                    if t > 0:
                        min_dist = min(min_dist, t)

                # Check objects (simplified as circles)
                for obj in room_data.get("objects", []):
                    obj_x, obj_y = obj.get("x", 0), obj.get("y", 0)
                    obj_type = obj.get("type", "obstacle")
                    obj_dims = self.object_dimensions.get(obj_type, (0.3, 0.3, 0.3))
                    radius = max(obj_dims[0], obj_dims[2]) / 2

                    # Ray-circle intersection
                    dx = obj_x - robot_x
                    dy = obj_y - robot_y

                    a = ray_dx * ray_dx + ray_dy * ray_dy
                    b = 2 * (ray_dx * (-dx) + ray_dy * (-dy))
                    c = dx*dx + dy*dy - radius*radius

                    discriminant = b*b - 4*a*c
                    if discriminant >= 0:
                        t = (-b - math.sqrt(discriminant)) / (2*a)
                        if t > 0.1:
                            min_dist = min(min_dist, t)
                            # Set intensity based on material
                            material = self.object_materials.get(obj_type)
                            if material:
                                props = MATERIAL_DB.get(material)
                                if props:
                                    intensities[i] = props.reflectivity

                ranges[i] = min_dist

        # Add noise
        if self.enable_noise:
            noise = np.random.normal(0, 0.02, num_rays)
            ranges = ranges + noise.astype(np.float32)
            ranges = np.clip(ranges, 0.1, 30.0)

        return LidarScan(
            angles=angles,
            ranges=ranges,
            intensities=intensities,
        )

    def _generate_audio_events(self, robot: Dict, rooms: List[Dict]) -> List[AudioEvent]:
        """Generate audio events in environment."""
        events = []

        # Random ambient sounds
        if random.random() < 0.1:
            event_types = ["hvac", "clock_tick", "distant_traffic", "birds", "appliance_hum"]
            events.append(AudioEvent(
                source_position=(random.uniform(0, 10), random.uniform(0, 10), 2.0),
                volume_db=random.uniform(20, 40),
                frequency_hz=random.uniform(100, 2000),
                duration_ms=random.uniform(100, 1000),
                event_type=random.choice(event_types),
                confidence=0.8,
            ))

        return events

    def _generate_proprioception(self, robot: Dict) -> ProprioceptionState:
        """Generate robot body state."""
        # Simplified 6-DOF arm
        num_joints = 6

        return ProprioceptionState(
            joint_positions=[random.gauss(0, 0.1) for _ in range(num_joints)],
            joint_velocities=[0.0] * num_joints,
            joint_torques=[random.gauss(0, 0.5) for _ in range(num_joints)],
            base_position=(float(robot.get("x", 0)), float(robot.get("y", 0)), 0.0),
            base_orientation=(0.0, 0.0, 0.0, 1.0),  # quaternion
            base_velocity=(0.0, 0.0, 0.0),
            base_angular_velocity=(0.0, 0.0, 0.0),
            battery_level=random.uniform(0.7, 1.0),
            temperature=random.uniform(30, 45),
        )

    def _generate_haptic_feedback(self, robot: Dict, rooms: List[Dict]) -> HapticFeedback:
        """Generate touch/force feedback."""
        # Check if robot is near any object
        inventory = robot.get("inventory", [])

        contact_points = []
        forces = []
        pressures = []

        if inventory:
            # Holding something
            contact_points.append((0.3, 0.0, 0.1))  # gripper position
            forces.append((0.0, 0.0, -9.8 * 0.5))  # weight force
            pressures.append(0.5)

        return HapticFeedback(
            contact_points=contact_points,
            forces=forces,
            pressure=pressures,
        )

    def _get_ambient_color(self) -> Tuple[int, int, int]:
        """Get ambient color based on lighting."""
        lighting_colors = {
            LightingCondition.BRIGHT_DAYLIGHT: (240, 240, 250),
            LightingCondition.OVERCAST: (200, 205, 210),
            LightingCondition.INDOOR_BRIGHT: (230, 225, 220),
            LightingCondition.INDOOR_DIM: (150, 145, 140),
            LightingCondition.NIGHT_WITH_LIGHTS: (100, 95, 90),
            LightingCondition.DARK: (20, 20, 25),
            LightingCondition.MIXED_LIGHTING: (180, 175, 170),
        }
        return lighting_colors.get(self.lighting, (200, 200, 200))

    def _apply_lighting(self, color: Tuple[int, int, int], distance: float = 0) -> Tuple[int, int, int]:
        """Apply lighting and distance attenuation to color."""
        r, g, b = color

        # Lighting multiplier
        lighting_mult = {
            LightingCondition.BRIGHT_DAYLIGHT: 1.1,
            LightingCondition.OVERCAST: 0.9,
            LightingCondition.INDOOR_BRIGHT: 1.0,
            LightingCondition.INDOOR_DIM: 0.6,
            LightingCondition.NIGHT_WITH_LIGHTS: 0.4,
            LightingCondition.DARK: 0.1,
            LightingCondition.MIXED_LIGHTING: 0.8,
        }
        mult = lighting_mult.get(self.lighting, 1.0)

        # Distance attenuation
        if distance > 0:
            mult *= max(0.3, 1.0 - distance / 15.0)

        r = int(min(255, r * mult))
        g = int(min(255, g * mult))
        b = int(min(255, b * mult))

        return (r, g, b)


def demo():
    """Demo the perception system."""
    print("=" * 60)
    print("High-Fidelity Perception System Demo")
    print("=" * 60)

    # Sample game state
    game_state = {
        "robot": {
            "x": 3,
            "y": 3,
            "dir": "east",
            "room": "living_room",
            "inventory": [],
        },
        "rooms": [
            {
                "name": "living_room",
                "type": "living_room",
                "bounds": [0, 0, 8, 6],
                "objects": [
                    {"type": "couch", "x": 6, "y": 3},
                    {"type": "tv", "x": 7, "y": 1},
                    {"type": "lamp", "x": 5, "y": 5},
                    {"type": "remote", "x": 5, "y": 3},
                    {"type": "pillow", "x": 6, "y": 4},
                ],
            }
        ],
    }

    # Create perception engine
    engine = PerceptionEngine(resolution=(320, 240))

    # Generate perception frame
    frame = engine.generate_perception(game_state)

    print(f"\nPerception Frame Generated:")
    print(f"  Timestamp: {frame.timestamp:.3f}s")
    print(f"  Lighting: {frame.lighting.value}")

    print(f"\nRGB Image:")
    print(f"  Resolution: {frame.rgb_image.width}x{frame.rgb_image.height}")
    print(f"  Data shape: {frame.rgb_image.data.shape}")

    print(f"\nDepth Map:")
    print(f"  Min depth: {frame.depth_map.data.min():.2f}m")
    print(f"  Max depth: {frame.depth_map.data.max():.2f}m")

    print(f"\nSemantic Segmentation:")
    unique_classes = np.unique(frame.semantic_seg.labels)
    print(f"  Classes present: {[frame.semantic_seg.CLASSES.get(c, c) for c in unique_classes]}")

    print(f"\nObject Detections: {len(frame.object_detections)}")
    for det in frame.object_detections[:5]:
        print(f"  - {det.class_name}: conf={det.confidence:.2f}, "
              f"pos=({det.position_3d[0]:.1f}, {det.position_3d[1]:.1f}), "
              f"graspable={det.graspable}")

    print(f"\nLiDAR Scan:")
    print(f"  Rays: {len(frame.lidar_scan.angles)}")
    print(f"  Min range: {frame.lidar_scan.ranges.min():.2f}m")
    print(f"  Max range: {frame.lidar_scan.ranges.max():.2f}m")

    print(f"\nProprioception:")
    print(f"  Joint positions: {frame.proprioception.joint_positions}")
    print(f"  Battery: {frame.proprioception.battery_level:.0%}")

    print(f"\nAudio Events: {len(frame.audio_events)}")
    for audio in frame.audio_events:
        print(f"  - {audio.event_type}: {audio.volume_db:.0f}dB")


if __name__ == "__main__":
    demo()
