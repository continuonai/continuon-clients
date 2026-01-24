"""
3D Scene Management for House Renderer

Manages the 3D scene graph, object transforms, and integrates with
existing room scanner pipeline and home_world.py.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import json
import math
import numpy as np
from pathlib import Path

from .assets import (
    Room, Wall, Floor, Ceiling, Furniture, LightFixture,
    Vector3, BoundingBox, RoomType, FurnitureType, HouseAssets, FurnitureCatalog,
    HOUSE_TEMPLATES,
)
from .materials import PBRMaterial, MATERIAL_LIBRARY, get_material


@dataclass
class Transform:
    """3D transformation (position, rotation, scale)."""
    position: Tuple[float, float, float] = (0, 0, 0)
    rotation: Tuple[float, float, float] = (0, 0, 0)  # Euler angles (degrees)
    scale: Tuple[float, float, float] = (1, 1, 1)

    def to_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        # Translation
        T = np.eye(4)
        T[:3, 3] = self.position

        # Rotation (XYZ Euler)
        rx, ry, rz = [math.radians(a) for a in self.rotation]

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])

        Ry = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])

        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])

        R = np.eye(4)
        R[:3, :3] = Rz @ Ry @ Rx

        # Scale
        S = np.diag([*self.scale, 1])

        return T @ R @ S

    def to_dict(self) -> dict:
        return {
            'position': list(self.position),
            'rotation': list(self.rotation),
            'scale': list(self.scale),
        }


@dataclass
class SceneObject:
    """
    Object in the 3D scene with transform and rendering properties.
    """
    name: str
    transform: Transform
    geometry_type: str = 'box'
    geometry_params: Dict = field(default_factory=dict)
    material: str = 'white_paint'
    secondary_material: Optional[str] = None
    visible: bool = True
    cast_shadow: bool = True
    receive_shadow: bool = True

    # Physics/collision
    is_obstacle: bool = True
    collision_shape: str = 'box'

    # Metadata
    tags: List[str] = field(default_factory=list)

    def get_bounding_box(self) -> BoundingBox:
        """Get axis-aligned bounding box in world space."""
        size = self.geometry_params.get('size', (1, 1, 1))
        # Ensure size is a tuple of numbers
        if isinstance(size, str):
            size = (1, 1, 1)
        elif not isinstance(size, (list, tuple)):
            size = (1, 1, 1)
        else:
            size = tuple(float(s) if isinstance(s, (int, float)) else 1.0 for s in size)
            if len(size) < 3:
                size = (size[0] if len(size) > 0 else 1, size[1] if len(size) > 1 else 1, 1)

        pos = self.transform.position
        scale = self.transform.scale

        half_size = (
            size[0] * scale[0] / 2,
            size[1] * scale[1] / 2,
            size[2] * scale[2] / 2,
        )

        return BoundingBox(
            min_point=Vector3(pos[0] - half_size[0], pos[1], pos[2] - half_size[2]),
            max_point=Vector3(pos[0] + half_size[0], pos[1] + size[1] * scale[1], pos[2] + half_size[2])
        )

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'transform': self.transform.to_dict(),
            'geometryType': self.geometry_type,
            'geometryParams': self.geometry_params,
            'material': self.material,
            'secondaryMaterial': self.secondary_material,
            'visible': self.visible,
            'castShadow': self.cast_shadow,
            'receiveShadow': self.receive_shadow,
            'isObstacle': self.is_obstacle,
            'tags': self.tags,
        }


@dataclass
class SceneLight:
    """Light source in the scene."""
    name: str
    light_type: str = 'point'  # point, directional, spot, ambient
    position: Tuple[float, float, float] = (0, 2, 0)
    direction: Optional[Tuple[float, float, float]] = None
    color: Tuple[int, int, int] = (255, 250, 240)
    intensity: float = 1.0
    range: float = 10.0
    angle: float = 60.0  # For spot lights
    cast_shadow: bool = True

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'type': self.light_type,
            'position': list(self.position),
            'direction': list(self.direction) if self.direction else None,
            'color': f'#{self.color[0]:02x}{self.color[1]:02x}{self.color[2]:02x}',
            'intensity': self.intensity,
            'range': self.range,
            'angle': self.angle,
            'castShadow': self.cast_shadow,
        }


class HouseScene:
    """
    Complete 3D house scene for rendering.

    Can be created from:
    - Pre-built templates (studio, 2-bedroom)
    - Room scanner results
    - home_world.py HomeWorld
    - JSON scene files
    """

    def __init__(self, name: str = 'Untitled'):
        self.name = name
        self.objects: List[SceneObject] = []
        self.lights: List[SceneLight] = []
        self.rooms: List[Room] = []

        # Scene bounds (auto-calculated)
        self._bounds_min = [float('inf')] * 3
        self._bounds_max = [float('-inf')] * 3

        # Ambient settings
        self.ambient_color = (40, 40, 50)
        self.ambient_intensity = 0.3
        self.background_color = (20, 25, 35)

        # Fog (optional)
        self.fog_enabled = False
        self.fog_color = (40, 45, 55)
        self.fog_near = 5.0
        self.fog_far = 20.0

    def add_object(self, obj: SceneObject):
        """Add an object to the scene."""
        self.objects.append(obj)
        self._update_bounds(obj)

    def add_light(self, light: SceneLight):
        """Add a light to the scene."""
        self.lights.append(light)

    def add_room(self, room: Room):
        """Add a room with its geometry to the scene."""
        self.rooms.append(room)
        self._build_room_geometry(room)

    def _update_bounds(self, obj: SceneObject):
        """Update scene bounds to include object."""
        bbox = obj.get_bounding_box()
        self._bounds_min[0] = min(self._bounds_min[0], bbox.min_point.x)
        self._bounds_min[1] = min(self._bounds_min[1], bbox.min_point.y)
        self._bounds_min[2] = min(self._bounds_min[2], bbox.min_point.z)
        self._bounds_max[0] = max(self._bounds_max[0], bbox.max_point.x)
        self._bounds_max[1] = max(self._bounds_max[1], bbox.max_point.y)
        self._bounds_max[2] = max(self._bounds_max[2], bbox.max_point.z)

    def get_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get scene bounding box."""
        if self._bounds_min[0] == float('inf'):
            return ((0, 0, 0), (10, 3, 10))  # Default bounds
        return (tuple(self._bounds_min), tuple(self._bounds_max))

    def get_floor_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get 2D floor bounds (x, z)."""
        bounds = self.get_bounds()
        return (
            (bounds[0][0], bounds[0][2]),
            (bounds[1][0], bounds[1][2])
        )

    def _build_room_geometry(self, room: Room):
        """Convert room to scene objects."""
        # Floor
        if room.floor:
            self._build_floor(room.floor, room.name)

        # Ceiling
        if room.ceiling:
            self._build_ceiling(room.ceiling, room.name)

        # Walls
        for i, wall in enumerate(room.walls):
            self._build_wall(wall, f"{room.name}_wall_{i}")

        # Furniture
        for furn in room.furniture:
            self._build_furniture(furn)

        # Lights
        for light in room.lights:
            self.add_light(SceneLight(
                name=light.name,
                light_type=light.light_type,
                position=light.position.to_tuple(),
                color=light.color,
                intensity=light.intensity,
                range=light.range,
            ))

    def _build_floor(self, floor: Floor, room_name: str):
        """Build floor geometry from Floor object."""
        if len(floor.vertices) < 3:
            return

        # Calculate bounds
        xs = [v[0] for v in floor.vertices]
        zs = [v[1] for v in floor.vertices]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

        width = max_x - min_x
        depth = max_z - min_z
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2

        obj = SceneObject(
            name=f'{room_name}_floor',
            transform=Transform(
                position=(center_x, floor.y_position, center_z),
            ),
            geometry_type='plane',
            geometry_params={
                'width': width,
                'depth': depth,
                'vertices': floor.vertices,
            },
            material=floor.material,
            is_obstacle=False,
            receive_shadow=True,
            cast_shadow=False,
            tags=['floor'],
        )
        self.add_object(obj)

    def _build_ceiling(self, ceiling: Ceiling, room_name: str):
        """Build ceiling geometry."""
        if len(ceiling.vertices) < 3:
            return

        xs = [v[0] for v in ceiling.vertices]
        zs = [v[1] for v in ceiling.vertices]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

        width = max_x - min_x
        depth = max_z - min_z
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2

        obj = SceneObject(
            name=f'{room_name}_ceiling',
            transform=Transform(
                position=(center_x, ceiling.height, center_z),
                rotation=(180, 0, 0),  # Flip to face down
            ),
            geometry_type='plane',
            geometry_params={
                'width': width,
                'depth': depth,
            },
            material=ceiling.material,
            is_obstacle=False,
            receive_shadow=False,
            cast_shadow=False,
            tags=['ceiling'],
        )
        self.add_object(obj)

    def _build_wall(self, wall: Wall, wall_name: str):
        """Build wall geometry with openings."""
        # Calculate wall dimensions
        dx = wall.end[0] - wall.start[0]
        dz = wall.end[1] - wall.start[1]
        length = math.sqrt(dx * dx + dz * dz)

        if length < 0.01:
            return

        # Wall center position
        center_x = (wall.start[0] + wall.end[0]) / 2
        center_z = (wall.start[1] + wall.end[1]) / 2

        # Wall rotation (around Y axis)
        angle = math.degrees(math.atan2(dx, dz))

        # Without openings: simple box
        if not wall.openings:
            obj = SceneObject(
                name=wall_name,
                transform=Transform(
                    position=(center_x, wall.height / 2, center_z),
                    rotation=(0, angle, 0),
                ),
                geometry_type='box',
                geometry_params={
                    'size': (length, wall.height, wall.thickness),
                },
                material=wall.material,
                is_obstacle=True,
                tags=['wall'],
            )
            self.add_object(obj)
        else:
            # With openings: build wall segments
            self._build_wall_with_openings(wall, wall_name, length, angle, center_x, center_z)

        # Baseboard
        if wall.has_baseboard:
            obj = SceneObject(
                name=f'{wall_name}_baseboard',
                transform=Transform(
                    position=(center_x, wall.baseboard_height / 2, center_z),
                    rotation=(0, angle, 0),
                ),
                geometry_type='box',
                geometry_params={
                    'size': (length, wall.baseboard_height, wall.thickness + 0.02),
                },
                material=wall.baseboard_material,
                is_obstacle=False,
                tags=['baseboard'],
            )
            self.add_object(obj)

    def _build_wall_with_openings(self, wall: Wall, name: str, length: float, angle: float, cx: float, cz: float):
        """Build wall segments around door/window openings."""
        # Sort openings by position
        openings = sorted(wall.openings, key=lambda o: o['position'])

        segments = []
        current_pos = 0

        for opening in openings:
            open_pos = opening['position'] * length
            open_width = opening['width']
            open_height = opening['height']
            open_bottom = opening.get('bottom', 0)

            # Segment before opening
            if open_pos - open_width / 2 > current_pos:
                segments.append({
                    'start': current_pos,
                    'end': open_pos - open_width / 2,
                    'bottom': 0,
                    'top': wall.height,
                })

            # Above opening
            if open_bottom + open_height < wall.height:
                segments.append({
                    'start': open_pos - open_width / 2,
                    'end': open_pos + open_width / 2,
                    'bottom': open_bottom + open_height,
                    'top': wall.height,
                })

            # Below opening (for windows)
            if open_bottom > 0:
                segments.append({
                    'start': open_pos - open_width / 2,
                    'end': open_pos + open_width / 2,
                    'bottom': 0,
                    'top': open_bottom,
                })

            current_pos = open_pos + open_width / 2

        # Final segment
        if current_pos < length:
            segments.append({
                'start': current_pos,
                'end': length,
                'bottom': 0,
                'top': wall.height,
            })

        # Build segment objects
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))

        for i, seg in enumerate(segments):
            seg_length = seg['end'] - seg['start']
            seg_height = seg['top'] - seg['bottom']
            seg_center_along = (seg['start'] + seg['end']) / 2 - length / 2

            # Offset from wall center
            seg_x = cx + seg_center_along * sin_a
            seg_z = cz + seg_center_along * cos_a

            obj = SceneObject(
                name=f'{name}_seg{i}',
                transform=Transform(
                    position=(seg_x, seg['bottom'] + seg_height / 2, seg_z),
                    rotation=(0, angle, 0),
                ),
                geometry_type='box',
                geometry_params={
                    'size': (seg_length, seg_height, wall.thickness),
                },
                material=wall.material,
                is_obstacle=True,
                tags=['wall'],
            )
            self.add_object(obj)

    def _build_furniture(self, furniture: Furniture):
        """Build furniture as scene object."""
        obj = SceneObject(
            name=furniture.name,
            transform=Transform(
                position=(
                    furniture.position.x,
                    furniture.position.y + furniture.height / 2,
                    furniture.position.z,
                ),
                rotation=(0, furniture.rotation, 0),
            ),
            geometry_type=furniture.geometry,
            geometry_params={
                **furniture.geometry_params,  # Spread first so size isn't overwritten
                'size': (furniture.width, furniture.height, furniture.depth),
            },
            material=furniture.material,
            secondary_material=furniture.secondary_material,
            is_obstacle=furniture.is_obstacle,
            tags=['furniture', furniture.type.name.lower()],
        )
        self.add_object(obj)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_template(cls, template_name: str) -> 'HouseScene':
        """
        Create scene from a pre-built house template.

        Args:
            template_name: One of 'studio_apartment', 'two_bedroom'
        """
        if template_name not in HOUSE_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. "
                           f"Available: {list(HOUSE_TEMPLATES.keys())}")

        scene = cls(name=template_name)
        rooms = HOUSE_TEMPLATES[template_name]()

        for room in rooms:
            scene.add_room(room)

        # Add ambient light
        scene.add_light(SceneLight(
            name='ambient',
            light_type='ambient',
            color=scene.ambient_color,
            intensity=scene.ambient_intensity,
        ))

        # Add directional (sun) light
        scene.add_light(SceneLight(
            name='sun',
            light_type='directional',
            position=(5, 10, 5),
            direction=(-1, -2, -1),
            color=(255, 250, 240),
            intensity=0.5,
            cast_shadow=True,
        ))

        return scene

    @classmethod
    def from_room_scan(cls, scan_result: dict) -> 'HouseScene':
        """
        Create scene from room scanner results.

        Args:
            scan_result: Result dict from /room/scan endpoint
        """
        scene = cls(name='Scanned Room')

        # Parse scan result
        room_data = scan_result.get('room', {})
        width = room_data.get('width', 5.0)
        depth = room_data.get('depth', 5.0)
        height = room_data.get('height', 2.7)

        # Create room
        room = HouseAssets.create_rectangular_room(
            room_type=RoomType.LIVING_ROOM,
            name='scanned_room',
            width=width,
            depth=depth,
            height=height,
            floor_material=room_data.get('floor_material', 'oak_floor'),
            wall_material=room_data.get('wall_material', 'white_paint'),
        )

        # Add detected objects
        for asset in scan_result.get('generated_assets', []):
            asset_type = asset.get('type', 'box')
            pos = asset.get('position', [0, 0, 0])
            size = asset.get('size', [1, 1, 1])

            furniture = Furniture(
                name=asset.get('name', 'Object'),
                type=FurnitureType.DECOR,
                position=Vector3(pos[0], pos[1], pos[2]),
                width=size[0],
                height=size[1],
                depth=size[2],
                material=asset.get('material', 'gray_fabric'),
            )
            room.furniture.append(furniture)

        scene.add_room(room)

        # Add lighting
        scene.add_light(SceneLight(
            name='ambient',
            light_type='ambient',
            intensity=0.4,
        ))
        scene.add_light(SceneLight(
            name='ceiling',
            light_type='point',
            position=(width / 2, height - 0.2, depth / 2),
            intensity=1.0,
        ))

        return scene

    @classmethod
    def from_home_world(cls, home_world) -> 'HouseScene':
        """
        Create scene from home_world.py HomeWorld object.

        Args:
            home_world: HomeWorld instance from brain_b/simulator/home_world.py
        """
        scene = cls(name='HomeWorld')

        # Map home_world room types to our RoomType
        room_type_map = {
            'living_room': RoomType.LIVING_ROOM,
            'kitchen': RoomType.KITCHEN,
            'bedroom': RoomType.BEDROOM,
            'bathroom': RoomType.BATHROOM,
            'hallway': RoomType.HALLWAY,
            'garage': RoomType.GARAGE,
            'office': RoomType.OFFICE,
            'dining_room': RoomType.DINING_ROOM,
        }

        # Convert rooms
        for room_id, hw_room in home_world.rooms.items():
            bounds = hw_room.bounds
            width = bounds[1].x - bounds[0].x
            depth = bounds[1].y - bounds[0].y
            height = bounds[1].z - bounds[0].z

            room_type = room_type_map.get(hw_room.room_type.value, RoomType.LIVING_ROOM)

            room = HouseAssets.create_rectangular_room(
                room_type=room_type,
                name=room_id,
                width=width,
                depth=depth,
                height=height,
                origin=(bounds[0].x, bounds[0].y),
            )

            scene.add_room(room)

        # Convert objects
        for hw_obj in home_world.objects:
            obj = SceneObject(
                name=hw_obj.object_type.value,
                transform=Transform(
                    position=(
                        hw_obj.position.x + hw_obj.size[0] / 2,
                        hw_obj.position.z + hw_obj.size[2] / 2,
                        hw_obj.position.y + hw_obj.size[1] / 2,
                    ),
                ),
                geometry_type='box',
                geometry_params={
                    'size': (hw_obj.size[0], hw_obj.size[2], hw_obj.size[1]),
                },
                material='gray_fabric',
                is_obstacle=hw_obj.is_solid,
            )
            scene.add_object(obj)

        # Add default lighting
        scene.add_light(SceneLight(
            name='ambient',
            light_type='ambient',
            intensity=0.3,
        ))

        return scene

    @classmethod
    def from_json(cls, json_path: str) -> 'HouseScene':
        """Load scene from JSON file."""
        with open(json_path) as f:
            data = json.load(f)

        scene = cls(name=data.get('name', 'Loaded Scene'))

        # Load settings
        if 'ambient' in data:
            scene.ambient_color = tuple(data['ambient'].get('color', [40, 40, 50]))
            scene.ambient_intensity = data['ambient'].get('intensity', 0.3)

        # Load objects
        for obj_data in data.get('objects', []):
            obj = SceneObject(
                name=obj_data['name'],
                transform=Transform(
                    position=tuple(obj_data['transform']['position']),
                    rotation=tuple(obj_data['transform'].get('rotation', [0, 0, 0])),
                    scale=tuple(obj_data['transform'].get('scale', [1, 1, 1])),
                ),
                geometry_type=obj_data.get('geometryType', 'box'),
                geometry_params=obj_data.get('geometryParams', {}),
                material=obj_data.get('material', 'white_paint'),
            )
            scene.add_object(obj)

        # Load lights
        for light_data in data.get('lights', []):
            light = SceneLight(
                name=light_data['name'],
                light_type=light_data.get('type', 'point'),
                position=tuple(light_data.get('position', [0, 2, 0])),
                intensity=light_data.get('intensity', 1.0),
            )
            scene.add_light(light)

        return scene

    def to_json(self) -> dict:
        """Export scene to JSON-serializable dict."""
        return {
            'name': self.name,
            'bounds': {
                'min': list(self._bounds_min) if self._bounds_min[0] != float('inf') else [0, 0, 0],
                'max': list(self._bounds_max) if self._bounds_max[0] != float('-inf') else [10, 3, 10],
            },
            'ambient': {
                'color': list(self.ambient_color),
                'intensity': self.ambient_intensity,
            },
            'background': list(self.background_color),
            'fog': {
                'enabled': self.fog_enabled,
                'color': list(self.fog_color),
                'near': self.fog_near,
                'far': self.fog_far,
            },
            'objects': [obj.to_dict() for obj in self.objects],
            'lights': [light.to_dict() for light in self.lights],
        }

    def save(self, path: str):
        """Save scene to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, indent=2)

    def get_obstacles(self) -> List[SceneObject]:
        """Get all obstacle objects for collision detection."""
        return [obj for obj in self.objects if obj.is_obstacle]

    def get_objects_by_tag(self, tag: str) -> List[SceneObject]:
        """Get objects with a specific tag."""
        return [obj for obj in self.objects if tag in obj.tags]
