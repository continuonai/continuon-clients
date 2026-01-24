"""
House Asset Definitions for 3D Training Environment

Defines room layouts, furniture catalog, and pre-built house templates
for photorealistic robot training scenarios.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class RoomType(Enum):
    """Types of rooms in a house."""
    LIVING_ROOM = auto()
    BEDROOM = auto()
    KITCHEN = auto()
    BATHROOM = auto()
    DINING_ROOM = auto()
    HALLWAY = auto()
    OFFICE = auto()
    GARAGE = auto()
    LAUNDRY = auto()
    CLOSET = auto()
    ENTRYWAY = auto()


class FurnitureType(Enum):
    """Categories of furniture."""
    SEATING = auto()      # Sofa, chair, stool
    TABLE = auto()        # Dining, coffee, desk
    STORAGE = auto()      # Cabinet, shelf, dresser
    BED = auto()          # Beds and mattresses
    APPLIANCE = auto()    # Kitchen/laundry appliances
    FIXTURE = auto()      # Toilet, sink, tub
    DECOR = auto()        # Plants, art, rugs
    ELECTRONICS = auto()  # TV, computer


@dataclass
class Vector3:
    """3D vector/point."""
    x: float
    y: float
    z: float

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> 'Vector3':
        return cls(t[0], t[1], t[2])


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_point: Vector3
    max_point: Vector3

    @property
    def size(self) -> Vector3:
        return self.max_point - self.min_point

    @property
    def center(self) -> Vector3:
        return Vector3(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2,
        )

    def contains(self, point: Vector3) -> bool:
        return (
            self.min_point.x <= point.x <= self.max_point.x and
            self.min_point.y <= point.y <= self.max_point.y and
            self.min_point.z <= point.z <= self.max_point.z
        )


@dataclass
class Wall:
    """
    Wall definition with position, dimensions, and materials.

    Walls are defined by start/end points (2D, on floor) and height.
    """
    start: Tuple[float, float]  # (x, z) floor position
    end: Tuple[float, float]    # (x, z) floor position
    height: float = 2.7         # Standard ceiling height in meters
    thickness: float = 0.15     # Wall thickness
    material: str = 'white_paint'
    has_baseboard: bool = True
    baseboard_height: float = 0.1
    baseboard_material: str = 'white_paint'

    # Openings (doors, windows)
    openings: List[Dict] = field(default_factory=list)

    @property
    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dz = self.end[1] - self.start[1]
        return np.sqrt(dx * dx + dz * dz)

    @property
    def normal(self) -> Tuple[float, float]:
        """Get wall normal (perpendicular to wall, on floor plane)."""
        dx = self.end[0] - self.start[0]
        dz = self.end[1] - self.start[1]
        length = self.length
        if length < 0.001:
            return (0, 1)
        # Rotate 90 degrees
        return (-dz / length, dx / length)

    def add_door(self, position: float, width: float = 0.9, height: float = 2.1):
        """Add a door opening at position along wall (0-1 normalized)."""
        self.openings.append({
            'type': 'door',
            'position': position,
            'width': width,
            'height': height,
            'bottom': 0.0,
        })

    def add_window(self, position: float, width: float = 1.2, height: float = 1.2, bottom: float = 0.9):
        """Add a window opening."""
        self.openings.append({
            'type': 'window',
            'position': position,
            'width': width,
            'height': height,
            'bottom': bottom,
        })

    def to_dict(self) -> dict:
        return {
            'start': self.start,
            'end': self.end,
            'height': self.height,
            'thickness': self.thickness,
            'material': self.material,
            'openings': self.openings,
            'hasBaseboard': self.has_baseboard,
            'baseboardHeight': self.baseboard_height,
            'baseboardMaterial': self.baseboard_material,
        }


@dataclass
class Floor:
    """Floor polygon with material."""
    vertices: List[Tuple[float, float]]  # 2D polygon vertices (x, z)
    material: str = 'oak_floor'
    y_position: float = 0.0  # Height of floor

    def to_dict(self) -> dict:
        return {
            'vertices': self.vertices,
            'material': self.material,
            'yPosition': self.y_position,
        }


@dataclass
class Ceiling:
    """Ceiling matching floor polygon."""
    vertices: List[Tuple[float, float]]
    material: str = 'white_paint'
    height: float = 2.7

    def to_dict(self) -> dict:
        return {
            'vertices': self.vertices,
            'material': self.material,
            'height': self.height,
        }


@dataclass
class Furniture:
    """
    Furniture item with position, rotation, and material.

    Position is the center-bottom of the bounding box.
    Rotation is around Y-axis in degrees.
    """
    name: str
    type: FurnitureType
    position: Vector3
    rotation: float = 0.0  # Degrees around Y-axis

    # Dimensions (bounding box)
    width: float = 1.0   # X dimension
    height: float = 1.0  # Y dimension
    depth: float = 1.0   # Z dimension

    # Appearance
    material: str = 'gray_fabric'
    secondary_material: Optional[str] = None  # For multi-material items

    # Geometry type for rendering
    geometry: str = 'box'  # 'box', 'cylinder', 'custom'
    geometry_params: Dict = field(default_factory=dict)

    # Collision
    is_obstacle: bool = True
    collision_margin: float = 0.05

    # Interaction
    is_interactive: bool = False
    interaction_type: Optional[str] = None  # 'sit', 'open', 'use'

    @property
    def bounding_box(self) -> BoundingBox:
        half_w = self.width / 2
        half_d = self.depth / 2
        return BoundingBox(
            min_point=Vector3(
                self.position.x - half_w,
                self.position.y,
                self.position.z - half_d
            ),
            max_point=Vector3(
                self.position.x + half_w,
                self.position.y + self.height,
                self.position.z + half_d
            )
        )

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'type': self.type.name,
            'position': self.position.to_tuple(),
            'rotation': self.rotation,
            'dimensions': {
                'width': self.width,
                'height': self.height,
                'depth': self.depth,
            },
            'material': self.material,
            'secondaryMaterial': self.secondary_material,
            'geometry': self.geometry,
            'geometryParams': self.geometry_params,
            'isObstacle': self.is_obstacle,
            'isInteractive': self.is_interactive,
            'interactionType': self.interaction_type,
        }


@dataclass
class LightFixture:
    """Light source in the scene."""
    name: str
    position: Vector3
    light_type: str = 'point'  # 'point', 'spot', 'area'
    color: Tuple[int, int, int] = (255, 250, 240)  # Warm white
    intensity: float = 1.0
    range: float = 10.0

    # For spot lights
    angle: float = 60.0
    direction: Optional[Vector3] = None

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'position': self.position.to_tuple(),
            'type': self.light_type,
            'color': f'#{self.color[0]:02x}{self.color[1]:02x}{self.color[2]:02x}',
            'intensity': self.intensity,
            'range': self.range,
            'angle': self.angle,
            'direction': self.direction.to_tuple() if self.direction else None,
        }


@dataclass
class Room:
    """
    Complete room definition with walls, floor, ceiling, and furniture.
    """
    name: str
    room_type: RoomType
    floor: Floor
    ceiling: Optional[Ceiling] = None
    walls: List[Wall] = field(default_factory=list)
    furniture: List[Furniture] = field(default_factory=list)
    lights: List[LightFixture] = field(default_factory=list)

    def add_furniture(self, furniture: Furniture):
        self.furniture.append(furniture)

    def add_wall(self, wall: Wall):
        self.walls.append(wall)

    def add_light(self, light: LightFixture):
        self.lights.append(light)

    def get_navigable_bounds(self) -> BoundingBox:
        """Get bounds of navigable floor area."""
        if not self.floor.vertices:
            return BoundingBox(Vector3(0, 0, 0), Vector3(1, 0, 1))

        xs = [v[0] for v in self.floor.vertices]
        zs = [v[1] for v in self.floor.vertices]
        return BoundingBox(
            min_point=Vector3(min(xs), 0, min(zs)),
            max_point=Vector3(max(xs), 0, max(zs))
        )

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'roomType': self.room_type.name,
            'floor': self.floor.to_dict(),
            'ceiling': self.ceiling.to_dict() if self.ceiling else None,
            'walls': [w.to_dict() for w in self.walls],
            'furniture': [f.to_dict() for f in self.furniture],
            'lights': [l.to_dict() for l in self.lights],
        }


# =============================================================================
# Furniture Catalog
# =============================================================================

class FurnitureCatalog:
    """Factory for creating common furniture items."""

    @staticmethod
    def sofa_3seat(position: Vector3, rotation: float = 0, material: str = 'gray_fabric') -> Furniture:
        return Furniture(
            name='Sofa (3-seat)',
            type=FurnitureType.SEATING,
            position=position,
            rotation=rotation,
            width=2.2,
            height=0.85,
            depth=0.9,
            material=material,
            geometry='custom',
            geometry_params={'type': 'sofa'},
            is_interactive=True,
            interaction_type='sit',
        )

    @staticmethod
    def sofa_2seat(position: Vector3, rotation: float = 0, material: str = 'gray_fabric') -> Furniture:
        return Furniture(
            name='Sofa (2-seat)',
            type=FurnitureType.SEATING,
            position=position,
            rotation=rotation,
            width=1.6,
            height=0.85,
            depth=0.85,
            material=material,
            geometry='custom',
            geometry_params={'type': 'sofa'},
            is_interactive=True,
            interaction_type='sit',
        )

    @staticmethod
    def armchair(position: Vector3, rotation: float = 0, material: str = 'beige_fabric') -> Furniture:
        return Furniture(
            name='Armchair',
            type=FurnitureType.SEATING,
            position=position,
            rotation=rotation,
            width=0.85,
            height=0.9,
            depth=0.85,
            material=material,
            geometry='custom',
            geometry_params={'type': 'armchair'},
            is_interactive=True,
            interaction_type='sit',
        )

    @staticmethod
    def dining_chair(position: Vector3, rotation: float = 0, material: str = 'walnut') -> Furniture:
        return Furniture(
            name='Dining Chair',
            type=FurnitureType.SEATING,
            position=position,
            rotation=rotation,
            width=0.45,
            height=0.95,
            depth=0.45,
            material=material,
            geometry='custom',
            geometry_params={'type': 'chair'},
            is_interactive=True,
            interaction_type='sit',
        )

    @staticmethod
    def office_chair(position: Vector3, rotation: float = 0, material: str = 'black_leather') -> Furniture:
        return Furniture(
            name='Office Chair',
            type=FurnitureType.SEATING,
            position=position,
            rotation=rotation,
            width=0.65,
            height=1.1,
            depth=0.65,
            material=material,
            geometry='custom',
            geometry_params={'type': 'office_chair'},
            is_interactive=True,
            interaction_type='sit',
        )

    @staticmethod
    def coffee_table(position: Vector3, rotation: float = 0, material: str = 'walnut') -> Furniture:
        return Furniture(
            name='Coffee Table',
            type=FurnitureType.TABLE,
            position=position,
            rotation=rotation,
            width=1.2,
            height=0.45,
            depth=0.6,
            material=material,
            geometry='box',
        )

    @staticmethod
    def dining_table(position: Vector3, rotation: float = 0, material: str = 'oak_floor') -> Furniture:
        return Furniture(
            name='Dining Table',
            type=FurnitureType.TABLE,
            position=position,
            rotation=rotation,
            width=1.8,
            height=0.75,
            depth=0.9,
            material=material,
            geometry='box',
        )

    @staticmethod
    def desk(position: Vector3, rotation: float = 0, material: str = 'walnut') -> Furniture:
        return Furniture(
            name='Desk',
            type=FurnitureType.TABLE,
            position=position,
            rotation=rotation,
            width=1.4,
            height=0.75,
            depth=0.7,
            material=material,
            geometry='custom',
            geometry_params={'type': 'desk'},
        )

    @staticmethod
    def nightstand(position: Vector3, rotation: float = 0, material: str = 'walnut') -> Furniture:
        return Furniture(
            name='Nightstand',
            type=FurnitureType.TABLE,
            position=position,
            rotation=rotation,
            width=0.5,
            height=0.55,
            depth=0.4,
            material=material,
            geometry='box',
        )

    @staticmethod
    def bed_queen(position: Vector3, rotation: float = 0, frame_material: str = 'walnut', bedding_material: str = 'beige_fabric') -> Furniture:
        return Furniture(
            name='Queen Bed',
            type=FurnitureType.BED,
            position=position,
            rotation=rotation,
            width=1.6,
            height=0.6,
            depth=2.1,
            material=frame_material,
            secondary_material=bedding_material,
            geometry='custom',
            geometry_params={'type': 'bed'},
        )

    @staticmethod
    def bed_king(position: Vector3, rotation: float = 0, frame_material: str = 'walnut', bedding_material: str = 'gray_fabric') -> Furniture:
        return Furniture(
            name='King Bed',
            type=FurnitureType.BED,
            position=position,
            rotation=rotation,
            width=1.9,
            height=0.6,
            depth=2.1,
            material=frame_material,
            secondary_material=bedding_material,
            geometry='custom',
            geometry_params={'type': 'bed'},
        )

    @staticmethod
    def dresser(position: Vector3, rotation: float = 0, material: str = 'walnut') -> Furniture:
        return Furniture(
            name='Dresser',
            type=FurnitureType.STORAGE,
            position=position,
            rotation=rotation,
            width=1.2,
            height=0.85,
            depth=0.5,
            material=material,
            geometry='box',
            is_interactive=True,
            interaction_type='open',
        )

    @staticmethod
    def bookshelf(position: Vector3, rotation: float = 0, material: str = 'walnut') -> Furniture:
        return Furniture(
            name='Bookshelf',
            type=FurnitureType.STORAGE,
            position=position,
            rotation=rotation,
            width=0.8,
            height=1.8,
            depth=0.3,
            material=material,
            geometry='custom',
            geometry_params={'type': 'bookshelf'},
        )

    @staticmethod
    def tv_console(position: Vector3, rotation: float = 0, material: str = 'matte_black_metal') -> Furniture:
        return Furniture(
            name='TV Console',
            type=FurnitureType.STORAGE,
            position=position,
            rotation=rotation,
            width=1.5,
            height=0.5,
            depth=0.45,
            material=material,
            geometry='box',
        )

    @staticmethod
    def tv(position: Vector3, rotation: float = 0, screen_on: bool = False) -> Furniture:
        return Furniture(
            name='TV',
            type=FurnitureType.ELECTRONICS,
            position=position,
            rotation=rotation,
            width=1.4,
            height=0.8,
            depth=0.08,
            material='screen_on' if screen_on else 'screen_off',
            secondary_material='matte_black_metal',
            geometry='custom',
            geometry_params={'type': 'tv'},
        )

    @staticmethod
    def refrigerator(position: Vector3, rotation: float = 0, material: str = 'stainless_steel') -> Furniture:
        return Furniture(
            name='Refrigerator',
            type=FurnitureType.APPLIANCE,
            position=position,
            rotation=rotation,
            width=0.9,
            height=1.8,
            depth=0.8,
            material=material,
            geometry='box',
            is_interactive=True,
            interaction_type='open',
        )

    @staticmethod
    def kitchen_counter(position: Vector3, rotation: float = 0, length: float = 2.0, material: str = 'granite_gray') -> Furniture:
        return Furniture(
            name='Kitchen Counter',
            type=FurnitureType.STORAGE,
            position=position,
            rotation=rotation,
            width=length,
            height=0.9,
            depth=0.6,
            material=material,
            secondary_material='white_appliance',
            geometry='custom',
            geometry_params={'type': 'counter'},
        )

    @staticmethod
    def kitchen_cabinet_upper(position: Vector3, rotation: float = 0, length: float = 1.0, material: str = 'white_appliance') -> Furniture:
        return Furniture(
            name='Upper Cabinet',
            type=FurnitureType.STORAGE,
            position=position,
            rotation=rotation,
            width=length,
            height=0.7,
            depth=0.35,
            material=material,
            geometry='box',
            is_interactive=True,
            interaction_type='open',
        )

    @staticmethod
    def stove(position: Vector3, rotation: float = 0, material: str = 'stainless_steel') -> Furniture:
        return Furniture(
            name='Stove',
            type=FurnitureType.APPLIANCE,
            position=position,
            rotation=rotation,
            width=0.76,
            height=0.9,
            depth=0.65,
            material=material,
            geometry='custom',
            geometry_params={'type': 'stove'},
            is_interactive=True,
            interaction_type='use',
        )

    @staticmethod
    def dishwasher(position: Vector3, rotation: float = 0, material: str = 'stainless_steel') -> Furniture:
        return Furniture(
            name='Dishwasher',
            type=FurnitureType.APPLIANCE,
            position=position,
            rotation=rotation,
            width=0.6,
            height=0.85,
            depth=0.6,
            material=material,
            geometry='box',
            is_interactive=True,
            interaction_type='open',
        )

    @staticmethod
    def toilet(position: Vector3, rotation: float = 0) -> Furniture:
        return Furniture(
            name='Toilet',
            type=FurnitureType.FIXTURE,
            position=position,
            rotation=rotation,
            width=0.45,
            height=0.75,
            depth=0.7,
            material='white_ceramic',
            geometry='custom',
            geometry_params={'type': 'toilet'},
        )

    @staticmethod
    def bathroom_sink(position: Vector3, rotation: float = 0) -> Furniture:
        return Furniture(
            name='Bathroom Sink',
            type=FurnitureType.FIXTURE,
            position=position,
            rotation=rotation,
            width=0.6,
            height=0.85,
            depth=0.5,
            material='white_ceramic',
            secondary_material='chrome',
            geometry='custom',
            geometry_params={'type': 'vanity'},
        )

    @staticmethod
    def bathtub(position: Vector3, rotation: float = 0) -> Furniture:
        return Furniture(
            name='Bathtub',
            type=FurnitureType.FIXTURE,
            position=position,
            rotation=rotation,
            width=0.8,
            height=0.6,
            depth=1.7,
            material='white_ceramic',
            geometry='custom',
            geometry_params={'type': 'bathtub'},
        )

    @staticmethod
    def shower(position: Vector3, rotation: float = 0) -> Furniture:
        return Furniture(
            name='Shower',
            type=FurnitureType.FIXTURE,
            position=position,
            rotation=rotation,
            width=0.9,
            height=2.2,
            depth=0.9,
            material='clear_glass',
            secondary_material='white_ceramic',
            geometry='custom',
            geometry_params={'type': 'shower'},
        )

    @staticmethod
    def plant_potted(position: Vector3, size: str = 'medium') -> Furniture:
        sizes = {
            'small': (0.3, 0.4, 0.3),
            'medium': (0.4, 0.8, 0.4),
            'large': (0.6, 1.5, 0.6),
        }
        w, h, d = sizes.get(size, sizes['medium'])
        return Furniture(
            name=f'Potted Plant ({size})',
            type=FurnitureType.DECOR,
            position=position,
            width=w,
            height=h,
            depth=d,
            material='terracotta',
            geometry='custom',
            geometry_params={'type': 'plant', 'size': size},
            is_obstacle=True,
            collision_margin=0.1,
        )

    @staticmethod
    def rug(position: Vector3, width: float = 2.0, depth: float = 3.0, material: str = 'beige_carpet') -> Furniture:
        return Furniture(
            name='Area Rug',
            type=FurnitureType.DECOR,
            position=position,
            width=width,
            height=0.02,
            depth=depth,
            material=material,
            geometry='box',
            is_obstacle=False,
        )

    @staticmethod
    def washer(position: Vector3, rotation: float = 0) -> Furniture:
        return Furniture(
            name='Washing Machine',
            type=FurnitureType.APPLIANCE,
            position=position,
            rotation=rotation,
            width=0.6,
            height=0.85,
            depth=0.6,
            material='white_appliance',
            geometry='box',
            is_interactive=True,
            interaction_type='use',
        )

    @staticmethod
    def dryer(position: Vector3, rotation: float = 0) -> Furniture:
        return Furniture(
            name='Dryer',
            type=FurnitureType.APPLIANCE,
            position=position,
            rotation=rotation,
            width=0.6,
            height=0.85,
            depth=0.6,
            material='white_appliance',
            geometry='box',
            is_interactive=True,
            interaction_type='use',
        )


# =============================================================================
# House Templates
# =============================================================================

class HouseAssets:
    """Factory for creating pre-built house layouts."""

    @staticmethod
    def create_rectangular_room(
        room_type: RoomType,
        name: str,
        width: float,
        depth: float,
        height: float = 2.7,
        floor_material: str = 'oak_floor',
        wall_material: str = 'white_paint',
        origin: Tuple[float, float] = (0, 0),
    ) -> Room:
        """Create a simple rectangular room."""
        ox, oz = origin

        # Floor polygon
        floor = Floor(
            vertices=[
                (ox, oz),
                (ox + width, oz),
                (ox + width, oz + depth),
                (ox, oz + depth),
            ],
            material=floor_material,
        )

        # Ceiling
        ceiling = Ceiling(
            vertices=floor.vertices.copy(),
            height=height,
        )

        # Walls (clockwise from origin)
        walls = [
            Wall(start=(ox, oz), end=(ox + width, oz), height=height, material=wall_material),
            Wall(start=(ox + width, oz), end=(ox + width, oz + depth), height=height, material=wall_material),
            Wall(start=(ox + width, oz + depth), end=(ox, oz + depth), height=height, material=wall_material),
            Wall(start=(ox, oz + depth), end=(ox, oz), height=height, material=wall_material),
        ]

        # Center ceiling light
        lights = [
            LightFixture(
                name='Ceiling Light',
                position=Vector3(ox + width / 2, height - 0.1, oz + depth / 2),
                light_type='point',
                intensity=1.0,
            )
        ]

        return Room(
            name=name,
            room_type=room_type,
            floor=floor,
            ceiling=ceiling,
            walls=walls,
            lights=lights,
        )

    @staticmethod
    def studio_apartment() -> List[Room]:
        """
        Create a studio apartment layout.

        ~500 sq ft open plan with kitchen area, living area, and bathroom.
        """
        rooms = []

        # Main open room (kitchen + living) - 6m x 7m
        main_room = HouseAssets.create_rectangular_room(
            room_type=RoomType.LIVING_ROOM,
            name='Main Room',
            width=6.0,
            depth=7.0,
            floor_material='oak_floor',
            wall_material='cream_paint',
        )

        # Add door to outside (south wall)
        main_room.walls[0].add_door(position=0.3)

        # Add windows
        main_room.walls[1].add_window(position=0.5, width=1.5)  # East wall
        main_room.walls[2].add_window(position=0.3, width=1.2)  # North wall
        main_room.walls[2].add_window(position=0.7, width=1.2)  # North wall

        # Furniture - Living area (south portion)
        catalog = FurnitureCatalog

        # Sofa and coffee table
        main_room.add_furniture(catalog.sofa_3seat(Vector3(3.0, 0, 2.5), rotation=0, material='gray_fabric'))
        main_room.add_furniture(catalog.coffee_table(Vector3(3.0, 0, 3.5), material='walnut'))
        main_room.add_furniture(catalog.armchair(Vector3(1.2, 0, 3.2), rotation=45, material='beige_fabric'))

        # TV area
        main_room.add_furniture(catalog.tv_console(Vector3(3.0, 0, 5.5)))
        main_room.add_furniture(catalog.tv(Vector3(3.0, 0.5, 5.6), screen_on=True))

        # Area rug
        main_room.add_furniture(catalog.rug(Vector3(3.0, 0, 3.5), width=2.5, depth=3.5, material='beige_carpet'))

        # Kitchen area (north portion)
        main_room.add_furniture(catalog.refrigerator(Vector3(0.5, 0, 6.5), rotation=180))
        main_room.add_furniture(catalog.kitchen_counter(Vector3(2.0, 0, 6.5), rotation=0, length=2.0, material='granite_gray'))
        main_room.add_furniture(catalog.stove(Vector3(4.5, 0, 6.5), rotation=180))
        main_room.add_furniture(catalog.kitchen_cabinet_upper(Vector3(2.0, 1.5, 6.7), length=2.0))

        # Dining
        main_room.add_furniture(catalog.dining_table(Vector3(5.0, 0, 4.5), material='walnut'))
        main_room.add_furniture(catalog.dining_chair(Vector3(4.5, 0, 4.0), rotation=0))
        main_room.add_furniture(catalog.dining_chair(Vector3(5.5, 0, 4.0), rotation=0))
        main_room.add_furniture(catalog.dining_chair(Vector3(4.5, 0, 5.0), rotation=180))
        main_room.add_furniture(catalog.dining_chair(Vector3(5.5, 0, 5.0), rotation=180))

        # Bed area (corner)
        main_room.add_furniture(catalog.bed_queen(Vector3(1.2, 0, 0.8), rotation=90, bedding_material='navy_blue'))
        main_room.add_furniture(catalog.nightstand(Vector3(0.3, 0, 1.8)))

        # Plants
        main_room.add_furniture(catalog.plant_potted(Vector3(5.7, 0, 0.3), size='large'))

        rooms.append(main_room)

        # Bathroom - 2m x 2.5m (attached to main room)
        bathroom = HouseAssets.create_rectangular_room(
            room_type=RoomType.BATHROOM,
            name='Bathroom',
            width=2.0,
            depth=2.5,
            floor_material='white_ceramic',
            wall_material='light_gray_paint',
            origin=(6.0, 0),
        )

        # Remove shared wall
        bathroom.walls[3].add_door(position=0.5, width=0.8)

        # Bathroom fixtures
        bathroom.add_furniture(catalog.toilet(Vector3(6.4, 0, 0.5), rotation=-90))
        bathroom.add_furniture(catalog.bathroom_sink(Vector3(7.5, 0, 0.5), rotation=90))
        bathroom.add_furniture(catalog.shower(Vector3(6.5, 0, 2.0)))

        rooms.append(bathroom)

        return rooms

    @staticmethod
    def two_bedroom_apartment() -> List[Room]:
        """
        Create a 2-bedroom apartment layout.

        ~900 sq ft with living room, kitchen, 2 bedrooms, bathroom.
        """
        rooms = []
        catalog = FurnitureCatalog

        # Living Room - 5m x 5m
        living = HouseAssets.create_rectangular_room(
            room_type=RoomType.LIVING_ROOM,
            name='Living Room',
            width=5.0,
            depth=5.0,
            floor_material='oak_floor',
            wall_material='cream_paint',
        )

        living.walls[0].add_door(position=0.2)  # Entry
        living.walls[1].add_window(position=0.5, width=2.0)
        living.walls[2].add_door(position=0.8)  # To kitchen

        living.add_furniture(catalog.sofa_3seat(Vector3(2.5, 0, 1.5), rotation=0, material='velvet_blue'))
        living.add_furniture(catalog.coffee_table(Vector3(2.5, 0, 2.5)))
        living.add_furniture(catalog.armchair(Vector3(0.8, 0, 2.5), rotation=60))
        living.add_furniture(catalog.tv_console(Vector3(2.5, 0, 4.3)))
        living.add_furniture(catalog.tv(Vector3(2.5, 0.5, 4.4), screen_on=True))
        living.add_furniture(catalog.bookshelf(Vector3(4.6, 0, 2.5)))
        living.add_furniture(catalog.rug(Vector3(2.5, 0, 2.5), width=3.0, depth=2.5))
        living.add_furniture(catalog.plant_potted(Vector3(0.4, 0, 4.6), size='medium'))

        rooms.append(living)

        # Kitchen - 3m x 4m
        kitchen = HouseAssets.create_rectangular_room(
            room_type=RoomType.KITCHEN,
            name='Kitchen',
            width=3.0,
            depth=4.0,
            floor_material='gray_tile',
            wall_material='white_paint',
            origin=(5.0, 0),
        )

        kitchen.walls[3].add_door(position=0.3)  # To living room
        kitchen.walls[1].add_window(position=0.5)

        kitchen.add_furniture(catalog.refrigerator(Vector3(5.5, 0, 3.5), rotation=180))
        kitchen.add_furniture(catalog.kitchen_counter(Vector3(6.5, 0, 3.5), length=1.5))
        kitchen.add_furniture(catalog.stove(Vector3(7.5, 0, 3.5), rotation=180))
        kitchen.add_furniture(catalog.dishwasher(Vector3(6.0, 0, 0.5), rotation=0))
        kitchen.add_furniture(catalog.kitchen_cabinet_upper(Vector3(6.5, 1.5, 3.7), length=2.0))

        rooms.append(kitchen)

        # Master Bedroom - 4m x 4m
        master = HouseAssets.create_rectangular_room(
            room_type=RoomType.BEDROOM,
            name='Master Bedroom',
            width=4.0,
            depth=4.0,
            floor_material='dark_hardwood',
            wall_material='sage_green',
            origin=(0, 5.0),
        )

        master.walls[0].add_door(position=0.8)  # To hallway
        master.walls[2].add_window(position=0.5, width=1.5)

        master.add_furniture(catalog.bed_queen(Vector3(2.0, 0, 2.5), rotation=0, bedding_material='beige_fabric'))
        master.add_furniture(catalog.nightstand(Vector3(0.5, 0, 2.5)))
        master.add_furniture(catalog.nightstand(Vector3(3.5, 0, 2.5)))
        master.add_furniture(catalog.dresser(Vector3(0.6, 0, 0.5), rotation=90))

        rooms.append(master)

        # Second Bedroom - 3.5m x 3.5m
        bedroom2 = HouseAssets.create_rectangular_room(
            room_type=RoomType.BEDROOM,
            name='Bedroom 2',
            width=3.5,
            depth=3.5,
            floor_material='oak_floor',
            wall_material='light_gray_paint',
            origin=(4.0, 5.0),
        )

        bedroom2.walls[0].add_door(position=0.3)
        bedroom2.walls[1].add_window(position=0.5)

        bedroom2.add_furniture(catalog.bed_queen(Vector3(5.75, 0, 7.0), rotation=0, bedding_material='gray_fabric'))
        bedroom2.add_furniture(catalog.nightstand(Vector3(4.5, 0, 7.0)))
        bedroom2.add_furniture(catalog.desk(Vector3(4.7, 0, 5.5), rotation=-90))
        bedroom2.add_furniture(catalog.office_chair(Vector3(5.5, 0, 5.5), rotation=-90))

        rooms.append(bedroom2)

        # Bathroom - 2.5m x 3m
        bathroom = HouseAssets.create_rectangular_room(
            room_type=RoomType.BATHROOM,
            name='Bathroom',
            width=2.5,
            depth=3.0,
            floor_material='white_ceramic',
            wall_material='white_paint',
            origin=(7.5, 5.0),
        )

        bathroom.walls[3].add_door(position=0.5)

        bathroom.add_furniture(catalog.toilet(Vector3(8.0, 0, 5.5), rotation=-90))
        bathroom.add_furniture(catalog.bathroom_sink(Vector3(9.5, 0, 5.5), rotation=90))
        bathroom.add_furniture(catalog.bathtub(Vector3(8.75, 0, 7.5), rotation=0))

        rooms.append(bathroom)

        return rooms


# Pre-built house templates
HOUSE_TEMPLATES = {
    'studio_apartment': HouseAssets.studio_apartment,
    'two_bedroom': HouseAssets.two_bedroom_apartment,
}
