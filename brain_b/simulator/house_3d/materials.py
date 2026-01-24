"""
PBR Material Definitions for House 3D Renderer

Defines physically-based rendering materials for photorealistic appearance:
- Wood (oak, walnut, pine)
- Tile (ceramic, stone, marble)
- Fabric (cotton, leather, velvet)
- Metal (steel, brass, chrome)
- Wall finishes (paint, wallpaper, brick)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, Dict, Optional, List
import numpy as np


class TextureType(Enum):
    """Types of texture maps for PBR rendering."""
    ALBEDO = auto()      # Base color/diffuse
    NORMAL = auto()      # Surface normals
    ROUGHNESS = auto()   # Surface roughness
    METALLIC = auto()    # Metalness
    AO = auto()          # Ambient occlusion
    HEIGHT = auto()      # Displacement/height


@dataclass
class Material:
    """Base material definition."""
    name: str
    color: Tuple[int, int, int]  # RGB 0-255
    opacity: float = 1.0

    def to_rgb_float(self) -> Tuple[float, float, float]:
        """Convert color to 0-1 range."""
        return (self.color[0] / 255, self.color[1] / 255, self.color[2] / 255)

    def to_dict(self) -> dict:
        """Export for Three.js."""
        return {
            'name': self.name,
            'color': f'#{self.color[0]:02x}{self.color[1]:02x}{self.color[2]:02x}',
            'opacity': self.opacity,
        }


@dataclass
class PBRMaterial(Material):
    """
    Physically-Based Rendering material.

    Attributes:
        roughness: 0 = smooth/glossy, 1 = rough/matte
        metallic: 0 = dielectric, 1 = metallic
        emissive: Emission color (for lights, screens)
        normal_strength: Normal map influence
        ao_strength: Ambient occlusion strength
        texture_scale: UV tiling scale
    """
    roughness: float = 0.5
    metallic: float = 0.0
    emissive: Tuple[int, int, int] = (0, 0, 0)
    emissive_intensity: float = 0.0
    normal_strength: float = 1.0
    ao_strength: float = 1.0
    texture_scale: Tuple[float, float] = (1.0, 1.0)

    # Procedural texture parameters
    has_procedural_texture: bool = False
    texture_type: Optional[str] = None
    texture_params: Dict = field(default_factory=dict)

    def sample_color(self, u: float, v: float) -> Tuple[int, int, int]:
        """
        Sample color at UV coordinates, applying procedural texture if defined.

        Args:
            u, v: Texture coordinates (0-1, will be tiled)

        Returns:
            RGB color tuple
        """
        if not self.has_procedural_texture:
            return self.color

        # Apply texture scaling
        u = (u * self.texture_scale[0]) % 1.0
        v = (v * self.texture_scale[1]) % 1.0

        if self.texture_type == 'checkerboard':
            return self._sample_checkerboard(u, v)
        elif self.texture_type == 'wood_grain':
            return self._sample_wood_grain(u, v)
        elif self.texture_type == 'tile_grid':
            return self._sample_tile_grid(u, v)
        elif self.texture_type == 'brick':
            return self._sample_brick(u, v)
        elif self.texture_type == 'fabric_weave':
            return self._sample_fabric_weave(u, v)
        elif self.texture_type == 'marble':
            return self._sample_marble(u, v)
        elif self.texture_type == 'concrete':
            return self._sample_concrete(u, v)

        return self.color

    def _sample_checkerboard(self, u: float, v: float) -> Tuple[int, int, int]:
        """Checkerboard pattern."""
        size = self.texture_params.get('size', 8)
        color2 = self.texture_params.get('color2', (255, 255, 255))

        if (int(u * size) + int(v * size)) % 2 == 0:
            return self.color
        return color2

    def _sample_wood_grain(self, u: float, v: float) -> Tuple[int, int, int]:
        """Procedural wood grain pattern."""
        frequency = self.texture_params.get('frequency', 20)
        variation = self.texture_params.get('variation', 0.3)

        # Create grain pattern
        grain = np.sin(v * frequency + np.sin(u * 5) * 2)
        grain = (grain + 1) / 2  # Normalize to 0-1

        # Add noise variation
        noise = np.sin(u * 50) * np.cos(v * 50) * variation
        grain = np.clip(grain + noise, 0, 1)

        # Blend between base color and darker grain
        dark_factor = 0.7
        r = int(self.color[0] * (dark_factor + (1 - dark_factor) * grain))
        g = int(self.color[1] * (dark_factor + (1 - dark_factor) * grain))
        b = int(self.color[2] * (dark_factor + (1 - dark_factor) * grain))

        return (r, g, b)

    def _sample_tile_grid(self, u: float, v: float) -> Tuple[int, int, int]:
        """Tile grid with grout lines."""
        tile_size = self.texture_params.get('tile_size', 0.125)
        grout_width = self.texture_params.get('grout_width', 0.01)
        grout_color = self.texture_params.get('grout_color', (100, 100, 100))

        # Check if in grout region
        u_mod = u % tile_size
        v_mod = v % tile_size

        if u_mod < grout_width or v_mod < grout_width:
            return grout_color

        return self.color

    def _sample_brick(self, u: float, v: float) -> Tuple[int, int, int]:
        """Brick pattern with mortar."""
        brick_width = self.texture_params.get('brick_width', 0.25)
        brick_height = self.texture_params.get('brick_height', 0.1)
        mortar_width = self.texture_params.get('mortar_width', 0.01)
        mortar_color = self.texture_params.get('mortar_color', (180, 180, 170))

        # Offset every other row
        row = int(v / brick_height)
        if row % 2 == 1:
            u = (u + brick_width / 2) % 1.0

        u_mod = u % brick_width
        v_mod = v % brick_height

        # Check mortar
        if u_mod < mortar_width or v_mod < mortar_width:
            return mortar_color

        # Add slight color variation per brick
        brick_id = int(u / brick_width) + int(v / brick_height) * 100
        variation = (np.sin(brick_id * 12.9898) + 1) / 2 * 0.2 - 0.1

        r = int(np.clip(self.color[0] * (1 + variation), 0, 255))
        g = int(np.clip(self.color[1] * (1 + variation), 0, 255))
        b = int(np.clip(self.color[2] * (1 + variation), 0, 255))

        return (r, g, b)

    def _sample_fabric_weave(self, u: float, v: float) -> Tuple[int, int, int]:
        """Simple fabric weave pattern."""
        weave_size = self.texture_params.get('weave_size', 0.01)

        u_int = int(u / weave_size)
        v_int = int(v / weave_size)

        # Alternate light/dark for weave
        if (u_int + v_int) % 2 == 0:
            factor = 1.05
        else:
            factor = 0.95

        r = int(np.clip(self.color[0] * factor, 0, 255))
        g = int(np.clip(self.color[1] * factor, 0, 255))
        b = int(np.clip(self.color[2] * factor, 0, 255))

        return (r, g, b)

    def _sample_marble(self, u: float, v: float) -> Tuple[int, int, int]:
        """Procedural marble veining."""
        vein_color = self.texture_params.get('vein_color', (100, 100, 110))
        frequency = self.texture_params.get('frequency', 5)

        # Create veining pattern using turbulence
        turbulence = 0
        amp = 1.0
        for i in range(4):
            turbulence += abs(np.sin(u * frequency * (2 ** i) +
                                    np.cos(v * frequency * (2 ** i)))) * amp
            amp *= 0.5

        turbulence = turbulence / 2  # Normalize
        vein = np.sin(v * 10 + turbulence * 5)
        vein = (vein + 1) / 2
        vein = vein ** 3  # Sharpen veins

        r = int(self.color[0] * (1 - vein) + vein_color[0] * vein)
        g = int(self.color[1] * (1 - vein) + vein_color[1] * vein)
        b = int(self.color[2] * (1 - vein) + vein_color[2] * vein)

        return (r, g, b)

    def _sample_concrete(self, u: float, v: float) -> Tuple[int, int, int]:
        """Concrete with noise variation."""
        noise_scale = self.texture_params.get('noise_scale', 50)
        variation = self.texture_params.get('variation', 0.15)

        # Simple noise
        noise = np.sin(u * noise_scale) * np.cos(v * noise_scale * 1.3)
        noise += np.sin(u * noise_scale * 2.1) * np.cos(v * noise_scale * 2.7) * 0.5
        noise = noise * variation

        r = int(np.clip(self.color[0] * (1 + noise), 0, 255))
        g = int(np.clip(self.color[1] * (1 + noise), 0, 255))
        b = int(np.clip(self.color[2] * (1 + noise), 0, 255))

        return (r, g, b)

    def to_dict(self) -> dict:
        """Export for Three.js MeshStandardMaterial."""
        base = super().to_dict()
        base.update({
            'roughness': self.roughness,
            'metallic': self.metallic,
            'emissive': f'#{self.emissive[0]:02x}{self.emissive[1]:02x}{self.emissive[2]:02x}',
            'emissiveIntensity': self.emissive_intensity,
            'textureScale': list(self.texture_scale),
            'hasProceduralTexture': self.has_procedural_texture,
            'textureType': self.texture_type,
            'textureParams': self.texture_params,
        })
        return base


# =============================================================================
# Material Library
# =============================================================================

MATERIAL_LIBRARY: Dict[str, PBRMaterial] = {
    # ---------------------------------------------------------------------
    # Wood Materials
    # ---------------------------------------------------------------------
    'oak_floor': PBRMaterial(
        name='Oak Floor',
        color=(180, 140, 100),
        roughness=0.4,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='wood_grain',
        texture_scale=(2.0, 0.5),
        texture_params={'frequency': 30, 'variation': 0.2}
    ),

    'walnut': PBRMaterial(
        name='Walnut',
        color=(90, 60, 40),
        roughness=0.3,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='wood_grain',
        texture_scale=(1.0, 0.3),
        texture_params={'frequency': 25, 'variation': 0.25}
    ),

    'pine': PBRMaterial(
        name='Pine',
        color=(220, 190, 140),
        roughness=0.5,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='wood_grain',
        texture_scale=(1.5, 0.4),
        texture_params={'frequency': 20, 'variation': 0.15}
    ),

    'dark_hardwood': PBRMaterial(
        name='Dark Hardwood',
        color=(60, 40, 30),
        roughness=0.35,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='wood_grain',
        texture_scale=(2.0, 0.5),
        texture_params={'frequency': 35, 'variation': 0.2}
    ),

    # ---------------------------------------------------------------------
    # Tile Materials
    # ---------------------------------------------------------------------
    'white_ceramic': PBRMaterial(
        name='White Ceramic Tile',
        color=(245, 245, 245),
        roughness=0.2,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='tile_grid',
        texture_scale=(4.0, 4.0),
        texture_params={'tile_size': 0.25, 'grout_width': 0.02, 'grout_color': (200, 200, 195)}
    ),

    'gray_tile': PBRMaterial(
        name='Gray Tile',
        color=(140, 140, 145),
        roughness=0.25,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='tile_grid',
        texture_scale=(4.0, 4.0),
        texture_params={'tile_size': 0.25, 'grout_width': 0.015, 'grout_color': (100, 100, 100)}
    ),

    'terracotta': PBRMaterial(
        name='Terracotta Tile',
        color=(180, 100, 70),
        roughness=0.6,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='tile_grid',
        texture_scale=(3.0, 3.0),
        texture_params={'tile_size': 0.33, 'grout_width': 0.02, 'grout_color': (160, 150, 140)}
    ),

    'marble_white': PBRMaterial(
        name='White Marble',
        color=(250, 250, 248),
        roughness=0.15,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='marble',
        texture_scale=(1.0, 1.0),
        texture_params={'vein_color': (150, 150, 160), 'frequency': 3}
    ),

    'marble_black': PBRMaterial(
        name='Black Marble',
        color=(30, 30, 35),
        roughness=0.15,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='marble',
        texture_scale=(1.0, 1.0),
        texture_params={'vein_color': (80, 80, 90), 'frequency': 4}
    ),

    # ---------------------------------------------------------------------
    # Wall Materials
    # ---------------------------------------------------------------------
    'white_paint': PBRMaterial(
        name='White Paint',
        color=(255, 255, 255),
        roughness=0.8,
        metallic=0.0,
    ),

    'cream_paint': PBRMaterial(
        name='Cream Paint',
        color=(255, 250, 240),
        roughness=0.8,
        metallic=0.0,
    ),

    'light_gray_paint': PBRMaterial(
        name='Light Gray Paint',
        color=(220, 220, 225),
        roughness=0.75,
        metallic=0.0,
    ),

    'sage_green': PBRMaterial(
        name='Sage Green',
        color=(180, 200, 180),
        roughness=0.8,
        metallic=0.0,
    ),

    'navy_blue': PBRMaterial(
        name='Navy Blue',
        color=(40, 50, 80),
        roughness=0.75,
        metallic=0.0,
    ),

    'brick_red': PBRMaterial(
        name='Red Brick',
        color=(150, 70, 60),
        roughness=0.9,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='brick',
        texture_scale=(1.0, 1.0),
        texture_params={
            'brick_width': 0.12,
            'brick_height': 0.06,
            'mortar_width': 0.008,
            'mortar_color': (180, 175, 165)
        }
    ),

    'exposed_brick': PBRMaterial(
        name='Exposed Brick',
        color=(165, 85, 70),
        roughness=0.85,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='brick',
        texture_scale=(1.0, 1.0),
        texture_params={
            'brick_width': 0.11,
            'brick_height': 0.055,
            'mortar_width': 0.01,
            'mortar_color': (200, 195, 185)
        }
    ),

    'concrete_wall': PBRMaterial(
        name='Concrete',
        color=(160, 160, 155),
        roughness=0.9,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='concrete',
        texture_scale=(1.0, 1.0),
        texture_params={'noise_scale': 30, 'variation': 0.1}
    ),

    # ---------------------------------------------------------------------
    # Fabric Materials
    # ---------------------------------------------------------------------
    'gray_fabric': PBRMaterial(
        name='Gray Fabric',
        color=(130, 130, 135),
        roughness=0.9,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='fabric_weave',
        texture_scale=(20.0, 20.0),
        texture_params={'weave_size': 0.005}
    ),

    'beige_fabric': PBRMaterial(
        name='Beige Fabric',
        color=(210, 195, 175),
        roughness=0.85,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='fabric_weave',
        texture_scale=(20.0, 20.0),
        texture_params={'weave_size': 0.004}
    ),

    'brown_leather': PBRMaterial(
        name='Brown Leather',
        color=(100, 65, 45),
        roughness=0.5,
        metallic=0.0,
    ),

    'black_leather': PBRMaterial(
        name='Black Leather',
        color=(30, 30, 35),
        roughness=0.45,
        metallic=0.0,
    ),

    'velvet_blue': PBRMaterial(
        name='Blue Velvet',
        color=(40, 60, 100),
        roughness=0.95,
        metallic=0.0,
    ),

    'velvet_green': PBRMaterial(
        name='Green Velvet',
        color=(30, 80, 50),
        roughness=0.95,
        metallic=0.0,
    ),

    # ---------------------------------------------------------------------
    # Metal Materials
    # ---------------------------------------------------------------------
    'stainless_steel': PBRMaterial(
        name='Stainless Steel',
        color=(200, 200, 205),
        roughness=0.3,
        metallic=0.9,
    ),

    'brushed_steel': PBRMaterial(
        name='Brushed Steel',
        color=(180, 180, 185),
        roughness=0.5,
        metallic=0.85,
    ),

    'chrome': PBRMaterial(
        name='Chrome',
        color=(230, 230, 235),
        roughness=0.1,
        metallic=1.0,
    ),

    'brass': PBRMaterial(
        name='Brass',
        color=(210, 175, 90),
        roughness=0.35,
        metallic=0.9,
    ),

    'copper': PBRMaterial(
        name='Copper',
        color=(190, 110, 80),
        roughness=0.4,
        metallic=0.85,
    ),

    'matte_black_metal': PBRMaterial(
        name='Matte Black Metal',
        color=(35, 35, 40),
        roughness=0.7,
        metallic=0.8,
    ),

    # ---------------------------------------------------------------------
    # Glass & Transparent
    # ---------------------------------------------------------------------
    'clear_glass': PBRMaterial(
        name='Clear Glass',
        color=(230, 240, 245),
        roughness=0.0,
        metallic=0.0,
        opacity=0.3,
    ),

    'frosted_glass': PBRMaterial(
        name='Frosted Glass',
        color=(240, 245, 250),
        roughness=0.6,
        metallic=0.0,
        opacity=0.5,
    ),

    # ---------------------------------------------------------------------
    # Carpet
    # ---------------------------------------------------------------------
    'beige_carpet': PBRMaterial(
        name='Beige Carpet',
        color=(195, 180, 160),
        roughness=1.0,
        metallic=0.0,
    ),

    'gray_carpet': PBRMaterial(
        name='Gray Carpet',
        color=(140, 140, 145),
        roughness=1.0,
        metallic=0.0,
    ),

    'dark_carpet': PBRMaterial(
        name='Dark Carpet',
        color=(50, 50, 55),
        roughness=1.0,
        metallic=0.0,
    ),

    # ---------------------------------------------------------------------
    # Appliances & Electronics
    # ---------------------------------------------------------------------
    'white_appliance': PBRMaterial(
        name='White Appliance',
        color=(250, 250, 252),
        roughness=0.3,
        metallic=0.1,
    ),

    'black_appliance': PBRMaterial(
        name='Black Appliance',
        color=(25, 25, 30),
        roughness=0.25,
        metallic=0.15,
    ),

    'screen_off': PBRMaterial(
        name='Screen (Off)',
        color=(15, 15, 20),
        roughness=0.1,
        metallic=0.2,
    ),

    'screen_on': PBRMaterial(
        name='Screen (On)',
        color=(40, 60, 80),
        roughness=0.05,
        metallic=0.1,
        emissive=(100, 150, 200),
        emissive_intensity=0.3,
    ),

    # ---------------------------------------------------------------------
    # Counter & Tabletops
    # ---------------------------------------------------------------------
    'granite_black': PBRMaterial(
        name='Black Granite',
        color=(40, 40, 45),
        roughness=0.2,
        metallic=0.0,
    ),

    'granite_gray': PBRMaterial(
        name='Gray Granite',
        color=(130, 130, 135),
        roughness=0.25,
        metallic=0.0,
    ),

    'quartz_white': PBRMaterial(
        name='White Quartz',
        color=(245, 245, 243),
        roughness=0.2,
        metallic=0.0,
    ),

    'butcher_block': PBRMaterial(
        name='Butcher Block',
        color=(180, 130, 80),
        roughness=0.6,
        metallic=0.0,
        has_procedural_texture=True,
        texture_type='wood_grain',
        texture_scale=(0.5, 2.0),
        texture_params={'frequency': 40, 'variation': 0.15}
    ),
}


def get_material(name: str) -> PBRMaterial:
    """Get a material by name, with fallback to default."""
    if name in MATERIAL_LIBRARY:
        return MATERIAL_LIBRARY[name]

    # Return a default gray material
    return PBRMaterial(
        name=f'unknown_{name}',
        color=(150, 150, 150),
        roughness=0.5,
        metallic=0.0,
    )


def list_materials() -> List[str]:
    """List all available material names."""
    return sorted(MATERIAL_LIBRARY.keys())


def get_materials_by_category(category: str) -> Dict[str, PBRMaterial]:
    """Get materials by category prefix (e.g., 'wood', 'tile', 'metal')."""
    return {
        name: mat for name, mat in MATERIAL_LIBRARY.items()
        if category.lower() in name.lower()
    }
