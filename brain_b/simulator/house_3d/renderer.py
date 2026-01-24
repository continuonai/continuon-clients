"""
Software 3D Renderer for House Scene

Generates RGB, depth, and segmentation frames for training data.
Uses NumPy for fast vectorized operations.

Features:
- Ray-traced rendering with material support
- Depth buffer generation
- Semantic segmentation
- Basic lighting and shadows
- Procedural textures
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto

from .scene import HouseScene, SceneObject, SceneLight, Transform
from .camera import Camera, PerspectiveCamera, OrthographicCamera, RobotCamera
from .materials import get_material, PBRMaterial


class RenderMode(Enum):
    """Rendering output modes."""
    RGB = auto()
    DEPTH = auto()
    SEGMENTATION = auto()
    NORMALS = auto()


@dataclass
class RenderSettings:
    """Settings for the renderer."""
    width: int = 640
    height: int = 480
    samples_per_pixel: int = 1  # Anti-aliasing (1 = no AA)
    max_bounces: int = 2
    shadow_samples: int = 1
    ambient_occlusion: bool = False
    ao_samples: int = 4
    ao_radius: float = 0.5


@dataclass
class LightSource:
    """Processed light for rendering."""
    position: np.ndarray
    color: np.ndarray  # RGB 0-1
    intensity: float
    light_type: str
    direction: Optional[np.ndarray] = None
    range: float = 10.0
    angle: float = 60.0


@dataclass
class RenderableObject:
    """Pre-processed object for fast rendering."""
    name: str
    obj_type: str
    center: np.ndarray
    half_size: np.ndarray
    rotation_matrix: np.ndarray
    inv_rotation: np.ndarray
    material: PBRMaterial
    is_plane: bool = False
    plane_normal: Optional[np.ndarray] = None


class HouseRenderer:
    """
    Software 3D renderer for house scenes.

    Generates training data with:
    - RGB frames from robot POV
    - Depth maps for 3D understanding
    - Semantic segmentation for object detection
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        settings: Optional[RenderSettings] = None,
    ):
        self.width = width
        self.height = height
        self.settings = settings or RenderSettings(width=width, height=height)

        # Pre-computed ray directions (for perspective camera)
        self._ray_dirs_cache = {}

        # Segmentation color map
        self._seg_colors = {
            'floor': np.array([100, 100, 100]),
            'ceiling': np.array([200, 200, 200]),
            'wall': np.array([150, 150, 180]),
            'furniture': np.array([100, 150, 100]),
            'seating': np.array([80, 120, 180]),
            'table': np.array([180, 120, 80]),
            'bed': np.array([180, 80, 120]),
            'appliance': np.array([80, 180, 180]),
            'fixture': np.array([180, 180, 80]),
            'decor': np.array([120, 180, 120]),
            'storage': np.array([150, 100, 150]),
            'electronics': np.array([100, 100, 180]),
            'default': np.array([128, 128, 128]),
        }

    def render(
        self,
        scene: HouseScene,
        camera: Camera,
        mode: RenderMode = RenderMode.RGB,
    ) -> np.ndarray:
        """
        Render the scene from camera viewpoint.

        Args:
            scene: HouseScene to render
            camera: Camera (perspective or orthographic)
            mode: What to render (RGB, depth, segmentation)

        Returns:
            NumPy array of shape (height, width, channels)
            - RGB: (H, W, 3) uint8
            - DEPTH: (H, W) float32
            - SEGMENTATION: (H, W, 3) uint8
            - NORMALS: (H, W, 3) float32
        """
        # Prepare scene for rendering
        renderables = self._prepare_scene(scene)
        lights = self._prepare_lights(scene)

        # Generate rays
        if isinstance(camera, OrthographicCamera):
            origins, directions = self._generate_ortho_rays(camera)
        else:
            origins, directions = self._generate_perspective_rays(camera)

        # Ray trace
        if mode == RenderMode.RGB:
            return self._render_rgb(renderables, lights, origins, directions, scene)
        elif mode == RenderMode.DEPTH:
            return self._render_depth(renderables, origins, directions)
        elif mode == RenderMode.SEGMENTATION:
            return self._render_segmentation(renderables, origins, directions)
        elif mode == RenderMode.NORMALS:
            return self._render_normals(renderables, origins, directions)

    def render_pov(
        self,
        scene: HouseScene,
        robot_position: Tuple[float, float, float],
        robot_yaw: float,
        robot_pitch: float = 0.0,
        eye_height: float = 0.8,
    ) -> np.ndarray:
        """
        Render robot's point-of-view RGB image.

        Args:
            scene: House scene
            robot_position: (x, y, z) world position
            robot_yaw: Heading in degrees
            robot_pitch: Looking up/down
            eye_height: Camera height on robot

        Returns:
            RGB image as (H, W, 3) uint8 array
        """
        camera = RobotCamera(eye_height=eye_height, aspect=self.width / self.height)
        camera.update_from_robot(robot_position, robot_yaw, robot_pitch)
        return self.render(scene, camera, RenderMode.RGB)

    def render_overhead(
        self,
        scene: HouseScene,
        robot_position: Optional[Tuple[float, float, float]] = None,
        size: int = 256,
    ) -> np.ndarray:
        """
        Render overhead map view.

        Args:
            scene: House scene
            robot_position: Optional robot position to show marker
            size: Output image size (square)

        Returns:
            RGB image as (size, size, 3) uint8 array
        """
        # Create orthographic camera looking down
        floor_bounds = scene.get_floor_bounds()
        camera = OrthographicCamera.from_bounds(floor_bounds[0], floor_bounds[1])

        # Temporarily adjust renderer size
        old_w, old_h = self.width, self.height
        self.width = self.height = size

        frame = self.render(scene, camera, RenderMode.RGB)

        self.width, self.height = old_w, old_h

        # Add robot marker if position given
        if robot_position:
            frame = self._add_robot_marker(frame, robot_position, floor_bounds, size)

        return frame

    def render_depth(
        self,
        scene: HouseScene,
        camera: Camera,
        max_depth: float = 10.0,
    ) -> np.ndarray:
        """
        Render depth map.

        Args:
            scene: House scene
            camera: Camera
            max_depth: Maximum depth for normalization

        Returns:
            Depth map as (H, W) float32 array (0 = near, 1 = far)
        """
        depth = self.render(scene, camera, RenderMode.DEPTH)
        return np.clip(depth / max_depth, 0, 1).astype(np.float32)

    def render_split_view(
        self,
        scene: HouseScene,
        robot_position: Tuple[float, float, float],
        robot_yaw: float,
        robot_pitch: float = 0.0,
        pov_size: Tuple[int, int] = (640, 480),
        overhead_size: int = 256,
    ) -> Dict[str, np.ndarray]:
        """
        Render split-view output (POV + overhead + depth).

        Returns:
            Dict with 'pov', 'overhead', 'depth' arrays
        """
        # POV camera
        old_w, old_h = self.width, self.height
        self.width, self.height = pov_size

        pov_camera = RobotCamera(eye_height=0.8, aspect=pov_size[0] / pov_size[1])
        pov_camera.update_from_robot(robot_position, robot_yaw, robot_pitch)

        pov_rgb = self.render(scene, pov_camera, RenderMode.RGB)
        pov_depth = self.render(scene, pov_camera, RenderMode.DEPTH)

        self.width, self.height = old_w, old_h

        # Overhead
        overhead = self.render_overhead(scene, robot_position, overhead_size)

        return {
            'pov': pov_rgb,
            'depth': pov_depth,
            'overhead': overhead,
        }

    # =========================================================================
    # Ray Generation
    # =========================================================================

    def _generate_perspective_rays(self, camera: PerspectiveCamera) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ray origins and directions for perspective camera."""
        cache_key = (self.width, self.height, camera.fov, camera.aspect)

        if cache_key not in self._ray_dirs_cache:
            # Generate pixel coordinates
            u = (np.arange(self.width) + 0.5) / self.width
            v = (np.arange(self.height) + 0.5) / self.height

            u, v = np.meshgrid(u, v)

            # NDC to view space
            ndc_x = u * 2 - 1
            ndc_y = 1 - v * 2  # Flip Y

            fov_rad = math.radians(camera.fov)
            tan_half_fov = math.tan(fov_rad / 2)

            view_x = ndc_x * tan_half_fov * camera.aspect
            view_y = ndc_y * tan_half_fov
            view_z = -np.ones_like(view_x)

            # Stack and normalize
            dirs = np.stack([view_x, view_y, view_z], axis=-1)
            dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

            self._ray_dirs_cache[cache_key] = dirs

        # Get cached directions in view space
        view_dirs = self._ray_dirs_cache[cache_key].copy()

        # Transform to world space
        view_matrix = camera.get_view_matrix()
        view_inv = np.linalg.inv(view_matrix)

        # Rotate directions
        world_dirs = np.einsum('ij,hwj->hwi', view_inv[:3, :3], view_dirs)
        world_dirs = world_dirs / np.linalg.norm(world_dirs, axis=-1, keepdims=True)

        # Origins all at camera position
        origins = np.broadcast_to(
            np.array(camera.position),
            (self.height, self.width, 3)
        ).copy()

        return origins, world_dirs

    def _generate_ortho_rays(self, camera: OrthographicCamera) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ray origins and directions for orthographic camera."""
        # Pixel coordinates
        u = (np.arange(self.width) + 0.5) / self.width
        v = (np.arange(self.height) + 0.5) / self.height
        u, v = np.meshgrid(u, v)

        # Map to camera view space
        x = camera.left + u * (camera.right - camera.left)
        y = camera.top - v * (camera.top - camera.bottom)  # Flip Y

        # All rays point in camera direction
        view_matrix = camera.get_view_matrix()
        view_inv = np.linalg.inv(view_matrix)

        # Camera forward is -Z in view space
        forward = -view_inv[:3, 2]

        directions = np.broadcast_to(forward, (self.height, self.width, 3)).copy()

        # Origins at each pixel position
        origins = np.zeros((self.height, self.width, 3))
        for i in range(self.height):
            for j in range(self.width):
                # View space origin
                view_origin = np.array([x[i, j], y[i, j], 0, 1])
                world_origin = view_inv @ view_origin
                origins[i, j] = world_origin[:3]

        return origins, directions

    # =========================================================================
    # Scene Preparation
    # =========================================================================

    def _prepare_scene(self, scene: HouseScene) -> List[RenderableObject]:
        """Convert scene objects to optimized renderables."""
        renderables = []

        for obj in scene.objects:
            mat = get_material(obj.material)
            size = obj.geometry_params.get('size', (1, 1, 1))

            # Build rotation matrix
            rot = obj.transform.rotation
            rx, ry, rz = [math.radians(a) for a in rot]

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
            R = Rz @ Ry @ Rx

            is_plane = obj.geometry_type == 'plane'

            renderable = RenderableObject(
                name=obj.name,
                obj_type=self._get_object_type(obj),
                center=np.array(obj.transform.position),
                half_size=np.array(size) * np.array(obj.transform.scale) / 2,
                rotation_matrix=R,
                inv_rotation=R.T,
                material=mat,
                is_plane=is_plane,
                plane_normal=np.array([0, 1, 0]) if is_plane else None,
            )
            renderables.append(renderable)

        return renderables

    def _get_object_type(self, obj: SceneObject) -> str:
        """Get object type for segmentation."""
        for tag in obj.tags:
            if tag in self._seg_colors:
                return tag
        return 'default'

    def _prepare_lights(self, scene: HouseScene) -> List[LightSource]:
        """Convert scene lights to light sources."""
        lights = []

        for light in scene.lights:
            # Parse color from hex or tuple
            if isinstance(light.color, str):
                hex_color = light.color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            else:
                rgb = light.color

            ls = LightSource(
                position=np.array(light.position),
                color=np.array(rgb) / 255.0,
                intensity=light.intensity,
                light_type=light.light_type,
                direction=np.array(light.direction) if light.direction else None,
                range=light.range,
                angle=light.angle,
            )
            lights.append(ls)

        return lights

    # =========================================================================
    # Ray Tracing
    # =========================================================================

    def _ray_box_intersect(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        box_center: np.ndarray,
        box_half_size: np.ndarray,
        inv_rotation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ray-box intersection test.

        Returns:
            (hit_mask, t_values) - boolean mask and intersection distances
        """
        # Transform ray to box's local space
        local_origins = np.einsum('ij,hwj->hwi', inv_rotation, origins - box_center)
        local_dirs = np.einsum('ij,hwj->hwi', inv_rotation, directions)

        # Avoid division by zero
        inv_dir = np.where(np.abs(local_dirs) > 1e-10, 1.0 / local_dirs, np.sign(local_dirs) * 1e10)

        # Slab intersection
        t1 = (-box_half_size - local_origins) * inv_dir
        t2 = (box_half_size - local_origins) * inv_dir

        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)

        t_enter = np.max(t_min, axis=-1)
        t_exit = np.min(t_max, axis=-1)

        hit = (t_enter < t_exit) & (t_exit > 0)
        t = np.where(t_enter > 0, t_enter, t_exit)

        return hit, t

    def _ray_plane_intersect(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        plane_center: np.ndarray,
        plane_normal: np.ndarray,
        plane_size: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ray-plane intersection test."""
        # Plane equation: dot(p - center, normal) = 0
        denom = np.sum(directions * plane_normal, axis=-1)

        # Avoid division by zero
        valid = np.abs(denom) > 1e-10
        t = np.where(
            valid,
            np.sum((plane_center - origins) * plane_normal, axis=-1) / np.where(valid, denom, 1),
            np.inf
        )

        # Check if hit point is within plane bounds
        hit_points = origins + directions * t[..., np.newaxis]
        relative = hit_points - plane_center

        in_bounds = (
            (np.abs(relative[..., 0]) <= plane_size[0]) &
            (np.abs(relative[..., 2]) <= plane_size[2])
        )

        hit = valid & (t > 0) & in_bounds

        return hit, t

    def _trace_scene(
        self,
        renderables: List[RenderableObject],
        origins: np.ndarray,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trace rays against all objects.

        Returns:
            (hit_mask, t_values, object_indices)
        """
        h, w = origins.shape[:2]
        best_t = np.full((h, w), np.inf)
        best_obj = np.full((h, w), -1, dtype=np.int32)
        any_hit = np.zeros((h, w), dtype=bool)

        for i, obj in enumerate(renderables):
            if obj.is_plane:
                hit, t = self._ray_plane_intersect(
                    origins, directions,
                    obj.center, obj.plane_normal, obj.half_size
                )
            else:
                hit, t = self._ray_box_intersect(
                    origins, directions,
                    obj.center, obj.half_size, obj.inv_rotation
                )

            closer = hit & (t < best_t) & (t > 0.001)
            best_t = np.where(closer, t, best_t)
            best_obj = np.where(closer, i, best_obj)
            any_hit = any_hit | closer

        return any_hit, best_t, best_obj

    # =========================================================================
    # Rendering
    # =========================================================================

    def _render_rgb(
        self,
        renderables: List[RenderableObject],
        lights: List[LightSource],
        origins: np.ndarray,
        directions: np.ndarray,
        scene: HouseScene,
    ) -> np.ndarray:
        """Render RGB image with lighting."""
        h, w = origins.shape[:2]
        frame = np.zeros((h, w, 3), dtype=np.float32)

        # Background color
        bg_color = np.array(scene.background_color) / 255.0
        frame[:] = bg_color

        # Trace rays
        hit, t, obj_idx = self._trace_scene(renderables, origins, directions)

        # Calculate hit points
        hit_points = origins + directions * t[..., np.newaxis]

        # Process each object
        for i, obj in enumerate(renderables):
            mask = (obj_idx == i)
            if not np.any(mask):
                continue

            # Get material color at hit points
            color = self._shade_object(
                obj, mask, hit_points, directions, lights, scene
            )
            frame[mask] = color[mask]

        # Clamp and convert to uint8
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        return frame

    def _shade_object(
        self,
        obj: RenderableObject,
        mask: np.ndarray,
        hit_points: np.ndarray,
        view_dirs: np.ndarray,
        lights: List[LightSource],
        scene: HouseScene,
    ) -> np.ndarray:
        """Calculate shading for object pixels."""
        h, w = mask.shape
        color = np.zeros((h, w, 3), dtype=np.float32)

        # Get surface normals
        normals = self._get_normals(obj, hit_points, mask)

        # Base color
        base_color = np.array(obj.material.to_rgb_float())

        # Sample procedural texture if enabled
        if obj.material.has_procedural_texture:
            uv = self._calculate_uv(obj, hit_points, mask)
            for i in range(h):
                for j in range(w):
                    if mask[i, j]:
                        tex_color = obj.material.sample_color(uv[i, j, 0], uv[i, j, 1])
                        color[i, j] = np.array(tex_color) / 255.0
        else:
            color[mask] = base_color

        # Lighting
        lit_color = np.zeros_like(color)

        for light in lights:
            if light.light_type == 'ambient':
                lit_color += color * light.color * light.intensity
            elif light.light_type == 'directional':
                if light.direction is not None:
                    light_dir = -light.direction / np.linalg.norm(light.direction)
                    diffuse = np.maximum(0, np.sum(normals * light_dir, axis=-1))
                    lit_color += color * light.color * light.intensity * diffuse[..., np.newaxis]
            elif light.light_type == 'point':
                light_vec = light.position - hit_points
                light_dist = np.linalg.norm(light_vec, axis=-1, keepdims=True)
                light_dir = light_vec / (light_dist + 1e-10)

                # Attenuation
                atten = 1.0 / (1.0 + (light_dist[..., 0] / light.range) ** 2)

                # Diffuse
                diffuse = np.maximum(0, np.sum(normals * light_dir, axis=-1))

                lit_color += color * light.color * light.intensity * diffuse[..., np.newaxis] * atten[..., np.newaxis]

        return np.clip(lit_color, 0, 1)

    def _get_normals(
        self,
        obj: RenderableObject,
        hit_points: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Calculate surface normals at hit points."""
        h, w = mask.shape
        normals = np.zeros((h, w, 3))

        if obj.is_plane:
            normals[mask] = obj.plane_normal
        else:
            # Box normals - determine which face was hit
            local_points = np.einsum('ij,hwj->hwi', obj.inv_rotation, hit_points - obj.center)

            # Find dominant axis
            abs_local = np.abs(local_points / (obj.half_size + 1e-10))
            dominant = np.argmax(abs_local, axis=-1)

            for i in range(h):
                for j in range(w):
                    if mask[i, j]:
                        axis = dominant[i, j]
                        sign = np.sign(local_points[i, j, axis])
                        local_normal = np.zeros(3)
                        local_normal[axis] = sign
                        # Transform back to world space
                        normals[i, j] = obj.rotation_matrix @ local_normal

        return normals

    def _calculate_uv(
        self,
        obj: RenderableObject,
        hit_points: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Calculate UV coordinates for texture mapping."""
        h, w = mask.shape
        uv = np.zeros((h, w, 2))

        # Simple planar projection
        local_points = np.einsum('ij,hwj->hwi', obj.inv_rotation, hit_points - obj.center)

        # Map to 0-1 based on object size
        uv[..., 0] = (local_points[..., 0] / (obj.half_size[0] * 2 + 1e-10) + 0.5)
        uv[..., 1] = (local_points[..., 2] / (obj.half_size[2] * 2 + 1e-10) + 0.5)

        return uv

    def _render_depth(
        self,
        renderables: List[RenderableObject],
        origins: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        """Render depth map."""
        hit, t, _ = self._trace_scene(renderables, origins, directions)
        depth = np.where(hit, t, np.inf)
        return depth.astype(np.float32)

    def _render_segmentation(
        self,
        renderables: List[RenderableObject],
        origins: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        """Render semantic segmentation."""
        h, w = origins.shape[:2]
        seg = np.zeros((h, w, 3), dtype=np.uint8)

        hit, t, obj_idx = self._trace_scene(renderables, origins, directions)

        for i, obj in enumerate(renderables):
            mask = (obj_idx == i)
            if np.any(mask):
                color = self._seg_colors.get(obj.obj_type, self._seg_colors['default'])
                seg[mask] = color

        return seg

    def _render_normals(
        self,
        renderables: List[RenderableObject],
        origins: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        """Render normal map."""
        h, w = origins.shape[:2]
        normals = np.zeros((h, w, 3), dtype=np.float32)

        hit, t, obj_idx = self._trace_scene(renderables, origins, directions)
        hit_points = origins + directions * t[..., np.newaxis]

        for i, obj in enumerate(renderables):
            mask = (obj_idx == i)
            if np.any(mask):
                obj_normals = self._get_normals(obj, hit_points, mask)
                normals[mask] = obj_normals[mask]

        # Map from [-1, 1] to [0, 1]
        return (normals + 1) / 2

    def _add_robot_marker(
        self,
        frame: np.ndarray,
        robot_position: Tuple[float, float, float],
        floor_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        size: int,
    ) -> np.ndarray:
        """Add robot marker to overhead view."""
        min_x, min_z = floor_bounds[0]
        max_x, max_z = floor_bounds[1]

        # Map robot position to pixel coordinates
        px = int((robot_position[0] - min_x) / (max_x - min_x) * size)
        py = int((robot_position[2] - min_z) / (max_z - min_z) * size)

        # Clamp to image bounds
        px = max(5, min(size - 5, px))
        py = max(5, min(size - 5, py))

        # Draw marker (simple circle)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if dx * dx + dy * dy <= 16:
                    y, x = py + dy, px + dx
                    if 0 <= y < size and 0 <= x < size:
                        frame[y, x] = [255, 100, 100]  # Red marker

        return frame

    # =========================================================================
    # Demo
    # =========================================================================

    def demo(self, output_path: str = 'house_render_demo.png'):
        """Generate and save a demo render."""
        from .scene import HouseScene

        # Create a scene
        scene = HouseScene.from_template('studio_apartment')

        # Render from robot's POV
        robot_pos = (3.0, 0.0, 3.0)
        robot_yaw = 45

        frame = self.render_pov(scene, robot_pos, robot_yaw)

        # Save
        try:
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(output_path)
            print(f"Demo render saved to {output_path}")
        except ImportError:
            # Save as raw numpy
            np.save(output_path.replace('.png', '.npy'), frame)
            print(f"Demo render saved to {output_path.replace('.png', '.npy')} (PIL not available)")

        return frame
