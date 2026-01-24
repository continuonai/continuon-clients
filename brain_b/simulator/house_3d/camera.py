"""
Camera and Projection System for House 3D Renderer

Provides perspective and orthographic cameras for robot POV
and overhead map views.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Camera:
    """
    Base camera class.

    Position is in world coordinates.
    Rotation uses pitch (up/down), yaw (left/right), roll (tilt).
    """
    position: Tuple[float, float, float] = (0, 0, 0)  # (x, y, z)
    pitch: float = 0.0    # Degrees, up/down (-90 to 90)
    yaw: float = 0.0      # Degrees, left/right (0 = +Z, 90 = +X)
    roll: float = 0.0     # Degrees, tilt

    near: float = 0.1     # Near clipping plane
    far: float = 100.0    # Far clipping plane

    def get_view_matrix(self) -> np.ndarray:
        """
        Get the view matrix (world -> camera space).

        Returns:
            4x4 view matrix
        """
        # Build rotation matrix from Euler angles
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)
        roll_rad = math.radians(self.roll)

        # Rotation matrices
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)

        # Pitch (X-axis rotation)
        Rx = np.array([
            [1, 0, 0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ])

        # Yaw (Y-axis rotation)
        Ry = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])

        # Roll (Z-axis rotation)
        Rz = np.array([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ])

        # Combined rotation: Rz * Rx * Ry (standard FPS order)
        R = Rz @ Rx @ Ry

        # Translation (move world opposite to camera position)
        t = np.array(self.position)

        # Build 4x4 view matrix
        view = np.eye(4)
        view[:3, :3] = R.T  # Transpose for inverse rotation
        view[:3, 3] = -R.T @ t  # Apply inverse translation

        return view

    def get_forward_vector(self) -> Tuple[float, float, float]:
        """Get unit vector pointing where camera is looking."""
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)

        x = math.cos(pitch_rad) * math.sin(yaw_rad)
        y = math.sin(pitch_rad)
        z = math.cos(pitch_rad) * math.cos(yaw_rad)

        return (x, y, z)

    def get_right_vector(self) -> Tuple[float, float, float]:
        """Get unit vector pointing to camera's right."""
        yaw_rad = math.radians(self.yaw)
        return (math.cos(yaw_rad), 0, -math.sin(yaw_rad))

    def get_up_vector(self) -> Tuple[float, float, float]:
        """Get unit vector pointing up from camera."""
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)

        x = -math.sin(pitch_rad) * math.sin(yaw_rad)
        y = math.cos(pitch_rad)
        z = -math.sin(pitch_rad) * math.cos(yaw_rad)

        return (x, y, z)

    def to_dict(self) -> dict:
        """Export for Three.js."""
        return {
            'position': list(self.position),
            'pitch': self.pitch,
            'yaw': self.yaw,
            'roll': self.roll,
            'near': self.near,
            'far': self.far,
        }


@dataclass
class PerspectiveCamera(Camera):
    """
    Perspective projection camera.

    Uses standard pinhole camera model with field-of-view.
    """
    fov: float = 60.0     # Vertical field of view in degrees
    aspect: float = 1.333  # Width / height (4:3 = 1.333, 16:9 = 1.778)

    def get_projection_matrix(self) -> np.ndarray:
        """
        Get the perspective projection matrix.

        Returns:
            4x4 projection matrix
        """
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2)

        # Standard perspective projection matrix
        proj = np.zeros((4, 4))
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1

        return proj

    def project_point(self, world_point: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """
        Project a 3D world point to 2D screen coordinates.

        Args:
            world_point: (x, y, z) in world space

        Returns:
            (screen_x, screen_y, depth) where screen coords are -1 to 1,
            or None if point is behind camera
        """
        view = self.get_view_matrix()
        proj = self.get_projection_matrix()

        # World to camera space
        p = np.array([*world_point, 1.0])
        camera_p = view @ p

        # Behind camera?
        if camera_p[2] >= -self.near:
            return None

        # Camera to clip space
        clip_p = proj @ camera_p

        # Perspective divide
        if abs(clip_p[3]) < 1e-10:
            return None

        ndc_x = clip_p[0] / clip_p[3]
        ndc_y = clip_p[1] / clip_p[3]
        depth = -camera_p[2]  # Positive depth

        return (ndc_x, ndc_y, depth)

    def screen_to_ray(
        self,
        screen_x: float,
        screen_y: float,
        width: int,
        height: int
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Convert screen coordinates to a world ray.

        Args:
            screen_x, screen_y: Pixel coordinates
            width, height: Screen dimensions

        Returns:
            (origin, direction) tuple for ray
        """
        # Screen to NDC (-1 to 1)
        ndc_x = (2.0 * screen_x / width) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / height)

        # NDC to view space ray direction
        fov_rad = math.radians(self.fov)
        tan_half_fov = math.tan(fov_rad / 2)

        view_x = ndc_x * tan_half_fov * self.aspect
        view_y = ndc_y * tan_half_fov
        view_z = -1.0

        # Transform to world space
        view_inv = np.linalg.inv(self.get_view_matrix())

        origin = np.array(self.position)

        dir_view = np.array([view_x, view_y, view_z, 0.0])
        dir_world = (view_inv @ dir_view)[:3]

        # Normalize
        length = np.linalg.norm(dir_world)
        if length > 0:
            dir_world = dir_world / length

        return (tuple(origin), tuple(dir_world))

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'type': 'perspective',
            'fov': self.fov,
            'aspect': self.aspect,
        })
        return base


@dataclass
class OrthographicCamera(Camera):
    """
    Orthographic projection camera.

    No perspective distortion - useful for overhead/map views.
    """
    left: float = -5.0
    right: float = 5.0
    bottom: float = -5.0
    top: float = 5.0

    def get_projection_matrix(self) -> np.ndarray:
        """
        Get the orthographic projection matrix.

        Returns:
            4x4 projection matrix
        """
        proj = np.zeros((4, 4))

        proj[0, 0] = 2.0 / (self.right - self.left)
        proj[1, 1] = 2.0 / (self.top - self.bottom)
        proj[2, 2] = -2.0 / (self.far - self.near)

        proj[0, 3] = -(self.right + self.left) / (self.right - self.left)
        proj[1, 3] = -(self.top + self.bottom) / (self.top - self.bottom)
        proj[2, 3] = -(self.far + self.near) / (self.far - self.near)
        proj[3, 3] = 1.0

        return proj

    def project_point(self, world_point: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """
        Project a 3D world point to 2D screen coordinates.

        Returns:
            (screen_x, screen_y, depth) where screen coords are -1 to 1
        """
        view = self.get_view_matrix()

        # World to camera space
        p = np.array([*world_point, 1.0])
        camera_p = view @ p

        # Orthographic projection
        ndc_x = (camera_p[0] - self.left) / (self.right - self.left) * 2 - 1
        ndc_y = (camera_p[1] - self.bottom) / (self.top - self.bottom) * 2 - 1
        depth = -camera_p[2]

        return (ndc_x, ndc_y, depth)

    @classmethod
    def from_bounds(cls, bounds_min: Tuple[float, float], bounds_max: Tuple[float, float], height: float = 10.0) -> 'OrthographicCamera':
        """
        Create orthographic camera from world bounds (for overhead view).

        Args:
            bounds_min: (x_min, z_min) floor bounds
            bounds_max: (x_max, z_max) floor bounds
            height: Camera height above floor
        """
        center_x = (bounds_min[0] + bounds_max[0]) / 2
        center_z = (bounds_min[1] + bounds_max[1]) / 2

        half_width = (bounds_max[0] - bounds_min[0]) / 2 * 1.1  # 10% margin
        half_depth = (bounds_max[1] - bounds_min[1]) / 2 * 1.1

        return cls(
            position=(center_x, height, center_z),
            pitch=-90,  # Looking straight down
            yaw=0,
            left=-half_width,
            right=half_width,
            bottom=-half_depth,
            top=half_depth,
            near=0.1,
            far=height + 5,
        )

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'type': 'orthographic',
            'left': self.left,
            'right': self.right,
            'bottom': self.bottom,
            'top': self.top,
        })
        return base


class RobotCamera(PerspectiveCamera):
    """
    Camera mounted on robot, follows robot position and heading.

    Integrates with home_world.py Robot3D model.
    """

    def __init__(
        self,
        eye_height: float = 0.8,      # Height of camera on robot
        fov: float = 75.0,             # Wide FOV for robot vision
        aspect: float = 1.333,
    ):
        super().__init__(fov=fov, aspect=aspect)
        self.eye_height = eye_height

    def update_from_robot(self, robot_position: Tuple[float, float, float], robot_yaw: float, robot_pitch: float = 0.0):
        """
        Update camera from robot state.

        Args:
            robot_position: (x, y, z) world position (y is usually 0 on floor)
            robot_yaw: Robot heading in degrees (0 = +Z direction)
            robot_pitch: Head pitch in degrees (looking up/down)
        """
        self.position = (
            robot_position[0],
            robot_position[1] + self.eye_height,  # Add eye height
            robot_position[2]
        )
        self.yaw = robot_yaw
        self.pitch = robot_pitch

    def update_from_home_world(self, robot):
        """
        Update from home_world.py Robot3D object.

        Args:
            robot: Robot3D instance from home_world.py
        """
        # home_world uses (x, y, z) where y is depth (north-south)
        # and z is height - convert to standard 3D (y-up)
        self.position = (
            robot.position.x,
            robot.position.z + self.eye_height,  # z is height
            robot.position.y   # y is depth
        )
        self.yaw = robot.rotation.yaw
        self.pitch = robot.rotation.pitch

    def to_dict(self) -> dict:
        base = super().to_dict()
        base['eyeHeight'] = self.eye_height
        return base


def create_split_view_cameras(
    scene_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    robot_position: Tuple[float, float, float],
    robot_yaw: float,
    pov_width: int = 640,
    pov_height: int = 480,
    overhead_size: int = 256,
) -> dict:
    """
    Create cameras for split-view display (POV + overhead + optional sensors).

    Args:
        scene_bounds: ((min_x, min_z), (max_x, max_z)) floor bounds
        robot_position: Current robot position
        robot_yaw: Robot heading
        pov_width, pov_height: Main POV view dimensions
        overhead_size: Overhead map size (square)

    Returns:
        Dict with 'pov', 'overhead', and camera settings
    """
    # Robot POV camera
    pov_camera = RobotCamera(
        eye_height=0.8,
        fov=75,
        aspect=pov_width / pov_height,
    )
    pov_camera.update_from_robot(robot_position, robot_yaw)

    # Overhead camera
    overhead_camera = OrthographicCamera.from_bounds(
        scene_bounds[0],
        scene_bounds[1],
        height=10.0
    )

    return {
        'pov': pov_camera,
        'overhead': overhead_camera,
        'pov_size': (pov_width, pov_height),
        'overhead_size': (overhead_size, overhead_size),
    }
