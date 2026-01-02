"""
Safety Bounds - Workspace and Motion Limits

Defines physical boundaries that the Ring 0 safety kernel enforces.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceBounds:
    """
    Defines the safe workspace boundaries.
    
    The safety kernel will block any motion that would exit these bounds.
    """
    # Spherical bounds (default)
    radius: float = 0.8  # meters
    center: Tuple[float, float, float] = (0.0, 0.0, 0.3)  # base frame
    
    # Box bounds (optional, in addition to sphere)
    box_enabled: bool = False
    box_min: Tuple[float, float, float] = (-0.5, -0.5, 0.0)
    box_max: Tuple[float, float, float] = (0.5, 0.5, 0.8)
    
    # Floor limit
    floor_z: float = 0.0  # Never go below floor
    
    # Forbidden zones (always blocked)
    forbidden_zones: List[Dict[str, Any]] = field(default_factory=list)
    
    def is_point_safe(self, point: Tuple[float, float, float]) -> bool:
        """Check if a point is within safe workspace."""
        x, y, z = point
        
        # Check floor
        if z < self.floor_z:
            return False
        
        # Check sphere
        dx = x - self.center[0]
        dy = y - self.center[1]
        dz = z - self.center[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > self.radius:
            return False
        
        # Check box (if enabled)
        if self.box_enabled:
            if not (self.box_min[0] <= x <= self.box_max[0] and
                    self.box_min[1] <= y <= self.box_max[1] and
                    self.box_min[2] <= z <= self.box_max[2]):
                return False
        
        # Check forbidden zones
        for zone in self.forbidden_zones:
            if self._in_zone(point, zone):
                return False
        
        return True
    
    def _in_zone(self, point: Tuple[float, float, float], zone: Dict[str, Any]) -> bool:
        """Check if point is in a forbidden zone."""
        zone_type = zone.get('type', 'box')
        
        if zone_type == 'box':
            bounds = zone.get('bounds', [[-1, -1, -1], [1, 1, 1]])
            return (bounds[0][0] <= point[0] <= bounds[1][0] and
                    bounds[0][1] <= point[1] <= bounds[1][1] and
                    bounds[0][2] <= point[2] <= bounds[1][2])
        
        elif zone_type == 'sphere':
            center = zone.get('center', [0, 0, 0])
            radius = zone.get('radius', 0.1)
            dist = math.sqrt(sum((p - c)**2 for p, c in zip(point, center)))
            return dist < radius
        
        elif zone_type == 'cylinder':
            center = zone.get('center', [0, 0, 0])
            radius = zone.get('radius', 0.1)
            height = zone.get('height', 0.2)
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            dz = point[2] - center[2]
            return (dx*dx + dy*dy < radius*radius and
                    -height/2 < dz < height/2)
        
        return False
    
    def get_safe_clamp(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Clamp a point to the nearest safe position."""
        x, y, z = point
        
        # Clamp to floor
        z = max(z, self.floor_z)
        
        # Clamp to sphere
        dx = x - self.center[0]
        dy = y - self.center[1]
        dz = z - self.center[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist > self.radius:
            scale = self.radius / dist
            x = self.center[0] + dx * scale
            y = self.center[1] + dy * scale
            z = self.center[2] + dz * scale
        
        return (x, y, z)


@dataclass
class JointBounds:
    """Joint position, velocity, and torque limits."""
    
    # Position limits (radians)
    position_min: List[float] = field(default_factory=lambda: [-3.14, -1.57, -3.14, -3.14, -3.14, -3.14, -3.14])
    position_max: List[float] = field(default_factory=lambda: [3.14, 1.57, 3.14, 3.14, 3.14, 3.14, 3.14])
    
    # Velocity limits (rad/s)
    velocity_max: List[float] = field(default_factory=lambda: [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 3.0])
    
    # Torque limits (Nm)
    torque_max: List[float] = field(default_factory=lambda: [5.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.5])
    
    # Acceleration limits (rad/s²)
    acceleration_max: List[float] = field(default_factory=lambda: [10.0, 10.0, 10.0, 15.0, 15.0, 20.0, 20.0])
    
    def check_position(self, positions: List[float]) -> Tuple[bool, Optional[str]]:
        """Check if positions are within limits."""
        for i, pos in enumerate(positions):
            if pos < self.position_min[i]:
                return False, f"Joint {i} below min: {pos:.3f} < {self.position_min[i]:.3f}"
            if pos > self.position_max[i]:
                return False, f"Joint {i} above max: {pos:.3f} > {self.position_max[i]:.3f}"
        return True, None
    
    def check_velocity(self, velocities: List[float]) -> Tuple[bool, Optional[str]]:
        """Check if velocities are within limits."""
        for i, vel in enumerate(velocities):
            if abs(vel) > self.velocity_max[i]:
                return False, f"Joint {i} velocity exceeded: {abs(vel):.3f} > {self.velocity_max[i]:.3f}"
        return True, None
    
    def check_torque(self, torques: List[float]) -> Tuple[bool, Optional[str]]:
        """Check if torques are within limits."""
        for i, torque in enumerate(torques):
            if abs(torque) > self.torque_max[i]:
                return False, f"Joint {i} torque exceeded: {abs(torque):.3f} > {self.torque_max[i]:.3f}"
        return True, None
    
    def clamp_velocity(self, velocities: List[float]) -> List[float]:
        """Clamp velocities to safe limits."""
        return [
            max(-self.velocity_max[i], min(self.velocity_max[i], v))
            for i, v in enumerate(velocities)
        ]


@dataclass 
class SafetyBounds:
    """Complete safety bounds configuration."""
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    joints: JointBounds = field(default_factory=JointBounds)
    
    # End-effector velocity
    max_ee_velocity: float = 1.0  # m/s
    max_ee_velocity_human: float = 0.25  # m/s when human present
    
    # Force limits
    max_contact_force: float = 50.0  # N
    max_contact_force_human: float = 10.0  # N
    
    def validate_action(self, action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate an action against all safety bounds.
        
        Returns:
            (is_safe, error_message)
        """
        action_type = action.get('type', '')
        
        if action_type == 'joint_position':
            positions = action.get('positions', [])
            return self.joints.check_position(positions)
        
        elif action_type == 'joint_velocity':
            velocities = action.get('velocities', [])
            return self.joints.check_velocity(velocities)
        
        elif action_type == 'cartesian_position':
            position = action.get('position', [0, 0, 0])
            if not self.workspace.is_point_safe(tuple(position)):
                return False, f"Position outside workspace: {position}"
            return True, None
        
        elif action_type == 'trajectory':
            waypoints = action.get('waypoints', [])
            for i, wp in enumerate(waypoints):
                if not self.workspace.is_point_safe(tuple(wp[:3])):
                    return False, f"Waypoint {i} outside workspace: {wp[:3]}"
            return True, None
        
        # Unknown action type - allow by default (other validators may block)
        return True, None


# Default bounds instance
DEFAULT_BOUNDS = SafetyBounds()

