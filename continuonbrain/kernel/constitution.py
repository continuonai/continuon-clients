import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("SafetyKernel.Constitution")

class Constitution:
    """
    Deterministic Safety Logic (Ring 0 Rules).
    Ensures commands respect physical and environmental limits.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Default Kinematic Limits
        self.joint_limits = self.config.get("joint_limits", {
            "base": (-180.0, 180.0),
            "shoulder": (-90.0, 90.0),
            "elbow": (-150.0, 150.0),
            "wrist": (-180.0, 180.0),
            "gripper": (0.0, 1.0)
        })
        
        self.max_velocity = self.config.get("max_velocity", 1.0)
        self.max_acceleration = self.config.get("max_acceleration", 2.0)
        
        # Environmental Constraints (Static Obstacles / No-Go Zones)
        # Format: list of polygons or bounding boxes
        self.no_go_zones = self.config.get("no_go_zones", [])
        
        # Power / Thermal Thresholds
        self.min_voltage = self.config.get("min_voltage", 10.0)
        self.max_cpu_temp = self.config.get("max_cpu_temp", 80.0)

    def validate_actuation(self, command: str, args: Dict[str, Any]) -> Tuple[int, Dict[str, Any], str]:
        """
        Validate an actuation command.
        Returns: (safety_level, adjusted_args, reason)
        Safety Levels:
          0: Denied (Violation)
          1: OK (Clipped/Adjusted)
          2: OK (Original)
        """
        
        # 1. Check Kinematic Limits (Joint Angles)
        if command == "move_joints" and "joints" in args:
            joints = args["joints"]
            adjusted_joints = {}
            clipping_occurred = False
            
            for joint_name, value in joints.items():
                if joint_name in self.joint_limits:
                    low, high = self.joint_limits[joint_name]
                    clipped = max(low, min(high, value))
                    if clipped != value:
                        clipping_occurred = True
                        logger.warning(f"Kinematic limit hit: {joint_name} ({value} -> {clipped})")
                    adjusted_joints[joint_name] = clipped
                else:
                    adjusted_joints[joint_name] = value
            
            args["joints"] = adjusted_joints
            if clipping_occurred:
                return 1, args, "joint_limit_clipping"

        # 2. Check Velocity Limits
        if "velocity" in args:
            original_v = args["velocity"]
            clipped_v = max(-self.max_velocity, min(self.max_velocity, original_v))
            if clipped_v != original_v:
                args["velocity"] = clipped_v
                return 1, args, "velocity_clipping"

        # 3. Check Environmental Constraints (Placeholder for IK/Collision logic)
        if "target_pose" in args:
            # Simple bounds check for target_pose (x, y, z)
            pose = args["target_pose"]
            if any(coord > 1000 for coord in pose.values()): # Example limit
                 return 0, {}, "out_of_reach"

        # 4. Power/Thermal Intercepts (Simulated here, would use real hardware sensors)
        # If we had a shared memory state or recent sensor readings
        
        return 2, args, "ok"

    def check_system_health(self, metrics: Dict[str, Any]) -> bool:
        """Verify if system metrics are within safe operating range."""
        voltage = metrics.get("voltage", 12.0)
        temp = metrics.get("cpu_temp", 45.0)
        
        if voltage < self.min_voltage:
            logger.error(f"Low voltage detected: {voltage}V")
            return False
            
        if temp > self.max_cpu_temp:
            logger.error(f"High thermal detected: {temp}C")
            return False
            
        return True
