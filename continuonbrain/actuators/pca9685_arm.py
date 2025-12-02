"""
PCA9685 servo controller for SO-ARM101 robot arm.
Provides bounded control with safety limits per PI5_CAR_READINESS.md.
"""
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import time

# Optional hardware imports (guarded for development)
try:
    from adafruit_servokit import ServoKit
    HAS_SERVOKIT = True
except ImportError:
    HAS_SERVOKIT = False
    print("Warning: adafruit-servokit not installed (mock mode)")


@dataclass
class ServoLimits:
    """Safety limits for a single servo joint."""
    min_angle: float  # degrees
    max_angle: float  # degrees
    default_angle: float  # safe default position
    max_speed: float = 60.0  # degrees per second


class ArmConfig:
    """SO-ARM101 arm configuration with 6 DOF."""
    def __init__(self):
        # Channel mapping on PCA9685
        self.base_channel: int = 0
        self.shoulder_channel: int = 1
        self.elbow_channel: int = 2
        self.wrist_pitch_channel: int = 3
        self.wrist_roll_channel: int = 4
        self.gripper_channel: int = 5
        
        # Safety limits per joint (degrees)
        self.base_limits = ServoLimits(0, 180, 90)
        self.shoulder_limits = ServoLimits(0, 180, 90)
        self.elbow_limits = ServoLimits(0, 180, 90)
        self.wrist_pitch_limits = ServoLimits(0, 180, 90)
        self.wrist_roll_limits = ServoLimits(0, 180, 90)
        self.gripper_limits = ServoLimits(30, 90, 30)  # 30=open, 90=closed


class PCA9685ArmController:
    """
    Controls SO-ARM101 robot arm via PCA9685 with safety bounds.
    Maps normalized actions [-1, 1] to servo angles.
    """
    
    def __init__(self, config: Optional[ArmConfig] = None, i2c_address: int = 0x40):
        self.config = config or ArmConfig()
        self.i2c_address = i2c_address
        self.kit: Optional[ServoKit] = None
        self.current_positions: Dict[str, float] = {}
        self.last_update_time: float = 0.0
        self.is_mock = not HAS_SERVOKIT
        
    def initialize(self) -> bool:
        """Initialize PCA9685 servo controller."""
        try:
            if self.is_mock:
                print(f"MOCK: Initializing PCA9685 at 0x{self.i2c_address:02x}")
                # Initialize mock positions
                self.current_positions = {
                    "base": self.config.base_limits.default_angle,
                    "shoulder": self.config.shoulder_limits.default_angle,
                    "elbow": self.config.elbow_limits.default_angle,
                    "wrist_pitch": self.config.wrist_pitch_limits.default_angle,
                    "wrist_roll": self.config.wrist_roll_limits.default_angle,
                    "gripper": self.config.gripper_limits.default_angle,
                }
                return True
            
            # Real hardware initialization
            self.kit = ServoKit(channels=16, address=self.i2c_address)
            
            # Move to safe default positions
            self._move_to_defaults()
            
            print(f"✅ PCA9685 initialized at 0x{self.i2c_address:02x}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize PCA9685: {e}")
            return False
    
    def _move_to_defaults(self):
        """Move all servos to safe default positions."""
        joints = [
            ("base", self.config.base_channel, self.config.base_limits),
            ("shoulder", self.config.shoulder_channel, self.config.shoulder_limits),
            ("elbow", self.config.elbow_channel, self.config.elbow_limits),
            ("wrist_pitch", self.config.wrist_pitch_channel, self.config.wrist_pitch_limits),
            ("wrist_roll", self.config.wrist_roll_channel, self.config.wrist_roll_limits),
            ("gripper", self.config.gripper_channel, self.config.gripper_limits),
        ]
        
        for name, channel, limits in joints:
            self._set_servo_angle(channel, limits.default_angle)
            self.current_positions[name] = limits.default_angle
        
        print("Arm moved to default safe positions")
    
    def _set_servo_angle(self, channel: int, angle: float):
        """Set servo angle with hardware abstraction."""
        if self.is_mock:
            # Mock mode - just track position
            pass
        else:
            self.kit.servo[channel].angle = angle
    
    def _clamp_angle(self, angle: float, limits: ServoLimits) -> float:
        """Clamp angle to safety limits."""
        return max(limits.min_angle, min(limits.max_angle, angle))
    
    def set_joint_angles(self, joint_angles: Dict[str, float]) -> bool:
        """
        Set joint angles in degrees with safety clamping.
        
        Args:
            joint_angles: Dict mapping joint names to angles (degrees)
                         e.g., {"base": 90, "shoulder": 45, "gripper": 60}
        
        Returns:
            True if successful
        """
        try:
            limits_map = {
                "base": (self.config.base_channel, self.config.base_limits),
                "shoulder": (self.config.shoulder_channel, self.config.shoulder_limits),
                "elbow": (self.config.elbow_channel, self.config.elbow_limits),
                "wrist_pitch": (self.config.wrist_pitch_channel, self.config.wrist_pitch_limits),
                "wrist_roll": (self.config.wrist_roll_channel, self.config.wrist_roll_limits),
                "gripper": (self.config.gripper_channel, self.config.gripper_limits),
            }
            
            for joint_name, angle in joint_angles.items():
                if joint_name not in limits_map:
                    print(f"Warning: Unknown joint {joint_name}")
                    continue
                
                channel, limits = limits_map[joint_name]
                clamped_angle = self._clamp_angle(angle, limits)
                
                if clamped_angle != angle:
                    print(f"Warning: {joint_name} angle {angle}° clamped to {clamped_angle}°")
                
                self._set_servo_angle(channel, clamped_angle)
                self.current_positions[joint_name] = clamped_angle
            
            self.last_update_time = time.time()
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to set joint angles: {e}")
            return False
    
    def set_normalized_action(self, action: List[float]) -> bool:
        """
        Set joint positions from normalized action vector [-1, 1].
        Maps to full range of each joint's limits.
        
        Args:
            action: List of 6 floats in [-1, 1] for [base, shoulder, elbow, 
                   wrist_pitch, wrist_roll, gripper]
        
        Returns:
            True if successful
        """
        if len(action) != 6:
            print(f"ERROR: Action must have 6 values, got {len(action)}")
            return False
        
        # Map normalized [-1, 1] to joint angles
        def map_to_range(normalized: float, limits: ServoLimits) -> float:
            # Clamp input to [-1, 1]
            normalized = max(-1.0, min(1.0, normalized))
            # Map to [min_angle, max_angle]
            range_size = limits.max_angle - limits.min_angle
            return limits.min_angle + (normalized + 1.0) * range_size / 2.0
        
        joint_angles = {
            "base": map_to_range(action[0], self.config.base_limits),
            "shoulder": map_to_range(action[1], self.config.shoulder_limits),
            "elbow": map_to_range(action[2], self.config.elbow_limits),
            "wrist_pitch": map_to_range(action[3], self.config.wrist_pitch_limits),
            "wrist_roll": map_to_range(action[4], self.config.wrist_roll_limits),
            "gripper": map_to_range(action[5], self.config.gripper_limits),
        }
        
        return self.set_joint_angles(joint_angles)
    
    def get_current_state(self) -> Dict[str, float]:
        """
        Get current joint positions for robot_state in RLDS.
        
        Returns:
            Dict mapping joint names to current angles (degrees)
        """
        return self.current_positions.copy()
    
    def get_normalized_state(self) -> List[float]:
        """
        Get current state as normalized vector [-1, 1] for RLDS observation.
        
        Returns:
            List of 6 floats representing normalized joint positions
        """
        def normalize(angle: float, limits: ServoLimits) -> float:
            range_size = limits.max_angle - limits.min_angle
            return 2.0 * (angle - limits.min_angle) / range_size - 1.0
        
        return [
            normalize(self.current_positions["base"], self.config.base_limits),
            normalize(self.current_positions["shoulder"], self.config.shoulder_limits),
            normalize(self.current_positions["elbow"], self.config.elbow_limits),
            normalize(self.current_positions["wrist_pitch"], self.config.wrist_pitch_limits),
            normalize(self.current_positions["wrist_roll"], self.config.wrist_roll_limits),
            normalize(self.current_positions["gripper"], self.config.gripper_limits),
        ]
    
    def emergency_stop(self):
        """Emergency stop - hold current positions."""
        print("⚠️  EMERGENCY STOP - Holding positions")
        # Servos will hold last commanded position
        # Could also move to safe default if needed
    
    def shutdown(self):
        """Graceful shutdown - return to defaults."""
        print("Shutting down arm controller...")
        self._move_to_defaults()
        time.sleep(0.5)  # Allow servos to reach position
        print("✅ Arm controller shutdown complete")


def test_arm_controller():
    """Test arm controller for development."""
    print("Testing PCA9685 Arm Controller...")
    
    arm = PCA9685ArmController()
    
    if not arm.initialize():
        print("Failed to initialize")
        return
    
    print(f"\nInitial state: {arm.get_current_state()}")
    print(f"Normalized: {arm.get_normalized_state()}")
    
    # Test normalized actions
    test_actions = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Center positions
        [0.5, 0.5, 0.5, 0.0, 0.0, -0.5],  # Mixed positions
        [-0.2, 0.3, -0.1, 0.4, -0.3, 0.8],  # Random positions
    ]
    
    for i, action in enumerate(test_actions, 1):
        print(f"\nTest action {i}: {action}")
        arm.set_normalized_action(action)
        print(f"  Result state: {arm.get_current_state()}")
        print(f"  Normalized: {[f'{x:.2f}' for x in arm.get_normalized_state()]}")
        time.sleep(1.0)
    
    # Return to defaults
    arm.shutdown()
    print("\n✅ Arm controller test complete")


if __name__ == "__main__":
    test_arm_controller()
