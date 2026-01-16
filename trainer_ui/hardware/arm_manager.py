"""
Dual arm controller for SO-ARM101 robot arms.

Provides:
- ArmController: Single arm control via PCA9685
- DualArmManager: Manages multiple arms with graceful fallbacks
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Optional hardware imports (guarded for development)
try:
    from adafruit_servokit import ServoKit
    HAS_SERVOKIT = True
except ImportError:
    HAS_SERVOKIT = False


@dataclass
class ServoLimits:
    """Safety limits for a single servo joint."""
    min_angle: float  # degrees
    max_angle: float  # degrees
    default_angle: float  # safe default position
    max_speed: float = 60.0  # degrees per second


@dataclass
class ArmState:
    """Current state of a robot arm."""
    arm_id: str
    joints: List[float] = field(default_factory=lambda: [0.0] * 6)  # Normalized [-1, 1]
    gripper: float = 0.0  # [0, 1]
    connected: bool = False
    is_mock: bool = False
    i2c_address: str = "0x40"


class ArmController:
    """
    Controls a single SO-ARM101 robot arm via PCA9685.
    Maps normalized actions [-1, 1] to servo angles.
    """

    # Channel mapping for SO-ARM101
    JOINT_CHANNELS = [0, 1, 2, 3, 4, 5]  # 6 joints
    GRIPPER_CHANNEL = 5  # Gripper is on channel 5

    # Default safety limits per joint (degrees)
    DEFAULT_LIMITS = [
        ServoLimits(0, 180, 90),  # Base
        ServoLimits(0, 180, 90),  # Shoulder
        ServoLimits(0, 180, 90),  # Elbow
        ServoLimits(0, 180, 90),  # Wrist pitch
        ServoLimits(0, 180, 90),  # Wrist roll
        ServoLimits(30, 90, 30),  # Gripper (30=open, 90=closed)
    ]

    def __init__(self, arm_id: str, i2c_address: int = 0x40, is_mock: bool = False):
        """
        Initialize arm controller.

        Args:
            arm_id: Unique identifier for this arm (e.g., "arm_0", "arm_1")
            i2c_address: I2C address of PCA9685 (0x40 or 0x41)
            is_mock: If True, run in mock mode without hardware
        """
        self.arm_id = arm_id
        self.i2c_address = i2c_address
        self.is_mock = is_mock or not HAS_SERVOKIT
        self.kit: Optional[Any] = None

        # Current state (normalized)
        self.joint_values: List[float] = [0.0] * 6
        self.gripper_value: float = 0.0

        # Current angles (degrees)
        self.joint_angles: List[float] = [90.0] * 5 + [30.0]  # Default positions

        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the arm controller."""
        if self._initialized:
            return True

        try:
            if self.is_mock:
                print(f"MOCK: Initializing arm '{self.arm_id}' at 0x{self.i2c_address:02x}")
                self._initialized = True
                return True

            # Real hardware initialization
            self.kit = ServoKit(channels=16, address=self.i2c_address)
            self._move_to_defaults()

            print(f"Initialized arm '{self.arm_id}' at 0x{self.i2c_address:02x}")
            self._initialized = True
            return True

        except Exception as e:
            print(f"ERROR: Failed to initialize arm '{self.arm_id}': {e}")
            self.is_mock = True  # Fall back to mock
            self._initialized = True
            return True  # Still return True so it can work in mock mode

    def _move_to_defaults(self):
        """Move all servos to safe default positions."""
        for i, limits in enumerate(self.DEFAULT_LIMITS):
            self._set_servo_angle(i, limits.default_angle)
            self.joint_angles[i] = limits.default_angle

        # Update normalized values
        self.joint_values = [0.0] * 6
        self.gripper_value = 0.0

    def _set_servo_angle(self, channel: int, angle: float):
        """Set servo angle with hardware abstraction."""
        if self.is_mock:
            return

        if self.kit is not None and 0 <= channel < 16:
            self.kit.servo[channel].angle = angle

    def _clamp_angle(self, angle: float, limits: ServoLimits) -> float:
        """Clamp angle to safety limits."""
        return max(limits.min_angle, min(limits.max_angle, angle))

    def _normalized_to_angle(self, value: float, limits: ServoLimits) -> float:
        """Convert normalized [-1, 1] to angle in degrees."""
        value = max(-1.0, min(1.0, value))
        range_size = limits.max_angle - limits.min_angle
        return limits.min_angle + (value + 1.0) * range_size / 2.0

    def _angle_to_normalized(self, angle: float, limits: ServoLimits) -> float:
        """Convert angle in degrees to normalized [-1, 1]."""
        range_size = limits.max_angle - limits.min_angle
        if range_size == 0:
            return 0.0
        return 2.0 * (angle - limits.min_angle) / range_size - 1.0

    def set_joint(self, joint: int, value: float) -> bool:
        """
        Set a joint position using normalized value [-1, 1].

        Args:
            joint: Joint index (0-4 for arm joints)
            value: Normalized position [-1, 1]

        Returns:
            True if successful
        """
        if not self._initialized:
            self.initialize()

        if joint < 0 or joint > 4:
            return False

        value = max(-1.0, min(1.0, value))
        limits = self.DEFAULT_LIMITS[joint]
        angle = self._normalized_to_angle(value, limits)
        angle = self._clamp_angle(angle, limits)

        self._set_servo_angle(joint, angle)
        self.joint_values[joint] = value
        self.joint_angles[joint] = angle

        return True

    def set_gripper(self, value: float) -> bool:
        """
        Set gripper position [0, 1] where 0=open, 1=closed.

        Args:
            value: Gripper position [0, 1]

        Returns:
            True if successful
        """
        if not self._initialized:
            self.initialize()

        value = max(0.0, min(1.0, value))
        limits = self.DEFAULT_LIMITS[5]  # Gripper limits

        # Map [0, 1] to gripper range
        angle = limits.min_angle + value * (limits.max_angle - limits.min_angle)

        self._set_servo_angle(self.GRIPPER_CHANNEL, angle)
        self.gripper_value = value
        self.joint_angles[5] = angle

        return True

    def get_state(self) -> ArmState:
        """Get current arm state."""
        return ArmState(
            arm_id=self.arm_id,
            joints=self.joint_values.copy(),
            gripper=self.gripper_value,
            connected=self._initialized,
            is_mock=self.is_mock,
            i2c_address=f"0x{self.i2c_address:02x}",
        )

    def emergency_stop(self):
        """Emergency stop - hold current positions."""
        print(f"EMERGENCY STOP: Arm '{self.arm_id}'")
        # Servos will hold last commanded position

    def shutdown(self):
        """Graceful shutdown - return to defaults."""
        print(f"Shutting down arm '{self.arm_id}'...")
        self._move_to_defaults()
        time.sleep(0.3)


class DualArmManager:
    """
    Manages multiple robot arms with graceful fallbacks.
    Supports 0, 1, or 2 arms depending on detected hardware.
    """

    def __init__(self, hw_config: Optional[Any] = None):
        """
        Initialize dual arm manager.

        Args:
            hw_config: HardwareConfig from detector, or None for auto-detect
        """
        self.arms: Dict[str, ArmController] = {}
        self.hw_config = hw_config

    def initialize(self) -> Dict[str, ArmState]:
        """
        Initialize all detected arms.

        Returns:
            Dict mapping arm_id to ArmState
        """
        if self.hw_config is None:
            # No config provided, create mock arms
            print("No hardware config provided, using mock arms")
            self._create_mock_arms()
        else:
            # Create arms based on detected hardware
            for arm_id, arm_info in self.hw_config.arms.items():
                try:
                    address_str = arm_info.get("i2c_address", "0x40")
                    address = int(address_str, 16) if isinstance(address_str, str) else address_str
                    is_mock = arm_info.get("is_mock", False)

                    arm = ArmController(
                        arm_id=arm_id,
                        i2c_address=address,
                        is_mock=is_mock
                    )

                    if arm.initialize():
                        self.arms[arm_id] = arm
                        print(f"Initialized arm: {arm_id}")

                except Exception as e:
                    print(f"Failed to initialize arm {arm_id}: {e}")

        return self.get_all_states()

    def _create_mock_arms(self):
        """Create mock arms for development."""
        for arm_id, address in [("arm_0", 0x40), ("arm_1", 0x41)]:
            arm = ArmController(arm_id=arm_id, i2c_address=address, is_mock=True)
            arm.initialize()
            self.arms[arm_id] = arm

    def get_arm(self, arm_id: str) -> Optional[ArmController]:
        """Get a specific arm by ID."""
        return self.arms.get(arm_id)

    def get_all_states(self) -> Dict[str, ArmState]:
        """Get states of all arms."""
        return {arm_id: arm.get_state() for arm_id, arm in self.arms.items()}

    def set_joint(self, arm_id: str, joint: int, value: float) -> bool:
        """Set a joint on a specific arm."""
        arm = self.get_arm(arm_id)
        if arm is None:
            print(f"Arm not found: {arm_id}")
            return False
        return arm.set_joint(joint, value)

    def set_gripper(self, arm_id: str, value: float) -> bool:
        """Set gripper on a specific arm."""
        arm = self.get_arm(arm_id)
        if arm is None:
            print(f"Arm not found: {arm_id}")
            return False
        return arm.set_gripper(value)

    def emergency_stop_all(self):
        """Emergency stop all arms."""
        print("EMERGENCY STOP: All arms")
        for arm in self.arms.values():
            arm.emergency_stop()

    def shutdown_all(self):
        """Graceful shutdown of all arms."""
        for arm in self.arms.values():
            arm.shutdown()

    @property
    def arm_count(self) -> int:
        """Number of connected arms."""
        return len(self.arms)

    @property
    def arm_ids(self) -> List[str]:
        """List of connected arm IDs."""
        return list(self.arms.keys())
