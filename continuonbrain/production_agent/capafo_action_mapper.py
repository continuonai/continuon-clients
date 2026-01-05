"""
Capafo Robot Action Mapper

Maps the 32-dimensional seed model output to Capafo robot hardware:
- Servos (arm, gripper, head)
- Motors (wheels, tracks)
- Special actions (LED, speaker, etc.)

Hardware abstraction for safe robot control.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import time


class ActionType(Enum):
    """Types of robot actions."""
    SERVO = "servo"
    MOTOR = "motor"
    DISCRETE = "discrete"
    SAFETY = "safety"


class SafetyLevel(Enum):
    """Safety levels for actions."""
    SAFE = 0       # No restrictions
    CAUTION = 1    # Reduced speed
    CRITICAL = 2   # Human confirmation required
    BLOCKED = 3    # Action blocked


@dataclass
class ActionChannel:
    """Definition of a single action channel."""
    name: str
    index: int  # Index in the 32-dim output
    action_type: ActionType
    min_value: float = -1.0
    max_value: float = 1.0
    default_value: float = 0.0
    hardware_min: float = 0.0  # Hardware units (e.g., servo angle)
    hardware_max: float = 180.0
    safety_level: SafetyLevel = SafetyLevel.SAFE
    smoothing: float = 0.0  # Exponential smoothing factor
    deadzone: float = 0.05  # Values below this are zeroed
    description: str = ""


@dataclass
class CapafoConfig:
    """Capafo robot hardware configuration."""
    name: str = "Capafo"
    version: str = "1.0"

    # Servo channels (PCA9685)
    num_servo_channels: int = 16
    servo_i2c_address: int = 0x40
    servo_frequency: int = 50  # Hz

    # Motor driver
    motor_driver: str = "L298N"
    motor_pwm_frequency: int = 1000

    # Safety
    max_speed_factor: float = 1.0
    emergency_stop_enabled: bool = True
    collision_detection: bool = True

    # Update rate
    control_frequency_hz: int = 20  # 50ms per update


# Default Capafo action mapping
CAPAFO_ACTION_MAP = [
    # Mobility (channels 0-5)
    ActionChannel(
        name="left_wheel_velocity",
        index=0,
        action_type=ActionType.MOTOR,
        min_value=-1.0, max_value=1.0,
        hardware_min=-255, hardware_max=255,
        safety_level=SafetyLevel.SAFE,
        smoothing=0.3,
        description="Left wheel motor velocity"
    ),
    ActionChannel(
        name="right_wheel_velocity",
        index=1,
        action_type=ActionType.MOTOR,
        min_value=-1.0, max_value=1.0,
        hardware_min=-255, hardware_max=255,
        safety_level=SafetyLevel.SAFE,
        smoothing=0.3,
        description="Right wheel motor velocity"
    ),
    ActionChannel(
        name="forward_velocity",
        index=2,
        action_type=ActionType.MOTOR,
        min_value=-1.0, max_value=1.0,
        hardware_min=-255, hardware_max=255,
        safety_level=SafetyLevel.SAFE,
        smoothing=0.2,
        description="Forward/backward velocity (differential drive helper)"
    ),
    ActionChannel(
        name="turn_rate",
        index=3,
        action_type=ActionType.MOTOR,
        min_value=-1.0, max_value=1.0,
        hardware_min=-180, hardware_max=180,
        safety_level=SafetyLevel.SAFE,
        smoothing=0.2,
        description="Turn rate (degrees/sec)"
    ),
    ActionChannel(
        name="reserved_mobility_1",
        index=4,
        action_type=ActionType.MOTOR,
        description="Reserved for future mobility"
    ),
    ActionChannel(
        name="reserved_mobility_2",
        index=5,
        action_type=ActionType.MOTOR,
        description="Reserved for future mobility"
    ),

    # Arm (channels 6-13)
    ActionChannel(
        name="arm_base_rotation",
        index=6,
        action_type=ActionType.SERVO,
        min_value=-1.0, max_value=1.0,
        hardware_min=0, hardware_max=180,
        safety_level=SafetyLevel.CAUTION,
        smoothing=0.5,
        description="Arm base rotation servo"
    ),
    ActionChannel(
        name="arm_shoulder",
        index=7,
        action_type=ActionType.SERVO,
        min_value=-1.0, max_value=1.0,
        hardware_min=30, hardware_max=150,
        safety_level=SafetyLevel.CAUTION,
        smoothing=0.5,
        description="Arm shoulder joint"
    ),
    ActionChannel(
        name="arm_elbow",
        index=8,
        action_type=ActionType.SERVO,
        min_value=-1.0, max_value=1.0,
        hardware_min=0, hardware_max=180,
        safety_level=SafetyLevel.SAFE,
        smoothing=0.5,
        description="Arm elbow joint"
    ),
    ActionChannel(
        name="arm_wrist_pitch",
        index=9,
        action_type=ActionType.SERVO,
        min_value=-1.0, max_value=1.0,
        hardware_min=0, hardware_max=180,
        safety_level=SafetyLevel.SAFE,
        smoothing=0.5,
        description="Wrist pitch rotation"
    ),
    ActionChannel(
        name="arm_wrist_roll",
        index=10,
        action_type=ActionType.SERVO,
        min_value=-1.0, max_value=1.0,
        hardware_min=0, hardware_max=180,
        safety_level=SafetyLevel.SAFE,
        smoothing=0.5,
        description="Wrist roll rotation"
    ),
    ActionChannel(
        name="gripper_open",
        index=11,
        action_type=ActionType.SERVO,
        min_value=0.0, max_value=1.0,
        hardware_min=10, hardware_max=90,
        safety_level=SafetyLevel.SAFE,
        smoothing=0.3,
        description="Gripper open/close"
    ),
    ActionChannel(
        name="gripper_force",
        index=12,
        action_type=ActionType.MOTOR,
        min_value=0.0, max_value=1.0,
        hardware_min=0, hardware_max=100,
        safety_level=SafetyLevel.SAFE,
        description="Gripper force limit (%)"
    ),
    ActionChannel(
        name="arm_speed_factor",
        index=13,
        action_type=ActionType.MOTOR,
        min_value=0.1, max_value=1.0,
        default_value=0.5,
        hardware_min=10, hardware_max=100,
        safety_level=SafetyLevel.SAFE,
        description="Arm movement speed factor"
    ),

    # Head (channels 14-17)
    ActionChannel(
        name="head_pan",
        index=14,
        action_type=ActionType.SERVO,
        min_value=-1.0, max_value=1.0,
        hardware_min=0, hardware_max=180,
        safety_level=SafetyLevel.SAFE,
        smoothing=0.4,
        description="Head pan (left/right)"
    ),
    ActionChannel(
        name="head_tilt",
        index=15,
        action_type=ActionType.SERVO,
        min_value=-1.0, max_value=1.0,
        hardware_min=45, hardware_max=135,
        safety_level=SafetyLevel.SAFE,
        smoothing=0.4,
        description="Head tilt (up/down)"
    ),
    ActionChannel(
        name="camera_zoom",
        index=16,
        action_type=ActionType.MOTOR,
        min_value=0.0, max_value=1.0,
        hardware_min=1.0, hardware_max=4.0,
        safety_level=SafetyLevel.SAFE,
        description="Camera zoom level"
    ),
    ActionChannel(
        name="camera_focus",
        index=17,
        action_type=ActionType.MOTOR,
        min_value=0.0, max_value=1.0,
        hardware_min=0, hardware_max=100,
        safety_level=SafetyLevel.SAFE,
        description="Camera focus (0=auto)"
    ),

    # Discrete actions (channels 18-23)
    ActionChannel(
        name="led_mode",
        index=18,
        action_type=ActionType.DISCRETE,
        min_value=0.0, max_value=1.0,
        hardware_min=0, hardware_max=7,
        safety_level=SafetyLevel.SAFE,
        description="LED mode (0=off, 1-7=patterns)"
    ),
    ActionChannel(
        name="speaker_volume",
        index=19,
        action_type=ActionType.MOTOR,
        min_value=0.0, max_value=1.0,
        hardware_min=0, hardware_max=100,
        safety_level=SafetyLevel.SAFE,
        description="Speaker volume %"
    ),
    ActionChannel(
        name="speak_intent",
        index=20,
        action_type=ActionType.DISCRETE,
        min_value=0.0, max_value=1.0,
        safety_level=SafetyLevel.SAFE,
        description="Intent to speak (triggers TTS)"
    ),
    ActionChannel(
        name="emotion_display",
        index=21,
        action_type=ActionType.DISCRETE,
        min_value=0.0, max_value=1.0,
        hardware_min=0, hardware_max=10,
        safety_level=SafetyLevel.SAFE,
        description="Emotion to display (0-10)"
    ),
    ActionChannel(
        name="attention_target_x",
        index=22,
        action_type=ActionType.MOTOR,
        min_value=-1.0, max_value=1.0,
        safety_level=SafetyLevel.SAFE,
        description="Attention target X coordinate"
    ),
    ActionChannel(
        name="attention_target_y",
        index=23,
        action_type=ActionType.MOTOR,
        min_value=-1.0, max_value=1.0,
        safety_level=SafetyLevel.SAFE,
        description="Attention target Y coordinate"
    ),

    # Safety (channels 24-27)
    ActionChannel(
        name="emergency_stop",
        index=24,
        action_type=ActionType.SAFETY,
        min_value=0.0, max_value=1.0,
        safety_level=SafetyLevel.CRITICAL,
        description="Emergency stop trigger (>0.5 = stop)"
    ),
    ActionChannel(
        name="speed_limit",
        index=25,
        action_type=ActionType.SAFETY,
        min_value=0.0, max_value=1.0,
        default_value=1.0,
        safety_level=SafetyLevel.SAFE,
        description="Global speed limit factor"
    ),
    ActionChannel(
        name="collision_override",
        index=26,
        action_type=ActionType.SAFETY,
        min_value=0.0, max_value=1.0,
        safety_level=SafetyLevel.CRITICAL,
        description="Override collision detection (dangerous)"
    ),
    ActionChannel(
        name="human_proximity_response",
        index=27,
        action_type=ActionType.SAFETY,
        min_value=0.0, max_value=1.0,
        safety_level=SafetyLevel.CAUTION,
        description="Response mode when human detected"
    ),

    # Reserved (channels 28-31)
    ActionChannel(
        name="reserved_1",
        index=28,
        action_type=ActionType.MOTOR,
        description="Reserved for expansion"
    ),
    ActionChannel(
        name="reserved_2",
        index=29,
        action_type=ActionType.MOTOR,
        description="Reserved for expansion"
    ),
    ActionChannel(
        name="reserved_3",
        index=30,
        action_type=ActionType.MOTOR,
        description="Reserved for expansion"
    ),
    ActionChannel(
        name="reserved_4",
        index=31,
        action_type=ActionType.MOTOR,
        description="Reserved for expansion"
    ),
]


@dataclass
class HardwareCommand:
    """Command to send to hardware."""
    channel_name: str
    hardware_value: float
    action_type: ActionType
    safety_level: SafetyLevel
    timestamp: float = field(default_factory=time.time)


class CapafoActionMapper:
    """
    Maps seed model output to Capafo robot hardware commands.

    Features:
    - Normalization and scaling
    - Safety filtering
    - Smoothing and deadzone
    - Emergency stop handling
    """

    def __init__(
        self,
        config: Optional[CapafoConfig] = None,
        action_map: Optional[List[ActionChannel]] = None,
        hardware_backend: Optional[Any] = None,
    ):
        self.config = config or CapafoConfig()
        self.action_map = {ch.index: ch for ch in (action_map or CAPAFO_ACTION_MAP)}
        self.hardware_backend = hardware_backend

        # State
        self._smoothed_values: Dict[int, float] = {}
        self._emergency_stop_active = False
        self._last_update_time = time.time()

        # Callbacks
        self._safety_callbacks: List[Callable[[str, float], bool]] = []

    def add_safety_callback(self, callback: Callable[[str, float], bool]) -> None:
        """Add a safety callback that can block actions."""
        self._safety_callbacks.append(callback)

    def _normalize_to_hardware(self, value: float, channel: ActionChannel) -> float:
        """Convert normalized [-1, 1] to hardware units."""
        # Clamp to valid range
        value = max(channel.min_value, min(channel.max_value, value))

        # Map to hardware range
        norm = (value - channel.min_value) / (channel.max_value - channel.min_value)
        hardware_value = channel.hardware_min + norm * (channel.hardware_max - channel.hardware_min)

        return hardware_value

    def _apply_deadzone(self, value: float, channel: ActionChannel) -> float:
        """Apply deadzone to value."""
        if abs(value) < channel.deadzone:
            return channel.default_value
        return value

    def _apply_smoothing(self, value: float, channel: ActionChannel) -> float:
        """Apply exponential smoothing."""
        if channel.smoothing <= 0:
            return value

        prev = self._smoothed_values.get(channel.index, value)
        smoothed = channel.smoothing * prev + (1 - channel.smoothing) * value
        self._smoothed_values[channel.index] = smoothed
        return smoothed

    def _check_safety(self, channel: ActionChannel, value: float) -> Tuple[bool, str]:
        """Check if action is safe to execute."""
        # Emergency stop check
        if self._emergency_stop_active and channel.action_type != ActionType.SAFETY:
            return False, "Emergency stop active"

        # Critical actions require confirmation
        if channel.safety_level == SafetyLevel.CRITICAL:
            # For now, block critical actions automatically
            # In production, this would trigger human confirmation
            return False, f"Critical action {channel.name} requires confirmation"

        # Run custom safety callbacks
        for callback in self._safety_callbacks:
            if not callback(channel.name, value):
                return False, f"Blocked by safety callback"

        return True, "OK"

    def map_output(
        self,
        model_output: np.ndarray,
        apply_safety: bool = True,
    ) -> Dict[str, HardwareCommand]:
        """
        Map 32-dim model output to hardware commands.

        Args:
            model_output: Shape (32,) or (1, 32) from seed model
            apply_safety: Whether to apply safety checks

        Returns:
            Dictionary of channel_name -> HardwareCommand
        """
        if model_output.ndim > 1:
            model_output = model_output.flatten()

        if len(model_output) != 32:
            raise ValueError(f"Expected 32-dim output, got {len(model_output)}")

        # Check for emergency stop first
        estop_channel = self.action_map.get(24)
        if estop_channel and model_output[24] > 0.5:
            self._emergency_stop_active = True

        commands = {}

        for idx, value in enumerate(model_output):
            channel = self.action_map.get(idx)
            if channel is None:
                continue

            # Apply processing
            value = self._apply_deadzone(float(value), channel)
            value = self._apply_smoothing(value, channel)

            # Safety check
            if apply_safety:
                safe, reason = self._check_safety(channel, value)
                if not safe:
                    continue

            # Convert to hardware
            hardware_value = self._normalize_to_hardware(value, channel)

            commands[channel.name] = HardwareCommand(
                channel_name=channel.name,
                hardware_value=hardware_value,
                action_type=channel.action_type,
                safety_level=channel.safety_level,
            )

        return commands

    def execute_commands(self, commands: Dict[str, HardwareCommand]) -> Dict[str, bool]:
        """
        Execute hardware commands.

        Returns dict of channel_name -> success
        """
        results = {}

        if self.hardware_backend is None:
            # Simulation mode
            for name, cmd in commands.items():
                results[name] = True
            return results

        # Group by action type for efficient execution
        servos = {k: v for k, v in commands.items() if v.action_type == ActionType.SERVO}
        motors = {k: v for k, v in commands.items() if v.action_type == ActionType.MOTOR}
        discrete = {k: v for k, v in commands.items() if v.action_type == ActionType.DISCRETE}

        # Execute servos
        if servos and hasattr(self.hardware_backend, 'set_servos'):
            servo_values = {k: v.hardware_value for k, v in servos.items()}
            try:
                self.hardware_backend.set_servos(servo_values)
                for k in servos:
                    results[k] = True
            except Exception as e:
                for k in servos:
                    results[k] = False

        # Execute motors
        if motors and hasattr(self.hardware_backend, 'set_motors'):
            motor_values = {k: v.hardware_value for k, v in motors.items()}
            try:
                self.hardware_backend.set_motors(motor_values)
                for k in motors:
                    results[k] = True
            except Exception as e:
                for k in motors:
                    results[k] = False

        # Execute discrete
        for name, cmd in discrete.items():
            try:
                if hasattr(self.hardware_backend, f'set_{name}'):
                    getattr(self.hardware_backend, f'set_{name}')(int(cmd.hardware_value))
                    results[name] = True
                else:
                    results[name] = False
            except Exception:
                results[name] = False

        return results

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop state."""
        self._emergency_stop_active = False
        self._smoothed_values.clear()

    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space."""
        channels_by_type = {}
        for idx, channel in self.action_map.items():
            type_name = channel.action_type.value
            if type_name not in channels_by_type:
                channels_by_type[type_name] = []
            channels_by_type[type_name].append({
                "index": idx,
                "name": channel.name,
                "range": [channel.min_value, channel.max_value],
                "hardware_range": [channel.hardware_min, channel.hardware_max],
                "safety": channel.safety_level.name,
            })

        return {
            "total_channels": 32,
            "channels_by_type": channels_by_type,
            "safety_channels": [
                ch.name for ch in self.action_map.values()
                if ch.safety_level in [SafetyLevel.CRITICAL, SafetyLevel.CAUTION]
            ],
        }

    def export_mapping(self, output_path: Path) -> None:
        """Export action mapping to JSON."""
        mapping = {
            "config": {
                "name": self.config.name,
                "version": self.config.version,
                "control_frequency_hz": self.config.control_frequency_hz,
            },
            "channels": [
                {
                    "index": ch.index,
                    "name": ch.name,
                    "type": ch.action_type.value,
                    "range": [ch.min_value, ch.max_value],
                    "hardware_range": [ch.hardware_min, ch.hardware_max],
                    "default": ch.default_value,
                    "safety": ch.safety_level.name,
                    "smoothing": ch.smoothing,
                    "deadzone": ch.deadzone,
                    "description": ch.description,
                }
                for ch in self.action_map.values()
            ],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)

        print(f"Exported action mapping to: {output_path}")


class MockHardwareBackend:
    """Mock hardware backend for testing."""

    def __init__(self):
        self.servo_states = {}
        self.motor_states = {}
        self.discrete_states = {}

    def set_servos(self, values: Dict[str, float]) -> None:
        self.servo_states.update(values)
        print(f"[MOCK] Servos: {values}")

    def set_motors(self, values: Dict[str, float]) -> None:
        self.motor_states.update(values)
        print(f"[MOCK] Motors: {values}")

    def set_led_mode(self, mode: int) -> None:
        self.discrete_states['led_mode'] = mode
        print(f"[MOCK] LED mode: {mode}")

    def set_emotion_display(self, emotion: int) -> None:
        self.discrete_states['emotion_display'] = emotion
        print(f"[MOCK] Emotion: {emotion}")


if __name__ == "__main__":
    # Test the action mapper
    mapper = CapafoActionMapper(hardware_backend=MockHardwareBackend())

    # Print action space info
    info = mapper.get_action_space_info()
    print("Action Space Info:")
    print(json.dumps(info, indent=2))

    # Test mapping
    test_output = np.random.randn(32) * 0.5
    test_output[24] = -1.0  # Don't trigger emergency stop

    print("\nMapping test output...")
    commands = mapper.map_output(test_output)

    print(f"\nGenerated {len(commands)} commands:")
    for name, cmd in list(commands.items())[:10]:
        print(f"  {name}: {cmd.hardware_value:.2f} ({cmd.action_type.value})")

    # Execute
    print("\nExecuting commands...")
    results = mapper.execute_commands(commands)

    # Export
    mapper.export_mapping(Path("capafo_action_mapping.json"))
