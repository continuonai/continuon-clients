"""
Motor Controller - Interface to RC car motors.

Supports both real hardware (GPIO/PWM) and mock mode for testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time


@dataclass
class MotorState:
    """Current state of the motors."""

    left_speed: float = 0.0  # -1.0 to 1.0
    right_speed: float = 0.0  # -1.0 to 1.0
    last_update: float = 0.0


class BaseMotorController(ABC):
    """Abstract base class for motor controllers."""

    @abstractmethod
    def set_motors(self, left: float, right: float):
        """Set motor speeds. Values from -1.0 (full reverse) to 1.0 (full forward)."""
        pass

    @abstractmethod
    def stop(self):
        """Stop all motors immediately."""
        pass

    @abstractmethod
    def get_state(self) -> MotorState:
        """Get current motor state."""
        pass


class MockMotorController(BaseMotorController):
    """
    Mock motor controller for testing without hardware.

    Prints actions to console instead of controlling real motors.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.state = MotorState()

    def set_motors(self, left: float, right: float):
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))

        self.state.left_speed = left
        self.state.right_speed = right
        self.state.last_update = time.time()

        if self.verbose:
            if left == 0 and right == 0:
                print("  [Motors] Stopped")
            elif left == right:
                direction = "forward" if left > 0 else "backward"
                print(f"  [Motors] {direction} at {abs(left) * 100:.0f}%")
            elif left == -right:
                direction = "left" if left < right else "right"
                print(f"  [Motors] Turning {direction}")
            else:
                print(f"  [Motors] L={left:.2f} R={right:.2f}")

    def stop(self):
        self.set_motors(0, 0)

    def get_state(self) -> MotorState:
        return self.state


class MotorController(BaseMotorController):
    """
    Real motor controller using GPIO/PWM.

    This is a template - adapt to your specific hardware:
    - PCA9685 servo controller
    - L298N motor driver
    - Direct GPIO PWM
    - etc.
    """

    def __init__(
        self,
        left_pin: int = 12,
        right_pin: int = 13,
        enable_pin: int | None = None,
    ):
        self.left_pin = left_pin
        self.right_pin = right_pin
        self.enable_pin = enable_pin
        self.state = MotorState()

        self._initialized = False
        self._init_hardware()

    def _init_hardware(self):
        """Initialize GPIO/PWM hardware."""
        try:
            # Try to import RPi.GPIO (Raspberry Pi)
            import RPi.GPIO as GPIO

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.left_pin, GPIO.OUT)
            GPIO.setup(self.right_pin, GPIO.OUT)

            if self.enable_pin:
                GPIO.setup(self.enable_pin, GPIO.OUT)
                GPIO.output(self.enable_pin, GPIO.HIGH)

            # Create PWM instances at 1000 Hz
            self._left_pwm = GPIO.PWM(self.left_pin, 1000)
            self._right_pwm = GPIO.PWM(self.right_pin, 1000)
            self._left_pwm.start(0)
            self._right_pwm.start(0)

            self._gpio = GPIO
            self._initialized = True
            print("[Motors] GPIO initialized")

        except ImportError:
            print("[Motors] RPi.GPIO not available, using mock mode")
            self._mock = MockMotorController(verbose=True)

        except Exception as e:
            print(f"[Motors] GPIO init failed: {e}, using mock mode")
            self._mock = MockMotorController(verbose=True)

    def set_motors(self, left: float, right: float):
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))

        self.state.left_speed = left
        self.state.right_speed = right
        self.state.last_update = time.time()

        if not self._initialized:
            self._mock.set_motors(left, right)
            return

        # Convert -1.0 to 1.0 range to PWM duty cycle
        # This is simplified - real implementation needs direction control
        left_duty = abs(left) * 100
        right_duty = abs(right) * 100

        self._left_pwm.ChangeDutyCycle(left_duty)
        self._right_pwm.ChangeDutyCycle(right_duty)

    def stop(self):
        self.set_motors(0, 0)

    def get_state(self) -> MotorState:
        return self.state

    def cleanup(self):
        """Clean up GPIO resources."""
        if self._initialized:
            self._left_pwm.stop()
            self._right_pwm.stop()
            self._gpio.cleanup()


# === Action Executor ===

def create_executor(controller: BaseMotorController, turn_duration: float = 0.3):
    """
    Create an action executor function for the given motor controller.

    This translates high-level actions into motor commands.
    """

    def execute(action: dict):
        action_type = action.get("type", "")
        speed = action.get("speed", 0.5)

        if action_type == "forward":
            controller.set_motors(speed, speed)

        elif action_type == "backward":
            controller.set_motors(-speed, -speed)

        elif action_type == "left":
            # Spin left: left motor backward, right motor forward
            controller.set_motors(-speed, speed)
            time.sleep(turn_duration)
            controller.stop()

        elif action_type == "right":
            # Spin right: left motor forward, right motor backward
            controller.set_motors(speed, -speed)
            time.sleep(turn_duration)
            controller.stop()

        elif action_type == "stop":
            controller.stop()

        else:
            print(f"  [Motors] Unknown action: {action_type}")

    return execute
