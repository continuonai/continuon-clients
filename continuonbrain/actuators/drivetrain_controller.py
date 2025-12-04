"""Lightweight drivetrain controller for steering/throttle over PCA9685.

Provides a simple abstraction for RC-car style steering (standard servo)
and throttle (continuous servo/ESC) with safety clamping in normalized
[-1.0, 1.0] space. Designed to operate in mock mode when hardware drivers
are unavailable.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Optional


ServoKit = None
servokit_spec = importlib.util.find_spec("adafruit_servokit")
if servokit_spec:
    from adafruit_servokit import ServoKit  # type: ignore


@dataclass
class DrivetrainConfig:
    """Configuration for the RC drivetrain."""

    steering_channel: int = 6
    throttle_channel: int = 7
    i2c_address: int = 0x40
    steering_range_degrees: float = 30.0  # +/- degrees from center
    steering_center_degrees: float = 90.0


class DrivetrainController:
    """Normalized steering/throttle controller with mock fallback."""

    def __init__(self, config: Optional[DrivetrainConfig] = None):
        self.config = config or DrivetrainConfig()
        self.kit: Optional[ServoKit] = None
        self.is_mock = ServoKit is None
        self.initialized = False
        self.last_command: Optional[dict] = None

    def initialize(self) -> bool:
        """Initialize PCA9685-backed drivetrain controller."""

        if self.is_mock:
            print("MOCK: Initializing drivetrain controller (PCA9685 mock)")
            self.initialized = True
            return True

        try:
            self.kit = ServoKit(channels=16, address=self.config.i2c_address)  # type: ignore[arg-type]
            self.initialized = True
            print(f"✅ Drivetrain controller ready at 0x{self.config.i2c_address:02x}")
            return True
        except Exception as exc:  # pragma: no cover - hardware init path
            print(f"ERROR: Failed to initialize drivetrain controller: {exc}")
            self.initialized = False
            return False

    def apply_drive(self, steering: float, throttle: float) -> dict:
        """Apply steering/throttle after clamping to [-1, 1]."""

        steering_clamped = max(-1.0, min(1.0, float(steering)))
        throttle_clamped = max(-1.0, min(1.0, float(throttle)))

        if not self.initialized:
            result = {
                "success": False,
                "message": "Drivetrain not initialized",
                "steering": steering_clamped,
                "throttle": throttle_clamped,
            }
            self.last_command = result
            return result

        try:
            steering_angle = (
                self.config.steering_center_degrees
                + steering_clamped * self.config.steering_range_degrees
            )

            if self.is_mock:
                # In mock mode, flag that hardware output is unavailable
                result = {
                    "success": False,
                    "message": "MOCK mode: PCA9685 output inactive; drive command not sent",
                    "steering": steering_clamped,
                    "throttle": throttle_clamped,
                    "mode": "mock",
                    "hardware_available": False,
                }
                self.last_command = result
                return result

            if self.kit is None:
                raise RuntimeError("ServoKit not initialized")

            # Apply steering to standard servo channel
            self.kit.servo[self.config.steering_channel].angle = steering_angle

            # Apply throttle using continuous servo interface when available
            if hasattr(self.kit, "continuous_servo"):
                self.kit.continuous_servo[self.config.throttle_channel].throttle = throttle_clamped
            else:  # Fallback to standard servo mapping
                throttle_angle = self.config.steering_center_degrees + throttle_clamped * self.config.steering_range_degrees
                self.kit.servo[self.config.throttle_channel].angle = throttle_angle

            result = {
                "success": True,
                "message": "Drive command applied",
                "steering": steering_clamped,
                "throttle": throttle_clamped,
                "mode": "real",
                "hardware_available": True,
            }
            self.last_command = result
            return result
        except Exception as exc:  # pragma: no cover - hardware path
            result = {
                "success": False,
                "message": f"Failed to apply drive command: {exc}",
                "steering": steering_clamped,
                "throttle": throttle_clamped,
            }
            self.last_command = result
            return result

    @property
    def mode(self) -> str:
        """Return current controller mode label."""

        if self.is_mock:
            return "mock"
        return "real" if self.initialized else "unavailable"

