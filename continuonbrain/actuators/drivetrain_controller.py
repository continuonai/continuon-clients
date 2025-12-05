"""Lightweight drivetrain controller for steering/throttle over PCA9685.

Provides a simple abstraction for RC-car style steering (standard servo)
and throttle (continuous servo/ESC) with safety clamping in normalized
[-1.0, 1.0] space. Designed to operate in mock mode when hardware drivers
are unavailable.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, asdict
import json
import os
from typing import Optional


ServoKit = None
servokit_spec = importlib.util.find_spec("adafruit_servokit")
if servokit_spec:
    from adafruit_servokit import ServoKit  # type: ignore


@dataclass
class DrivetrainConfig:
    """Configuration for the RC drivetrain.

    Defaults target the documented PCA9685 wiring (``servo[0]`` for steering,
    ``continuous_servo[1]`` for throttle). Channels can be overridden via
    environment variables or a JSON config file:

    - ``DRIVETRAIN_STEERING_CHANNEL``
    - ``DRIVETRAIN_THROTTLE_CHANNEL``
    - ``DRIVETRAIN_I2C_ADDRESS`` (hex or int)
    - ``DRIVETRAIN_CONFIG_PATH`` (JSON file containing these keys, optionally
      nested under a ``"drivetrain"`` entry)
    """

    steering_channel: int = 0
    throttle_channel: int = 1
    i2c_address: int = 0x40
    steering_range_degrees: float = 30.0  # +/- degrees from center
    steering_center_degrees: float = 90.0

    @classmethod
    def _coerce_int(cls, value: object, label: str) -> Optional[int]:
        try:
            return int(str(value), 0)
        except (TypeError, ValueError):
            print(f"⚠️  Ignoring invalid {label} override: {value}")
            return None

    def channel_summary(self) -> str:
        """Return a human-readable summary of the configured channels."""

        return (
            f"steering=servo[{self.steering_channel}], "
            f"throttle=continuous_servo[{self.throttle_channel}]"
        )

    @classmethod
    def from_sources(cls, json_config_path: Optional[str] = None) -> "DrivetrainConfig":
        """Load configuration with environment/JSON overrides applied."""

        base_config = cls()
        overrides: dict[str, object] = {}

        config_path = json_config_path or os.getenv("DRIVETRAIN_CONFIG_PATH")
        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as fp:
                    loaded = json.load(fp)
                if isinstance(loaded, dict):
                    candidate = loaded.get("drivetrain", loaded)
                    overrides.update(candidate)
            except Exception as exc:
                print(f"⚠️  Failed to load drivetrain config from {config_path}: {exc}")

        env_overrides = {
            "steering_channel": os.getenv("DRIVETRAIN_STEERING_CHANNEL"),
            "throttle_channel": os.getenv("DRIVETRAIN_THROTTLE_CHANNEL"),
            "i2c_address": os.getenv("DRIVETRAIN_I2C_ADDRESS"),
        }

        sanitized_overrides: dict[str, object] = {}
        for key, value in {**overrides, **env_overrides}.items():
            if value is None:
                continue
            if key in {"steering_channel", "throttle_channel", "i2c_address"}:
                coerced = cls._coerce_int(value, key)
                if coerced is not None:
                    sanitized_overrides[key] = coerced
            elif key in {"steering_range_degrees", "steering_center_degrees"}:
                try:
                    sanitized_overrides[key] = float(value)
                except (TypeError, ValueError):
                    print(f"⚠️  Ignoring invalid {key} override: {value}")

        config_values = {**base_config.__dict__, **sanitized_overrides}
        return cls(**config_values)


class DrivetrainController:
    """Normalized steering/throttle controller with mock fallback."""

    def __init__(self, config: Optional[DrivetrainConfig] = None, json_config_path: Optional[str] = None):
        self.config = config or DrivetrainConfig.from_sources(json_config_path=json_config_path)
        self.kit: Optional[ServoKit] = None
        self.is_mock = ServoKit is None
        self.initialized = False
        self.last_command: Optional[dict] = None

    def initialize(self) -> bool:
        """Initialize PCA9685-backed drivetrain controller."""

        channel_summary = self.config.channel_summary()

        if self.is_mock:
            print(
                "MOCK: Initializing drivetrain controller (PCA9685 mock) "
                f"with {channel_summary}"
            )
            self.initialized = True
            return True

        try:
            self.kit = ServoKit(channels=16, address=self.config.i2c_address)  # type: ignore[arg-type]
            self.initialized = True
            print(
                f"✅ Drivetrain controller ready at 0x{self.config.i2c_address:02x} "
                f"using {channel_summary}"
            )
            return True
        except Exception as exc:  # pragma: no cover - hardware init path
            print(
                "ERROR: Failed to initialize drivetrain controller "
                f"at 0x{self.config.i2c_address:02x} ({channel_summary}): {exc}"
            )
            self.initialized = False
            return False

    def status(self) -> dict:
        """Return a status summary including selected channels for troubleshooting."""

        mode = self.mode
        return {
            "mode": mode,
            "initialized": self.initialized,
            "hardware_available": mode == "real",
            "channels": self.config.channel_summary(),
            "config": asdict(self.config),
        }

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

