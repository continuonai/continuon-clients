"""
Lightweight hardware detection for Trainer UI.

Detects:
- I2C devices at 0x40, 0x41 (dual PCA9685 for arms)
- Hailo-8 via lspci (device ID 1e60:2864, 26 TOPS)
- Audio output (espeak-ng availability)
- Cameras via V4L2
"""

import os
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class I2CDevice:
    """Detected I2C device."""
    address: int
    name: str
    device_type: str


@dataclass
class HardwareConfig:
    """Hardware configuration detected at startup."""
    # Arms (PCA9685 servo controllers)
    arms: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Hailo AI accelerator
    hailo_available: bool = False
    hailo_model: str = ""
    hailo_tops: float = 0.0

    # Audio
    audio_available: bool = False
    audio_backend: str = ""

    # Cameras
    cameras: List[str] = field(default_factory=list)

    # Mock mode
    is_mock: bool = False


class TrainerHardwareDetector:
    """
    Lightweight hardware detection for Trainer UI.
    Focuses on components needed for robot training.
    """

    def __init__(self, mock_mode: bool = False):
        """
        Initialize detector.

        Args:
            mock_mode: If True, inject mock hardware for testing
        """
        self.mock_mode = mock_mode or os.environ.get("TRAINER_MOCK_HARDWARE", "0") in ("1", "true", "yes")
        self.arm_0_address = int(os.environ.get("TRAINER_ARM_0_ADDRESS", "0x40"), 16)
        self.arm_1_address = int(os.environ.get("TRAINER_ARM_1_ADDRESS", "0x41"), 16)

    def detect_all(self) -> HardwareConfig:
        """Run all detection and return hardware configuration."""
        config = HardwareConfig(is_mock=self.mock_mode)

        if self.mock_mode:
            return self._inject_mock_hardware(config)

        # Detect components
        self._detect_i2c_arms(config)
        self._detect_hailo(config)
        self._detect_audio(config)
        self._detect_cameras(config)

        # If nothing detected, fall back to mock
        if not config.arms and not config.hailo_available and not config.cameras:
            print("No hardware detected, falling back to mock mode")
            return self._inject_mock_hardware(config)

        return config

    def _detect_i2c_arms(self, config: HardwareConfig) -> None:
        """Detect PCA9685 servo controllers for robot arms."""
        if not shutil.which("i2cdetect"):
            return

        try:
            result = subprocess.run(
                ["i2cdetect", "-y", "1"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return

            # Parse i2cdetect output
            detected_addresses = set()
            for line in result.stdout.strip().split("\n")[1:]:
                parts = line.split()
                if len(parts) < 2:
                    continue
                for addr in parts[1:]:
                    if addr != "--" and addr != "UU":
                        try:
                            detected_addresses.add(int(addr, 16))
                        except ValueError:
                            continue

            # Check for arm controllers
            if self.arm_0_address in detected_addresses:
                config.arms["arm_0"] = {
                    "i2c_address": f"0x{self.arm_0_address:02x}",
                    "connected": True,
                    "is_mock": False,
                }
                print(f"Found arm controller at 0x{self.arm_0_address:02x} (arm_0)")

            if self.arm_1_address in detected_addresses:
                config.arms["arm_1"] = {
                    "i2c_address": f"0x{self.arm_1_address:02x}",
                    "connected": True,
                    "is_mock": False,
                }
                print(f"Found arm controller at 0x{self.arm_1_address:02x} (arm_1)")

        except subprocess.TimeoutExpired:
            print("Warning: i2cdetect timed out")
        except Exception as e:
            print(f"Warning: I2C detection failed: {e}")

    def _detect_hailo(self, config: HardwareConfig) -> None:
        """Detect Hailo-8/8L AI accelerator."""
        if not shutil.which("lspci"):
            return

        try:
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return

            for line in result.stdout.lower().split("\n"):
                if "hailo" in line or "1e60:2864" in line or "1e60:2862" in line:
                    config.hailo_available = True

                    # Identify model
                    if "hailo-8l" in line or "1e60:2862" in line:
                        config.hailo_model = "Hailo-8L"
                        config.hailo_tops = 13.0
                    else:
                        config.hailo_model = "Hailo-8"
                        config.hailo_tops = 26.0

                    print(f"Found {config.hailo_model} ({config.hailo_tops} TOPS)")
                    break
        except Exception as e:
            print(f"Warning: Hailo detection failed: {e}")

        # Also check for /dev/hailo* devices
        if not config.hailo_available:
            try:
                hailo_devs = list(Path("/dev").glob("hailo*"))
                if hailo_devs:
                    config.hailo_available = True
                    config.hailo_model = "Hailo-8"  # Assume full version
                    config.hailo_tops = 26.0
                    print(f"Found Hailo via device file: {hailo_devs[0]}")
            except Exception:
                pass

    def _detect_audio(self, config: HardwareConfig) -> None:
        """Detect audio output capability."""
        # Check for espeak-ng first (preferred)
        if shutil.which("espeak-ng"):
            config.audio_available = True
            config.audio_backend = "espeak-ng"
            print("Found audio backend: espeak-ng")
            return

        # Fall back to espeak
        if shutil.which("espeak"):
            config.audio_available = True
            config.audio_backend = "espeak"
            print("Found audio backend: espeak")
            return

        # Check for macOS say command
        if shutil.which("say"):
            config.audio_available = True
            config.audio_backend = "say"
            print("Found audio backend: say (macOS)")

    def _detect_cameras(self, config: HardwareConfig) -> None:
        """Detect available cameras."""
        # Check V4L2 devices
        if shutil.which("v4l2-ctl"):
            try:
                result = subprocess.run(
                    ["v4l2-ctl", "--list-devices"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and "/dev/video" in result.stdout:
                    # Parse camera names
                    lines = result.stdout.strip().split("\n")
                    current_camera = None
                    for line in lines:
                        if not line.startswith("\t") and line.strip():
                            current_camera = line.strip().rstrip(":")
                        elif "/dev/video" in line and current_camera:
                            config.cameras.append(current_camera)
                            current_camera = None

                    if not config.cameras:
                        config.cameras.append("Webcam")

                    print(f"Found cameras: {config.cameras}")
            except Exception as e:
                print(f"Warning: Camera detection failed: {e}")

        # If no V4L2, check for generic video devices
        if not config.cameras:
            video_devices = list(Path("/dev").glob("video*"))
            if video_devices:
                config.cameras.append("Webcam")
                print("Found generic video device")

    def _inject_mock_hardware(self, config: HardwareConfig) -> HardwareConfig:
        """Inject mock hardware for testing."""
        print("Injecting mock hardware for development...")

        config.is_mock = True

        # Mock dual arms
        config.arms = {
            "arm_0": {
                "i2c_address": f"0x{self.arm_0_address:02x}",
                "connected": True,
                "is_mock": True,
            },
            "arm_1": {
                "i2c_address": f"0x{self.arm_1_address:02x}",
                "connected": True,
                "is_mock": True,
            },
        }

        # Mock Hailo
        config.hailo_available = True
        config.hailo_model = "Hailo-8 (Mock)"
        config.hailo_tops = 26.0

        # Mock audio
        config.audio_available = True
        config.audio_backend = "mock"

        # Mock camera
        config.cameras = ["Webcam (Mock)"]

        return config
