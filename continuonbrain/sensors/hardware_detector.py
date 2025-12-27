"""
Hardware detection and auto-configuration for ContinuonBrain robot systems.
Detects cameras, HATs, servos, and accessories automatically across platforms.
Supports Mock fallback for development on non-robot hardware (Windows/macOS).
"""
import subprocess
import os
import json
import platform
import importlib.util
import sys
import shlex
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from datetime import datetime

from continuonbrain.system_installer import SystemInstaller


@dataclass
class HardwareDevice:
    """Detected hardware device information."""
    device_type: str  # camera, hat, servo_controller, imu, etc.
    name: str
    vendor: str
    interface: str  # usb, i2c, spi, gpio, pcie
    address: Optional[str] = None  # I2C address, USB port, etc.
    capabilities: List[str] = None
    config: Dict[str, Any] = None
    is_mock: bool = False
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.config is None:
            self.config = {}


class HardwareDetector:
    """
    Auto-detects connected hardware.
    Identifies cameras, HATs, servo controllers, and other accessories.
    Provides mock fallbacks for local development environments.
    """
    
    def __init__(self):
        self.detected_devices: List[HardwareDevice] = []
        self.platform_info: Dict[str, Any] = {}
        self.missing_dependencies: List[str] = []
        self.hailo_offload_enabled = os.environ.get("CONTINUON_HAILO_OFFLOAD", "1").lower() in ("1", "true", "yes")
        self.use_mock = os.environ.get("CONTINUON_MOCK_HARDWARE", "0").lower() in ("1", "true", "yes")
    
    def detect_all(self, auto_install: bool = False, allow_system_install: bool = False) -> List[HardwareDevice]:
        """Run all detection methods and return found devices."""
        self.detected_devices = []
        self._detect_environment()
        
        system = self.platform_info["os"]
        print(f"Scanning for hardware devices on {system} ({self.platform_info.get('distro', 'Generic')})...")
        
        # Cross-platform USB detection
        self.detect_usb_devices()
        
        # Platform-specific detection
        if system == "Linux":
            self.detect_i2c_devices()
            self.detect_cameras_linux()
            self.detect_hats()
            self.detect_accelerators_linux()
            self.detect_gpio_devices()
        elif system == "Windows":
            self.detect_cameras_windows()
            
        self.detect_desktop_peripherals()

        # Mock Fallback
        if not self.detected_devices or self.use_mock:
            self._inject_mocks()

        if auto_install and self.missing_dependencies:
            self._auto_install_missing(allow_system_install=allow_system_install)
            # Re-check
            self.missing_dependencies = [
                dep for dep in self.missing_dependencies if not self._has_module(dep.split()[0])
            ]

        if self.missing_dependencies:
            print("⚠️  Missing optional drivers/libs:")
            for dep in self.missing_dependencies:
                print(f"   - {dep}")
            print()

        return self.detected_devices

    def _detect_environment(self) -> None:
        """Record platform info and gather dependency hints."""
        system = platform.system()
        machine = platform.machine()
        self.platform_info = {"os": system, "arch": machine}

        # OS-specific driver hints
        driver_hints = {
            "Linux": [
                ("depthai", "pip install depthai --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-local/"),
                ("numpy", "pip install numpy"),
                ("smbus2", "pip install smbus2"),
            ],
            "Windows": [
                ("depthai", "pip install depthai"),
                ("numpy", "pip install numpy"),
            ],
            "Darwin": [
                ("depthai", "pip install depthai"),
                ("numpy", "pip install numpy"),
            ],
        }

        for name, install_hint in driver_hints.get(system, []):
            if not self._has_module(name):
                self.missing_dependencies.append(f"{name} (hint: {install_hint})")

        if system == "Linux":
            try:
                with open("/etc/os-release", "r", encoding="utf-8") as f:
                    os_release = f.read().lower()
                if "raspbian" in os_release or "debian" in os_release:
                    self.platform_info["distro"] = "Raspberry Pi OS"
            except Exception:
                pass

    def _has_module(self, name: str) -> bool:
        """Return True if the python module is importable."""
        return importlib.util.find_spec(name) is not None

    def _auto_install_missing(self, allow_system_install: bool = False) -> None:
        """Attempt best-effort installation of missing Python dependencies."""
        print("🔧 Attempting to install missing Python drivers...")
        for dep in list(self.missing_dependencies):
            parts = dep.split("(hint:")
            if len(parts) < 2: continue
            hint = parts[1].strip().rstrip(")")
            if not hint.startswith("pip "): continue

            try:
                cmd_tokens = shlex.split(hint)
                if cmd_tokens[0] == "pip":
                    cmd_tokens = [sys.executable, "-m", "pip"] + cmd_tokens[1:]
                subprocess.run(cmd_tokens, check=False)
            except Exception as exc:
                print(f"   ⚠️  Install attempt failed for {dep}: {exc}")

    def detect_usb_devices(self):
        """Detect USB devices (Linux/macOS/Windows)."""
        system = self.platform_info["os"]
        
        if system == "Linux":
            self._detect_usb_linux()
        elif system == "Windows":
            self._detect_usb_windows()
        # macOS detection could be added here via 'system_profiler SPUSBDataType'

    def _detect_usb_linux(self):
        try:
            import shutil
            if not shutil.which('lsusb'): return
            
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            if result.returncode != 0: return
            
            for line in result.stdout.strip().split('\n'):
                match = re.match(r'Bus (\d+) Device (\d+): ID ([0-9a-f]{4}):([0-9a-f]{4}) (.+)', line)
                if not match: continue
                bus, device, vid, pid, name = match.groups()
                self._parse_usb_device(vid, pid, name, f"bus_{bus}_dev_{device}")
        except Exception as e:
            print(f"Warning: Linux USB detection failed: {e}")

    def _detect_usb_windows(self):
        """Simple USB detection for Windows via WMIC."""
        try:
            # This is a broad check for connected USB devices
            result = subprocess.run(['wmic', 'path', 'Win32_USBControllerDevice', 'get', 'Dependent'], 
                                   capture_output=True, text=True)
            if result.returncode != 0: return
            
            # We don't get PID/VID easily here without deeper parsing, 
            # but we can look for known vendor strings in PNPDeviceID if we queried that instead.
            # For now, we'll rely more on Python module detection for things like DepthAI.
            pass
        except Exception:
            pass

    def _parse_usb_device(self, vid: str, pid: str, name: str, address: str):
        # OAK cameras (Luxonis)
        if vid == '03e7':
            device_name = "OAK-D Lite" if "MyriadX" in name else "OAK Camera"
            self.detected_devices.append(HardwareDevice(
                device_type="depth_camera", name=device_name, vendor="Luxonis",
                interface="usb3", address=address, capabilities=["rgb", "depth", "stereo", "ai"],
                config={"vendor_id": vid, "product_id": pid, "driver": "depthai", "hailo_offload": self.hailo_offload_enabled}
            ))
            print(f"✅ Found: {device_name} (USB)")
        
        # Intel RealSense
        elif vid == '8086' and 'RealSense' in name:
            self.detected_devices.append(HardwareDevice(
                device_type="depth_camera", name="RealSense", vendor="Intel",
                interface="usb3", address=address, capabilities=["rgb", "depth", "imu"],
                config={"vendor_id": vid, "product_id": pid, "driver": "librealsense2"}
            ))
            print(f"✅ Found: Intel RealSense (USB)")

    def detect_i2c_devices(self):
        """Linux-only I2C scan."""
        try:
            import shutil
            if not shutil.which('i2cdetect'): return
            
            result = subprocess.run(['i2cdetect', '-y', '1'], capture_output=True, text=True)
            if result.returncode != 0: return
            
            for line in result.stdout.strip().split('\n')[1:]:
                parts = line.split()
                if len(parts) < 2: continue
                for addr in parts[1:]:
                    if addr != '--' and addr != 'UU':
                        try:
                            val = int(addr, 16)
                            device = self._identify_i2c_device(val)
                            if device:
                                self.detected_devices.append(device)
                                print(f"✅ Found: {device.name} at I2C 0x{val:02x}")
                        except ValueError: continue
        except Exception: pass

    def _identify_i2c_device(self, address: int) -> Optional[HardwareDevice]:
        if address == 0x40:
            return HardwareDevice(
                device_type="servo_controller", name="PCA9685 16-Channel PWM",
                vendor="Adafruit/NXP", interface="i2c", address=f"0x{address:02x}",
                capabilities=["pwm", "servo", "led"], config={"channels": 16, "driver": "adafruit_servokit"}
            )
        elif address == 0x68 or address == 0x69:
            return HardwareDevice(
                device_type="imu", name="MPU6050 IMU", vendor="InvenSense",
                interface="i2c", address=f"0x{address:02x}", capabilities=["accel", "gyro"],
                config={"driver": "mpu6050"}
            )
        return None

    def detect_cameras_linux(self):
        # OAK via depthai module
        try:
            import depthai as dai
            for dev in dai.Device.getAllAvailableDevices():
                if not any("OAK" in d.name for d in self.detected_devices):
                    self.detected_devices.append(HardwareDevice(
                        device_type="depth_camera", name="OAK Camera (API)", vendor="Luxonis",
                        interface="usb3", capabilities=["rgb", "depth", "ai"], config={"driver": "depthai"}
                    ))
        except Exception: pass

        # V4L2
        try:
            import shutil
            if shutil.which('v4l2-ctl'):
                res = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
                if res.returncode == 0 and '/dev/video' in res.stdout:
                    if not any(d.device_type == "camera" for d in self.detected_devices):
                        self.detected_devices.append(HardwareDevice(
                            device_type="camera", name="Webcam", vendor="Generic",
                            interface="usb", address="/dev/video0", capabilities=["rgb"]
                        ))
        except Exception: pass

    def detect_cameras_windows(self):
        """Check for cameras on Windows via common APIs."""
        # Check DepthAI (OAK)
        try:
            import depthai as dai
            devices = dai.Device.getAllAvailableDevices()
            if devices:
                self.detected_devices.append(HardwareDevice(
                    device_type="depth_camera", name="OAK Camera (Windows)", vendor="Luxonis",
                    interface="usb", capabilities=["rgb", "depth", "ai"], config={"driver": "depthai"}
                ))
                print("✅ Found: OAK Camera (DepthAI Windows)")
        except Exception: pass

    def detect_accelerators_linux(self):
        try:
            res = subprocess.run(['lspci'], capture_output=True, text=True)
            if res.returncode == 0 and 'hailo' in res.stdout.lower():
                self.detected_devices.append(HardwareDevice(
                    device_type="ai_accelerator", name="Hailo-8", vendor="Hailo",
                    interface="pcie", capabilities=["ai", "inference"], config={"driver": "hailo_platform"}
                ))
                print("✅ Found: Hailo-8 AI Accelerator")
        except Exception: pass

    def detect_hats(self):
        try:
            if Path("/proc/device-tree/hat/product").exists():
                name = Path("/proc/device-tree/hat/product").read_text().strip('\x00')
                self.detected_devices.append(HardwareDevice(
                    device_type="hat", name=name, vendor="Unknown", interface="gpio"
                ))
        except Exception: pass

    def detect_gpio_devices(self): pass

    def detect_desktop_peripherals(self):
        # Mouse/Keyboard check (platform agnostic via simple existence)
        self.detected_devices.append(HardwareDevice(
            device_type="human_interface", name="System Keyboard/Mouse", vendor="Generic", 
            interface="system", capabilities=["input"]
        ))

    def _inject_mocks(self):
        """Injected virtual hardware for development mode."""
        print("🛠️  Injecting Mock Hardware for development environment...")
        
        # Virtual Depth Camera
        self.detected_devices.append(HardwareDevice(
            device_type="depth_camera", name="MOCK OAK-D", vendor="Luxonis (Virtual)",
            interface="virtual", capabilities=["rgb", "depth", "ai"], is_mock=True,
            config={"driver": "mock_depthai"}
        ))
        
        # Virtual Servo Controller
        self.detected_devices.append(HardwareDevice(
            device_type="servo_controller", name="MOCK PCA9685", vendor="Adafruit (Virtual)",
            interface="virtual", address="0x40", capabilities=["pwm", "servo"], is_mock=True,
            config={"driver": "mock_servokit"}
        ))
        
        # Virtual IMU
        self.detected_devices.append(HardwareDevice(
            device_type="imu", name="MOCK IMU", vendor="Generic (Virtual)",
            interface="virtual", capabilities=["accel", "gyro"], is_mock=True
        ))

    def generate_config(self) -> Dict[str, Any]:
        config = {
            "hardware_profile": "auto_detected",
            "platform": self.platform_info["os"].lower(),
            "detected_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "devices": {},
        }
        for device in self.detected_devices:
            dtype = device.device_type
            if dtype not in config["devices"]:
                config["devices"][dtype] = []
            config["devices"][dtype].append(asdict(device))
        
        config["primary"] = {}
        # Depth Cam
        depth_cams = [d for d in self.detected_devices if d.device_type == "depth_camera"]
        if depth_cams:
            config["primary"]["depth_camera"] = depth_cams[0].name
            config["primary"]["depth_camera_driver"] = depth_cams[0].config.get("driver")
            config["primary"]["depth_camera_is_mock"] = depth_cams[0].is_mock

        # Servo
        servos = [d for d in self.detected_devices if d.device_type == "servo_controller"]
        if servos:
            config["primary"]["servo_controller"] = servos[0].name
            config["primary"]["servo_controller_is_mock"] = servos[0].is_mock

        return config

    def save_config(self, output_path: str = "/tmp/hardware_config.json"):
        config = self.generate_config()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Configuration saved to: {out}")
        return out

    def print_summary(self):
        """Print a human-readable summary of detected hardware."""
        print("\n📋 Hardware Detection Summary:")
        print("-" * 40)
        for device in self.detected_devices:
            status = "✅" if not device.is_mock else "🔧 (mock)"
            print(f"  {status} {device.name} ({device.device_type})")
        if not self.detected_devices:
            print("  No devices detected")
        print("-" * 40)


def main():
    detector = HardwareDetector()
    detector.detect_all()
    config = detector.generate_config()
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()