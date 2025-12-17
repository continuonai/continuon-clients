"""
Hardware detection and auto-configuration for Pi5 robot systems.
Detects cameras, HATs, servos, and accessories automatically.
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
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.config is None:
            self.config = {}


class HardwareDetector:
    """
    Auto-detects connected hardware on Raspberry Pi 5.
    Identifies cameras, HATs, servo controllers, and other accessories.
    """
    
    def __init__(self):
        self.detected_devices: List[HardwareDevice] = []
        self.platform_info: Dict[str, Any] = {}
        self.missing_dependencies: List[str] = []
        self.hailo_offload_enabled = os.environ.get("CONTINUON_HAILO_OFFLOAD", "1").lower() in ("1", "true", "yes")
    
    def detect_all(self, auto_install: bool = False, allow_system_install: bool = False) -> List[HardwareDevice]:
        """Run all detection methods and return found devices.

        Args:
            auto_install: If True, attempt to install missing Python drivers/libraries
                          based on the current OS before detecting hardware.
            allow_system_install: If True, allow SystemInstaller to run OS-level package manager
                                  (gated by CONTINUON_ALLOW_SYSTEM_INSTALL=1).
        """
        self.detected_devices = []
        self._detect_environment()
        
        print("Scanning for hardware devices...\n")
        
        # Detect in order of interface type
        self.detect_usb_devices()
        self.detect_i2c_devices()
        self.detect_cameras()
        self.detect_hats()
        self.detect_accelerators()
        self.detect_gpio_devices()
        self.detect_desktop_peripherals()

        if auto_install and self.missing_dependencies:
            self._auto_install_missing(allow_system_install=allow_system_install)
            # Re-check so we only nag for remaining missing deps
            self.missing_dependencies = [
                dep for dep in self.missing_dependencies if not self._has_module(dep.split()[0])
            ]

        if self.missing_dependencies:
            print("⚠️  Missing optional drivers/libs:")
            for dep in self.missing_dependencies:
                print(f"   - {dep}")
            print("   Install these for full hardware support on this OS.")
            print()

        return self.detected_devices

    def _detect_environment(self) -> None:
        """Record platform info and gather dependency hints per OS."""
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

        # Raspberry Pi specific note
        if system == "Linux":
            try:
                with open("/etc/os-release", "r", encoding="utf-8") as f:
                    os_release = f.read().lower()
                if "raspbian" in os_release or "debian" in os_release:
                    self.platform_info["distro"] = "Raspberry Pi OS"
            except Exception:
                pass
        
        return self.detected_devices

    def _has_module(self, name: str) -> bool:
        """Return True if the python module is importable."""
        return importlib.util.find_spec(name) is not None

    def _auto_install_missing(self, allow_system_install: bool = False) -> None:
        """Attempt best-effort installation of missing Python dependencies."""
        print("🔧 Attempting to install missing Python drivers...")
        unresolved: List[str] = []
        for dep in list(self.missing_dependencies):
            # Expect dep string like "depthai (hint: pip install depthai ...)"
            parts = dep.split("(hint:")
            if len(parts) < 2:
                unresolved.append(dep)
                continue
            hint = parts[1].strip().rstrip(")")
            if not hint.startswith("pip "):
                unresolved.append(dep)
                continue  # Only auto-run pip hints to avoid system package risk

            try:
                cmd_tokens = shlex.split(hint)
                if cmd_tokens[0] == "pip":
                    cmd_tokens = [sys.executable, "-m", "pip"] + cmd_tokens[1:]
                print(f"   → {' '.join(cmd_tokens)}")
                subprocess.run(cmd_tokens, check=False)
            except Exception as exc:
                print(f"   ⚠️  Install attempt failed for {dep}: {exc}")
                unresolved.append(dep)

        # Optional system-level install for any unresolved deps
        if allow_system_install and unresolved:
            installer = SystemInstaller()
            pkg_names = [u.split()[0] for u in unresolved]
            plans = installer.install_packages(pkg_names, allow_run=True)
            for plan in plans:
                msg = f"System install via {plan.manager}: {' '.join(plan.command)} -> {plan.message}"
                print(msg)
    
    def detect_usb_devices(self):
        """Detect USB cameras and devices."""
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            if result.returncode != 0:
                return
            
            for line in result.stdout.strip().split('\n'):
                # Parse lsusb output: Bus XXX Device XXX: ID vendor:product name
                match = re.match(r'Bus (\d+) Device (\d+): ID ([0-9a-f]{4}):([0-9a-f]{4}) (.+)', line)
                if not match:
                    continue
                
                bus, device, vendor_id, product_id, name = match.groups()
                
                # Detect OAK cameras (Luxonis/Intel Movidius)
                if vendor_id == '03e7':  # Luxonis vendor ID
                    device_name = "OAK-D Lite" if "MyriadX" in name else "OAK Camera"
                    self.detected_devices.append(HardwareDevice(
                        device_type="depth_camera",
                        name=device_name,
                        vendor="Luxonis",
                        interface="usb3",
                        address=f"bus_{bus}_dev_{device}",
                        capabilities=["rgb", "depth", "stereo", "ai"],
                        config={
                            "vendor_id": vendor_id,
                            "product_id": product_id,
                            "driver": "depthai",
                            "hailo_offload": self.hailo_offload_enabled,
                        }
                    ))
                    print(f"✅ Found: {device_name} (USB 3.0)")
                
                # Detect Intel RealSense cameras
                elif vendor_id == '8086' and 'RealSense' in name:
                    self.detected_devices.append(HardwareDevice(
                        device_type="depth_camera",
                        name=name.split('RealSense')[1].strip(),
                        vendor="Intel",
                        interface="usb3",
                        address=f"bus_{bus}_dev_{device}",
                        capabilities=["rgb", "depth", "imu"],
                        config={
                            "vendor_id": vendor_id,
                            "product_id": product_id,
                            "driver": "librealsense2",
                        }
                    ))
                    print(f"✅ Found: Intel RealSense {name} (USB 3.0)")
                
                # Detect generic USB cameras
                elif any(keyword in name.lower() for keyword in ['camera', 'webcam', 'video']):
                    self.detected_devices.append(HardwareDevice(
                        device_type="camera",
                        name=name,
                        vendor="Generic",
                        interface="usb",
                        address=f"bus_{bus}_dev_{device}",
                        capabilities=["rgb"],
                        config={
                            "vendor_id": vendor_id,
                            "product_id": product_id,
                            "driver": "uvc",
                        }
                    ))
                    print(f"✅ Found: {name} (USB)")
        
        except Exception as e:
            print(f"Warning: USB detection failed: {e}")
    
    def detect_i2c_devices(self):
        """Detect I2C devices including servo controllers and sensors."""
        try:
            # Scan I2C bus 1 (standard Pi bus)
            result = subprocess.run(['i2cdetect', '-y', '1'], capture_output=True, text=True)
            if result.returncode != 0:
                return
            
            # Parse i2cdetect grid output
            addresses = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                for addr in parts[1:]:
                    if addr != '--' and addr != 'UU':
                        try:
                            addresses.append(int(addr, 16))
                        except ValueError:
                            continue
            
            # Identify devices by address
            for addr in addresses:
                device = self._identify_i2c_device(addr)
                if device:
                    self.detected_devices.append(device)
                    print(f"✅ Found: {device.name} at I2C address 0x{addr:02x}")
        
        except Exception as e:
            print(f"Warning: I2C detection failed: {e}")
    
    def _identify_i2c_device(self, address: int) -> Optional[HardwareDevice]:
        """Identify I2C device by address."""
        # PCA9685 16-channel PWM/servo controller
        if address == 0x40:
            return HardwareDevice(
                device_type="servo_controller",
                name="PCA9685 16-Channel PWM",
                vendor="Adafruit/NXP",
                interface="i2c",
                address=f"0x{address:02x}",
                capabilities=["pwm", "servo", "led"],
                config={
                    "channels": 16,
                    "frequency_hz": 50,
                    "driver": "adafruit_servokit",
                }
            )
        
        # MPU6050 IMU
        elif address == 0x68 or address == 0x69:
            return HardwareDevice(
                device_type="imu",
                name="MPU6050 IMU",
                vendor="InvenSense",
                interface="i2c",
                address=f"0x{address:02x}",
                capabilities=["accel", "gyro", "temp"],
                config={
                    "accel_range": "±2g",
                    "gyro_range": "±250°/s",
                    "driver": "mpu6050",
                }
            )
        
        # BNO055 9-axis IMU
        elif address == 0x28 or address == 0x29:
            return HardwareDevice(
                device_type="imu",
                name="BNO055 9-Axis IMU",
                vendor="Bosch",
                interface="i2c",
                address=f"0x{address:02x}",
                capabilities=["accel", "gyro", "mag", "fusion"],
                config={
                    "fusion": True,
                    "driver": "adafruit_bno055",
                }
            )
        
        # INA219 Current/Power Monitor
        elif address == 0x41:
            return HardwareDevice(
                device_type="power_monitor",
                name="INA219 Power Monitor",
                vendor="Texas Instruments",
                interface="i2c",
                address=f"0x{address:02x}",
                capabilities=["current", "voltage", "power"],
                config={"driver": "ina219"}
            )
        
        # Generic I2C device
        else:
            return HardwareDevice(
                device_type="i2c_device",
                name=f"Unknown I2C Device",
                vendor="Unknown",
                interface="i2c",
                address=f"0x{address:02x}",
                capabilities=[],
            )
    
    def detect_cameras(self):
        """Detect Video4Linux cameras."""
        try:
            import depthai as dai
            
            # Detect OAK cameras via DepthAI
            devices = dai.Device.getAllAvailableDevices()
            for dev_info in devices:
                # Skip if already detected via USB
                if any(d.device_type == "depth_camera" and "OAK" in d.name 
                       for d in self.detected_devices):
                    continue
                
                self.detected_devices.append(HardwareDevice(
                    device_type="depth_camera",
                    name="OAK Camera (DepthAI)",
                    vendor="Luxonis",
                    interface="usb3",
                    capabilities=["rgb", "depth", "stereo", "ai"],
                    config={"driver": "depthai"}
                ))
                print(f"✅ Found: OAK Camera via DepthAI")
        
        except ImportError:
            pass  # DepthAI not installed
        except Exception as e:
            print(f"Warning: DepthAI camera detection failed: {e}")
        
        # Detect V4L2 cameras
        try:
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                # Parse v4l2-ctl output for non-platform cameras
                current_device = None
                for line in result.stdout.strip().split('\n'):
                    if not line.startswith('\t') and not line.startswith(' '):
                        current_device = line.strip()
                    elif '/dev/video' in line and 'platform' not in (current_device or ''):
                        video_dev = line.strip()
                        # Skip if already detected
                        if not any(d.device_type in ["camera", "depth_camera"] 
                                  for d in self.detected_devices):
                            self.detected_devices.append(HardwareDevice(
                                device_type="camera",
                                name=current_device or "USB Camera",
                                vendor="Generic",
                                interface="usb",
                                address=video_dev,
                                capabilities=["rgb"],
                                config={"driver": "v4l2", "device": video_dev}
                            ))
                            print(f"✅ Found: {current_device} at {video_dev}")
                        break  # Only first video device per camera
        
        except Exception as e:
            print(f"Warning: V4L2 camera detection failed: {e}")
    
    def detect_accelerators(self):
        """Detect AI accelerators (Hailo, Coral, NPU)."""
        # Check for Hailo AI HAT+ (PCIe device)
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if 'hailo' in line.lower():
                        self.detected_devices.append(HardwareDevice(
                            device_type="ai_accelerator",
                            name="Hailo AI HAT+",
                            vendor="Hailo",
                            interface="pcie",
                            capabilities=["ai", "inference", "neural_network", "hailo8"],
                            config={
                                "model": "Hailo-8",
                                "tops": 26,
                                "driver": "hailo_platform",
                                "hailo_offload": self.hailo_offload_enabled,
                            }
                        ))
                        print(f"✅ Found: Hailo AI HAT+ (PCIe)")
        except Exception as e:
            print(f"Warning: PCIe accelerator detection failed: {e}")
            
        # Check for Google Coral (USB)
        # Already checked in scan_usb but let's be specific here if needed
        # Often appears as "Global Unichip Corp."
        try:
             result = subprocess.run(['lsusb'], capture_output=True, text=True)
             if "Google" in result.stdout or "Global Unichip" in result.stdout:
                 self.detected_devices.append(HardwareDevice(
                    device_type="ai_accelerator",
                    name="Coral USB Accelerator",
                    vendor="Google",
                    interface="usb",
                    capabilities=["ai", "inference", "tpu"],
                    config={"driver": "pycoral"}
                 ))
                 print("✅ Found: Google Coral USB Accelerator")
        except: pass

    def detect_hats(self):
        """Detect Raspberry Pi HATs via device tree."""
        # Check device tree for HAT EEPROM
        try:
            hat_vendor = Path("/proc/device-tree/hat/vendor")
            hat_product = Path("/proc/device-tree/hat/product")
            
            if hat_vendor.exists() and hat_product.exists():
                vendor = hat_vendor.read_text().strip('\x00')
                product = hat_product.read_text().strip('\x00')
                
                self.detected_devices.append(HardwareDevice(
                    device_type="hat",
                    name=product,
                    vendor=vendor,
                    interface="gpio",
                    capabilities=["gpio_header"],
                ))
                print(f"✅ Found HAT: {vendor} {product}")
        
        except Exception as e:
            print(f"Warning: HAT EEPROM detection failed: {e}")
    
    def detect_gpio_devices(self):
        """Detect devices connected to GPIO pins."""
        # This is harder to auto-detect without probing
        # Could check for known GPIO usage patterns
        pass

    def detect_desktop_peripherals(self):
        """Detect desktop-class peripherals (Monitor, KBM)."""
        import shutil
        
        # Check for Display (xrandr)
        if shutil.which("xrandr"):
            try:
                res = subprocess.run(["xrandr"], capture_output=True, text=True)
                if " connected" in res.stdout:
                    self.detected_devices.append(HardwareDevice(
                        device_type="display", name="Desktop Monitor", vendor="Generic", interface="hdmi/dp",
                        capabilities=["visual_output"]
                    ))
                    print("✅ Found: Desktop Monitor")
            except Exception:
                pass
        
        # Check for Input Devices (simple check)
        # On Linux, /proc/bus/input/devices is good
        try:
            if Path("/proc/bus/input/devices").exists():
                text = Path("/proc/bus/input/devices").read_text()
                if "Handlers=mouse" in text:
                    self.detected_devices.append(HardwareDevice(
                        device_type="human_interface", name="Mouse", vendor="Generic", interface="usb/bt",
                        capabilities=["pointer_input"]
                    ))
                if "Handlers=kbd" in text or "sysrq" in text: # approximate
                    self.detected_devices.append(HardwareDevice(
                        device_type="human_interface", name="Keyboard", vendor="Generic", interface="usb/bt",
                        capabilities=["text_input"]
                    ))
        except Exception:
            pass
    
    def generate_config(self) -> Dict[str, Any]:
        """
        Generate hardware configuration for ContinuonBrain.
        
        Returns config dict with auto-detected settings.
        """
        config = {
            "hardware_profile": "auto_detected",
            "platform": "raspberry_pi_5",
            "detected_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "devices": {},
        }
        
        # Group devices by type
        for device in self.detected_devices:
            device_type = device.device_type
            if device_type not in config["devices"]:
                config["devices"][device_type] = []
            config["devices"][device_type].append(asdict(device))
        
        # Auto-select primary devices
        config["primary"] = {}
        
        # Primary depth camera (prefer OAK-D)
        depth_cams = [d for d in self.detected_devices if d.device_type == "depth_camera"]
        if depth_cams:
            oak_cams = [d for d in depth_cams if "OAK" in d.name]
            primary_cam = oak_cams[0] if oak_cams else depth_cams[0]
            config["primary"]["depth_camera"] = primary_cam.name
            config["primary"]["depth_camera_driver"] = primary_cam.config.get("driver")
        
        # Primary servo controller
        servo_controllers = [d for d in self.detected_devices 
                            if d.device_type == "servo_controller"]
        if servo_controllers:
            config["primary"]["servo_controller"] = servo_controllers[0].name
            config["primary"]["servo_controller_address"] = servo_controllers[0].address
        
        # AI accelerator
        ai_devices = [d for d in self.detected_devices 
                     if d.device_type == "ai_accelerator"]
        if ai_devices:
            config["primary"]["ai_accelerator"] = ai_devices[0].name
        
        return config
    
    def save_config(self, output_path: str = "/tmp/hardware_config.json"):
        """Save detected configuration to JSON file."""
        config = self.generate_config()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Configuration saved to: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print detection summary."""
        print(f"\n{'='*60}")
        print(f"Hardware Detection Summary")
        print(f"{'='*60}")
        print(f"Total devices found: {len(self.detected_devices)}\n")
        
        # Group by type
        by_type = {}
        for device in self.detected_devices:
            if device.device_type not in by_type:
                by_type[device.device_type] = []
            by_type[device.device_type].append(device)
        
        for device_type, devices in sorted(by_type.items()):
            print(f"{device_type.replace('_', ' ').title()}:")
            for device in devices:
                interface_str = f"({device.interface})"
                if device.address:
                    interface_str = f"({device.interface}: {device.address})"
                print(f"  • {device.name} {interface_str}")
                if device.capabilities:
                    print(f"    Capabilities: {', '.join(device.capabilities)}")
            print()
        
        print(f"{'='*60}\n")


def main():
    """Run hardware detection."""
    detector = HardwareDetector()
    devices = detector.detect_all()
    
    if not devices:
        print("⚠️  No hardware devices detected")
        print("\nTroubleshooting:")
        print("  • Ensure devices are connected and powered")
        print("  • Check I2C is enabled: sudo raspi-config")
        print("  • Verify USB devices: lsusb")
        print("  • Check I2C devices: i2cdetect -y 1")
        return
    
    detector.print_summary()
    
    # Save configuration
    config_file = detector.save_config()
    
    # Show auto-selected primaries
    config = detector.generate_config()
    if config["primary"]:
        print("🎯 Auto-selected primary devices:")
        for key, value in config["primary"].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print()
    
    print("💡 Next steps:")
    print(f"  1. Review config: cat {config_file}")
    print("  2. Use config in your application")
    print("  3. Run hardware-specific initialization")


if __name__ == "__main__":
    main()
