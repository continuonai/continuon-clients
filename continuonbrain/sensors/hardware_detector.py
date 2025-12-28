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
        """Parse USB device and identify known inference-capable hardware."""
        
        # OAK cameras (Luxonis) - VID 03e7 is Movidius/Intel Myriad
        if vid == '03e7':
            # Identify specific OAK model by PID
            oak_models = {
                '2485': ('OAK-D', ['rgb', 'depth', 'stereo', 'ai', 'spatial_ai']),
                'f63b': ('OAK-D Lite', ['rgb', 'depth', 'stereo', 'ai']),
                'f63c': ('OAK-D Pro', ['rgb', 'depth', 'stereo', 'ai', 'ir_flood', 'ir_dot']),
                'f63d': ('OAK-D S2', ['rgb', 'depth', 'stereo', 'ai', 'poe']),
                'f63e': ('OAK-D Pro W', ['rgb', 'depth', 'stereo', 'ai', 'wide_fov', 'ir_flood']),
                'f63f': ('OAK-D LR', ['rgb', 'depth', 'stereo', 'ai', 'long_range']),
                '2150': ('OAK-1', ['rgb', 'ai']),
                'f64a': ('OAK-1 Lite', ['rgb', 'ai']),
            }
            model_info = oak_models.get(pid, ('OAK Camera', ['rgb', 'depth', 'ai']))
            device_name, capabilities = model_info
            
            self.detected_devices.append(HardwareDevice(
                device_type="depth_camera", name=device_name, vendor="Luxonis",
                interface="usb3", address=address, capabilities=capabilities,
                config={
                    "vendor_id": vid, "product_id": pid, "driver": "depthai", 
                    "hailo_offload": self.hailo_offload_enabled,
                    "inference_capable": True,
                    "onboard_vpu": "Myriad X",
                    "recommended_tasks": ["object_detection", "depth_estimation", "pose_estimation", "segmentation"]
                }
            ))
            print(f"✅ Found: {device_name} (USB) - Onboard VPU: Myriad X")
        
        # Intel RealSense cameras
        elif vid == '8086':
            realsense_models = {
                '0b07': ('RealSense D435', ['rgb', 'depth', 'stereo']),
                '0b5c': ('RealSense D435i', ['rgb', 'depth', 'stereo', 'imu']),
                '0b3a': ('RealSense D415', ['rgb', 'depth', 'stereo']),
                '0af6': ('RealSense D435', ['rgb', 'depth', 'stereo']),
                '0b64': ('RealSense D455', ['rgb', 'depth', 'stereo', 'imu', 'wide_baseline']),
                '0ade': ('RealSense D405', ['rgb', 'depth', 'stereo', 'close_range']),
                '0aba': ('RealSense L515', ['rgb', 'depth', 'lidar', 'imu']),
                '0b55': ('RealSense T265', ['stereo', 'imu', 'slam', 'tracking']),
            }
            if 'RealSense' in name or pid in realsense_models:
                model_info = realsense_models.get(pid, ('RealSense', ['rgb', 'depth']))
                device_name, capabilities = model_info
                self.detected_devices.append(HardwareDevice(
                    device_type="depth_camera", name=device_name, vendor="Intel",
                    interface="usb3", address=address, capabilities=capabilities,
                    config={
                        "vendor_id": vid, "product_id": pid, "driver": "librealsense2",
                        "inference_capable": False,
                        "recommended_tasks": ["depth_estimation", "slam", "3d_reconstruction"]
                    }
                ))
                print(f"✅ Found: {device_name} (USB)")
        
        # Google Coral USB Accelerator (VID 1a6e or 18d1)
        elif vid in ('1a6e', '18d1') and ('Coral' in name or 'Edge TPU' in name or pid in ('089a', '9302')):
            self.detected_devices.append(HardwareDevice(
                device_type="ai_accelerator", name="Google Coral USB", vendor="Google",
                interface="usb3", address=address, capabilities=["ai", "inference", "tpu", "edge_tpu"],
                config={
                    "vendor_id": vid, "product_id": pid, "driver": "pycoral",
                    "inference_capable": True,
                    "accelerator_type": "Edge TPU",
                    "tops": 4.0,  # 4 TOPS
                    "recommended_tasks": ["object_detection", "classification", "pose_estimation"]
                }
            ))
            print(f"✅ Found: Google Coral USB Accelerator (4 TOPS)")
        
        # Intel Neural Compute Stick 2 (NCS2) - VID 03e7 with specific PIDs
        elif vid == '03e7' and pid in ('2485', '2150'):
            # Already handled by OAK detection above, but standalone NCS2 check
            pass
        
        # Intel Movidius Neural Compute Stick (original)
        elif vid == '03e7' and pid == '2150':
            self.detected_devices.append(HardwareDevice(
                device_type="ai_accelerator", name="Intel NCS2", vendor="Intel",
                interface="usb3", address=address, capabilities=["ai", "inference", "vpu"],
                config={
                    "vendor_id": vid, "product_id": pid, "driver": "openvino",
                    "inference_capable": True,
                    "accelerator_type": "Myriad X VPU",
                    "recommended_tasks": ["object_detection", "classification"]
                }
            ))
            print(f"✅ Found: Intel Neural Compute Stick 2")
        
        # Stereolabs ZED cameras
        elif vid == '2b03':
            zed_models = {
                'f580': ('ZED', ['rgb', 'depth', 'stereo']),
                'f582': ('ZED 2', ['rgb', 'depth', 'stereo', 'imu', 'barometer']),
                'f583': ('ZED Mini', ['rgb', 'depth', 'stereo', 'imu']),
                'f681': ('ZED 2i', ['rgb', 'depth', 'stereo', 'imu', 'ip65']),
                'f780': ('ZED X', ['rgb', 'depth', 'stereo', 'imu', 'gmsl2']),
            }
            model_info = zed_models.get(pid, ('ZED Camera', ['rgb', 'depth', 'stereo']))
            device_name, capabilities = model_info
            self.detected_devices.append(HardwareDevice(
                device_type="depth_camera", name=device_name, vendor="Stereolabs",
                interface="usb3", address=address, capabilities=capabilities,
                config={
                    "vendor_id": vid, "product_id": pid, "driver": "pyzed",
                    "inference_capable": False,
                    "recommended_tasks": ["depth_estimation", "slam", "3d_reconstruction", "positional_tracking"]
                }
            ))
            print(f"✅ Found: {device_name} (USB)")
        
        # Orbbec depth cameras
        elif vid == '2bc5':
            self.detected_devices.append(HardwareDevice(
                device_type="depth_camera", name="Orbbec Camera", vendor="Orbbec",
                interface="usb3", address=address, capabilities=["rgb", "depth", "stereo"],
                config={
                    "vendor_id": vid, "product_id": pid, "driver": "pyorbbecsdk",
                    "recommended_tasks": ["depth_estimation", "3d_scanning"]
                }
            ))
            print(f"✅ Found: Orbbec Depth Camera (USB)")

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
        """Detect PCIe/M.2 AI accelerators on Linux."""
        self._detect_hailo_accelerator()
        self._detect_coral_pcie()
        self._detect_nvidia_gpu()
    
    def _detect_hailo_accelerator(self):
        """Detect Hailo-8/8L PCIe accelerator with detailed info."""
        hailo_found = False
        hailo_info = {"driver": "hailo_platform"}
        
        # Check via lspci
        try:
            import shutil
            if shutil.which('lspci'):
                res = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
                if res.returncode == 0:
                    for line in res.stdout.split('\n'):
                        line_lower = line.lower()
                        if 'hailo' in line_lower:
                            hailo_found = True
                            # Extract PCI address first
                            pci_addr = line.split()[0] if line else None
                            hailo_info["pci_address"] = pci_addr
                            
                            # Identify model - Hailo-8L is explicitly labeled, otherwise it's Hailo-8
                            # Device IDs: Hailo-8 = 1e60:2864, Hailo-8L = 1e60:2862
                            if 'hailo-8l' in line_lower or '1e60:2862' in line_lower:
                                hailo_info["model"] = "Hailo-8L"
                                hailo_info["tops"] = 13.0
                            else:
                                # Hailo-8 (full version) - 26 TOPS
                                hailo_info["model"] = "Hailo-8"
                                hailo_info["tops"] = 26.0
                            break
        except Exception:
            pass
        
        # Check via /dev/hailo*
        if not hailo_found:
            try:
                hailo_devs = list(Path("/dev").glob("hailo*"))
                if hailo_devs:
                    hailo_found = True
                    hailo_info["device_path"] = str(hailo_devs[0])
            except Exception:
                pass
        
        # Check Hailo Runtime status via hailortcli
        if hailo_found:
            try:
                import shutil
                if shutil.which('hailortcli'):
                    res = subprocess.run(['hailortcli', 'fw-control', 'identify'], 
                                        capture_output=True, text=True, timeout=5)
                    if res.returncode == 0:
                        hailo_info["runtime_status"] = "active"
                        # Parse firmware version if available
                        for line in res.stdout.split('\n'):
                            if 'firmware' in line.lower():
                                hailo_info["firmware_version"] = line.split(':')[-1].strip()
                            if 'device' in line.lower() and 'architecture' in line.lower():
                                if '8l' in line.lower():
                                    hailo_info["model"] = "Hailo-8L"
                                    hailo_info["tops"] = 13.0
            except Exception:
                hailo_info["runtime_status"] = "driver_loaded"
        
        if hailo_found:
            model_name = hailo_info.get("model", "Hailo-8")
            tops = hailo_info.get("tops", 26.0)
            
            self.detected_devices.append(HardwareDevice(
                device_type="ai_accelerator", name=model_name, vendor="Hailo",
                interface="pcie", address=hailo_info.get("pci_address"),
                capabilities=["ai", "inference", "npu", "object_detection", "segmentation", "pose"],
                config={
                    **hailo_info,
                    "inference_capable": True,
                    "accelerator_type": "NPU",
                    "hef_support": True,
                    "recommended_tasks": [
                        "object_detection", "pose_estimation", "semantic_segmentation",
                        "instance_segmentation", "face_detection", "license_plate_recognition"
                    ]
                }
            ))
            print(f"✅ Found: {model_name} AI Accelerator ({tops} TOPS) - PCIe")
    
    def _detect_coral_pcie(self):
        """Detect Google Coral PCIe/M.2 accelerator."""
        try:
            import shutil
            if not shutil.which('lspci'):
                return
            
            res = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
            if res.returncode == 0:
                for line in res.stdout.lower().split('\n'):
                    # Coral PCIe uses Global Unichip Corp vendor ID
                    if '1ac1:089a' in line or 'coral' in line or 'edge tpu' in line:
                        pci_addr = line.split()[0] if line else None
                        self.detected_devices.append(HardwareDevice(
                            device_type="ai_accelerator", name="Google Coral PCIe/M.2", vendor="Google",
                            interface="pcie", address=pci_addr,
                            capabilities=["ai", "inference", "tpu", "edge_tpu"],
                            config={
                                "driver": "gasket",
                                "inference_capable": True,
                                "accelerator_type": "Edge TPU",
                                "tops": 4.0,
                                "recommended_tasks": ["object_detection", "classification", "pose_estimation"]
                            }
                        ))
                        print("✅ Found: Google Coral PCIe/M.2 Accelerator (4 TOPS)")
                        break
        except Exception:
            pass
    
    def _detect_nvidia_gpu(self):
        """Detect NVIDIA GPUs for CUDA inference."""
        try:
            import shutil
            if not shutil.which('nvidia-smi'):
                return
            
            res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                 '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                for line in res.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpu_name, vram_mb, driver_ver = parts[0], parts[1], parts[2]
                        try:
                            vram_gb = float(vram_mb) / 1024
                        except ValueError:
                            vram_gb = 0
                        
                        self.detected_devices.append(HardwareDevice(
                            device_type="gpu", name=gpu_name, vendor="NVIDIA",
                            interface="pcie", capabilities=["cuda", "ai", "inference", "training"],
                            config={
                                "driver_version": driver_ver,
                                "vram_gb": round(vram_gb, 1),
                                "inference_capable": True,
                                "training_capable": True,
                                "accelerator_type": "CUDA GPU",
                                "recommended_tasks": [
                                    "object_detection", "llm_inference", "image_generation",
                                    "training", "fine_tuning", "video_processing"
                                ]
                            }
                        ))
                        print(f"✅ Found: {gpu_name} ({vram_gb:.1f}GB VRAM) - CUDA GPU")
        except Exception:
            pass

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
        print("\n" + "=" * 60)
        print("📋 Hardware Detection Summary")
        print("=" * 60)
        
        # Group by device type
        by_type = {}
        for device in self.detected_devices:
            dtype = device.device_type
            if dtype not in by_type:
                by_type[dtype] = []
            by_type[dtype].append(device)
        
        # Display order
        type_order = ["gpu", "ai_accelerator", "depth_camera", "camera", "servo_controller", "imu", "hat", "human_interface"]
        type_labels = {
            "gpu": "🎮 GPUs",
            "ai_accelerator": "🧠 AI Accelerators",
            "depth_camera": "📷 Depth Cameras",
            "camera": "📹 Cameras",
            "servo_controller": "🦾 Servo Controllers",
            "imu": "🧭 IMU Sensors",
            "hat": "🎩 HATs/Add-ons",
            "human_interface": "⌨️ Input Devices"
        }
        
        for dtype in type_order:
            if dtype in by_type:
                print(f"\n{type_labels.get(dtype, dtype)}:")
                for device in by_type[dtype]:
                    status = "✅" if not device.is_mock else "🔧 MOCK"
                    caps = ", ".join(device.capabilities[:4]) if device.capabilities else ""
                    tops = device.config.get("tops", "")
                    tops_str = f" [{tops} TOPS]" if tops else ""
                    vram = device.config.get("vram_gb", "")
                    vram_str = f" [{vram}GB VRAM]" if vram else ""
                    print(f"  {status} {device.name} ({device.vendor}){tops_str}{vram_str}")
                    if caps:
                        print(f"      Capabilities: {caps}")
        
        # Show devices not in order
        for dtype, devices in by_type.items():
            if dtype not in type_order:
                print(f"\n{dtype}:")
                for device in devices:
                    print(f"  ✅ {device.name}")
        
        if not self.detected_devices:
            print("  ⚠️ No devices detected")
        
        print("\n" + "=" * 60)
    
    def get_inference_recommendations(self) -> Dict[str, Any]:
        """Analyze detected hardware and recommend optimal inference configuration."""
        recommendations = {
            "primary_accelerator": None,
            "depth_camera": None,
            "inference_capable": False,
            "recommended_backend": "cpu",
            "supported_tasks": [],
            "configuration": {}
        }
        
        # Find best AI accelerator
        accelerators = [d for d in self.detected_devices if d.device_type in ("ai_accelerator", "gpu")]
        depth_cameras = [d for d in self.detected_devices if d.device_type == "depth_camera"]
        
        # Priority: NVIDIA GPU > Hailo-8 > Hailo-8L > Coral > OAK VPU
        for acc in accelerators:
            if acc.vendor == "NVIDIA":
                recommendations["primary_accelerator"] = acc.name
                recommendations["recommended_backend"] = "cuda"
                recommendations["inference_capable"] = True
                recommendations["configuration"]["cuda_device"] = 0
                recommendations["supported_tasks"] = acc.config.get("recommended_tasks", [])
                break
            elif acc.name in ("Hailo-8", "Hailo-8L"):
                recommendations["primary_accelerator"] = acc.name
                recommendations["recommended_backend"] = "hailo"
                recommendations["inference_capable"] = True
                recommendations["configuration"]["hailo_device"] = acc.config.get("device_path", "/dev/hailo0")
                recommendations["configuration"]["hef_path"] = "/opt/continuonos/brain/models/hailo"
                recommendations["supported_tasks"] = acc.config.get("recommended_tasks", [])
                break
            elif "Coral" in acc.name:
                recommendations["primary_accelerator"] = acc.name
                recommendations["recommended_backend"] = "edgetpu"
                recommendations["inference_capable"] = True
                recommendations["supported_tasks"] = acc.config.get("recommended_tasks", [])
                break
        
        # If no dedicated accelerator, check for OAK camera with onboard VPU
        if not recommendations["primary_accelerator"]:
            for cam in depth_cameras:
                if cam.vendor == "Luxonis" and cam.config.get("onboard_vpu"):
                    recommendations["primary_accelerator"] = f"{cam.name} (onboard VPU)"
                    recommendations["recommended_backend"] = "depthai"
                    recommendations["inference_capable"] = True
                    recommendations["supported_tasks"] = cam.config.get("recommended_tasks", [])
                    break
        
        # Set depth camera
        if depth_cameras:
            real_cams = [c for c in depth_cameras if not c.is_mock]
            if real_cams:
                recommendations["depth_camera"] = real_cams[0].name
                recommendations["configuration"]["depth_driver"] = real_cams[0].config.get("driver")
        
        # Fallback to CPU
        if not recommendations["inference_capable"]:
            recommendations["recommended_backend"] = "cpu"
            recommendations["supported_tasks"] = ["object_detection", "classification"]
            recommendations["configuration"]["warning"] = "No AI accelerator detected - using CPU inference (slower)"
        
        return recommendations
    
    def print_inference_recommendations(self):
        """Print inference configuration recommendations."""
        recs = self.get_inference_recommendations()
        
        print("\n" + "=" * 60)
        print("🎯 Inference Configuration Recommendations")
        print("=" * 60)
        
        if recs["inference_capable"]:
            print(f"  ✅ Hardware acceleration available!")
            print(f"  Primary Accelerator: {recs['primary_accelerator']}")
            print(f"  Recommended Backend: {recs['recommended_backend'].upper()}")
        else:
            print("  ⚠️ No AI accelerator detected")
            print("  Fallback: CPU inference (will be slower)")
        
        if recs["depth_camera"]:
            print(f"  Depth Camera: {recs['depth_camera']}")
        
        if recs["supported_tasks"]:
            print(f"\n  Supported Tasks:")
            for task in recs["supported_tasks"][:6]:
                print(f"    • {task.replace('_', ' ').title()}")
        
        if recs["configuration"]:
            print(f"\n  Configuration:")
            for key, val in recs["configuration"].items():
                if key != "warning":
                    print(f"    {key}: {val}")
            if "warning" in recs["configuration"]:
                print(f"\n  ⚠️ {recs['configuration']['warning']}")
        
        print("=" * 60)


def main():
    """Run hardware detection and print comprehensive report."""
    print("\n🔍 ContinuonBrain Hardware Auto-Detection")
    print("Scanning for inference-capable accessories...\n")
    
    detector = HardwareDetector()
    detector.detect_all(auto_install=False)
    
    # Print summaries
    detector.print_summary()
    detector.print_inference_recommendations()
    
    # Save config
    config = detector.generate_config()
    config["inference"] = detector.get_inference_recommendations()
    
    output_path = "/tmp/hardware_config.json"
    detector.save_config(output_path)
    
    # Also print JSON if verbose
    if os.environ.get("VERBOSE", "0") == "1":
        print("\n📄 Full Configuration (JSON):")
        print(json.dumps(config, indent=2))
    
    return detector


if __name__ == "__main__":
    main()