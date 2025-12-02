# Hardware Auto-Detection

ContinuonBrain OS includes intelligent hardware detection that automatically identifies and configures cameras, HATs, servo controllers, and other peripherals connected to your Raspberry Pi 5.

## Overview

The hardware detection system scans multiple interfaces (USB, I2C, PCIe, GPIO) to discover:
- **Depth Cameras**: OAK-D, Intel RealSense, generic USB cameras
- **Servo Controllers**: PCA9685 16-channel PWM boards
- **IMUs**: MPU6050, BNO055 9-axis sensors
- **AI Accelerators**: Hailo AI HAT+
- **Power Monitors**: INA219 current sensors
- **Other I2C/USB Devices**: Automatically cataloged

## Supported Devices

### Cameras

| Device | Interface | Capabilities | Auto-Detected |
|--------|-----------|--------------|---------------|
| OAK-D Lite | USB3 | RGB, Depth, Stereo, AI | ✅ |
| OAK-D | USB3 | RGB, Depth, Stereo, AI | ✅ |
| Intel RealSense | USB3 | RGB, Depth, IMU | ✅ |
| USB Webcam | USB | RGB | ✅ |
| Pi Camera Module | CSI | RGB | 🚧 Planned |

### Servo Controllers

| Device | Interface | Channels | Auto-Detected |
|--------|-----------|----------|---------------|
| PCA9685 | I2C (0x40) | 16 PWM | ✅ |

### AI Accelerators

| Device | Interface | Performance | Auto-Detected |
|--------|-----------|-------------|---------------|
| Hailo AI HAT+ | PCIe | 13 TOPS | ✅ |
| Coral USB | USB | 4 TOPS | 🚧 Planned |

### IMUs

| Device | Interface | Capabilities | Auto-Detected |
|--------|-----------|--------------|---------------|
| MPU6050 | I2C (0x68/0x69) | Accel, Gyro | ✅ |
| BNO055 | I2C (0x28/0x29) | 9-axis, Fusion | ✅ |

## Usage

### Standalone Detection

Run hardware detection as a standalone tool:

```bash
# Auto-detect all hardware
PYTHONPATH=$PWD python3 continuonbrain/sensors/hardware_detector.py

# Output:
# 🔍 Scanning for hardware devices...
# ✅ Found: OAK-D Lite (USB 3.0)
# ✅ Found: PCA9685 16-Channel PWM at I2C address 0x40
# 💾 Configuration saved to: /tmp/hardware_config.json
```

### In Python Code

```python
from continuonbrain.sensors.hardware_detector import HardwareDetector

# Detect all hardware
detector = HardwareDetector()
devices = detector.detect_all()

# Print summary
detector.print_summary()

# Generate config
config = detector.generate_config()

# Auto-selected primary devices
primary_camera = config["primary"]["depth_camera"]  # "OAK-D Lite"
primary_servo = config["primary"]["servo_controller"]  # "PCA9685 16-Channel PWM"

# Save for later use
detector.save_config("/opt/continuonos/brain/hardware_config.json")
```

### Integration Test

The integration test automatically detects hardware:

```bash
# Auto-detect and use available hardware
PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --real-hardware

# Only detect (don't run test)
PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --detect-only

# Disable auto-detection (use hardcoded config)
PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --no-auto-detect
```

### Mock Server

The mock Robot API server also supports auto-detection:

```bash
# Auto-detect hardware (prefers real, falls back to mock if controllers are missing)
PYTHONPATH=$PWD python3 -m continuonbrain.robot_api_server

# Force real detected hardware and fail fast when devices are absent
PYTHONPATH=$PWD python3 -m continuonbrain.robot_api_server --real-hardware

# Force mock mode
PYTHONPATH=$PWD python3 -m continuonbrain.robot_api_server --mock-hardware

# Disable auto-detection
PYTHONPATH=$PWD python3 -m continuonbrain.robot_api_server --no-auto-detect
```

## Configuration Format

Auto-detection generates a JSON config:

```json
{
  "hardware_profile": "auto_detected",
  "platform": "raspberry_pi_5",
  "detected_timestamp": "2025-12-01 15:43:01",
  "devices": {
    "depth_camera": [
      {
        "device_type": "depth_camera",
        "name": "OAK-D Lite",
        "vendor": "Luxonis",
        "interface": "usb3",
        "address": "bus_001_dev_008",
        "capabilities": ["rgb", "depth", "stereo", "ai"],
        "config": {
          "vendor_id": "03e7",
          "product_id": "2485",
          "driver": "depthai"
        }
      }
    ],
    "servo_controller": [
      {
        "device_type": "servo_controller",
        "name": "PCA9685 16-Channel PWM",
        "vendor": "Adafruit/NXP",
        "interface": "i2c",
        "address": "0x40",
        "capabilities": ["pwm", "servo", "led"],
        "config": {
          "channels": 16,
          "frequency_hz": 50,
          "driver": "adafruit_servokit"
        }
      }
    ]
  },
  "primary": {
    "depth_camera": "OAK-D Lite",
    "depth_camera_driver": "depthai",
    "servo_controller": "PCA9685 16-Channel PWM",
    "servo_controller_address": "0x40"
  }
}
```

## Detection Methods

### USB Devices

Scans `lsusb` output for known vendor IDs:
- **Luxonis** (0x03e7): OAK-D cameras
- **Intel** (0x8086): RealSense cameras
- Generic camera keywords

### I2C Devices

Scans `i2cdetect -y 1` for known addresses:
- **0x40**: PCA9685 servo controller
- **0x68/0x69**: MPU6050 IMU
- **0x28/0x29**: BNO055 IMU
- **0x41**: INA219 power monitor

### PCIe Devices

Scans `lspci` for known devices:
- Hailo AI accelerators

### Device Tree

Reads `/proc/device-tree/hat/*` for HAT EEPROM data.

## Capabilities

Each detected device includes capability flags:

| Capability | Description |
|------------|-------------|
| `rgb` | RGB image capture |
| `depth` | Depth sensing |
| `stereo` | Stereo vision |
| `ai` | Onboard AI inference |
| `accel` | Accelerometer |
| `gyro` | Gyroscope |
| `mag` | Magnetometer |
| `fusion` | Sensor fusion |
| `pwm` | PWM signal generation |
| `servo` | Servo motor control |

## Primary Device Selection

Auto-detection automatically selects primary devices for common use cases:

1. **Primary Depth Camera**: Prefers OAK-D over generic cameras
2. **Primary Servo Controller**: First detected PCA9685
3. **Primary AI Accelerator**: First detected Hailo/Coral device

Access via:
```python
config["primary"]["depth_camera"]  # Device name
config["primary"]["depth_camera_driver"]  # Driver name
```

## Troubleshooting

### No Devices Detected

```bash
# Check I2C is enabled
sudo raspi-config
# Interface Options → I2C → Enable

# Verify USB devices
lsusb

# Verify I2C devices
i2cdetect -y 1
```

### Permission Issues

```bash
# Add user to i2c group
sudo usermod -a -G i2c $USER

# Add udev rules for OAK-D
sudo cp /etc/udev/rules.d/80-movidius.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Unknown I2C Device

The detector will log unknown I2C devices. To identify:

1. Check device datasheet for default I2C address
2. Add device to `_identify_i2c_device()` in `hardware_detector.py`
3. Submit a PR to add support!

## Extending Detection

To add support for new hardware:

### New I2C Device

Edit `continuonbrain/sensors/hardware_detector.py`:

```python
def _identify_i2c_device(self, address: int) -> Optional[HardwareDevice]:
    # Add your device
    if address == 0x42:  # Your device's I2C address
        return HardwareDevice(
            device_type="your_device_type",
            name="Your Device Name",
            vendor="Vendor",
            interface="i2c",
            address=f"0x{address:02x}",
            capabilities=["cap1", "cap2"],
            config={
                "driver": "your_driver",
                "setting": "value"
            }
        )
```

### New USB Device

Add vendor/product ID detection in `detect_usb_devices()`:

```python
if vendor_id == 'your_vendor_id':
    self.detected_devices.append(HardwareDevice(
        device_type="your_device",
        name="Your Device",
        vendor="Vendor",
        interface="usb",
        # ...
    ))
```

## Future Roadmap

- [ ] CSI camera detection (Pi Camera Module)
- [ ] SPI device detection
- [ ] GPIO pin mapping detection
- [ ] Automatic driver installation
- [ ] Device capability testing
- [ ] Hardware compatibility validation
- [ ] Multi-camera configuration
- [ ] Device health monitoring

## See Also

- [Pi5 Car Readiness](../continuonbrain/PI5_CAR_READINESS.md) - Initial hardware setup
- [RLDS Schema](rlds-schema.md) - Episode recording format
- [System Architecture](system-architecture.md) - Overall system design
