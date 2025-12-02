# Self-Charging Robot System

This document describes the hardware and software implementation for autonomous battery charging in the ContinuonBrain robot stack.

## Hardware Architecture

```
┌─────────────────────────┐
│  Charging Dock          │
│  - 12V/5A PSU           │
│  - Contact plates       │
│  - AprilTag ID 42       │
│  - Alignment guides     │
└─────────────────────────┘
         ↑ (pogo pins)
┌─────────────────────────┐
│  Battery Pack (3S LiPo) │
│  - 11.1V nominal        │
│  - 5000mAh capacity     │
│  - XT60 connectors      │
└─────────────────────────┘
    ↓                ↓
 Buck Conv.      Servo PSU
  5.1V/5A         6V/5A
    ↓                ↓
┌──────────┐   ┌──────────┐
│  Pi 5    │   │ Servos   │
└──────────┘   └──────────┘

┌─────────────────────────┐
│  INA219 Sensor          │ ← I2C addr 0x40
│  - Battery monitor      │    (shares bus with PCA9685)
└─────────────────────────┘
```

## Hardware Components Required

### Power System
1. **Battery Pack**: 3S LiPo (11.1V, 5000mAh+)
   - Provides ~45 min runtime at 10W average draw
   - Discharge to 9.9V minimum (3.3V/cell)
   - Example: Zeee 3S 5200mAh 50C LiPo

2. **Buck Converter (Pi 5)**: 12V → 5.1V/5A
   - Must handle 27W peak (Pi 5 + AI HAT+)
   - Example: DROK LM2596 DC-DC with display

3. **Buck Converter (Servos)**: 12V → 6V/5A
   - Powers PCA9685 + SO-ARM101 servos
   - Example: Pololu D24V50F6

4. **Battery Monitor**: INA219 or INA260 I2C module
   - Measures voltage, current, power
   - I2C address: 0x40 (default)
   - Example: Adafruit INA219 breakout

### Charging Dock
5. **Charging Station Base**:
   - 12V 5A wall adapter (60W)
   - Pogo pin contacts (spring-loaded, gold-plated)
   - TP4056 or BMS charging module
   - AprilTag marker (ID 42, 200mm printed on white)
   - Alignment funnel (3D printed guide rails)

6. **Pogo Pin Contacts**:
   - 6-8mm diameter spring pins
   - Mount on robot underside
   - Example: P75-E2 pogo pins (4-pack)

### Optional: Visual Homing
7. **Camera for Dock Detection**:
   - Already present: OAK-D depth camera
   - Detects AprilTag on dock for autonomous navigation
   - Or use Pi Camera v3 as fallback

## Software Components

### 1. Battery Monitor (`continuonbrain/sensors/battery_monitor.py`)
- Reads INA219 sensor via I2C
- Estimates charge percentage from voltage curve
- Detects charging state (current direction)
- Provides battery health diagnostics

**Usage:**
```python
from continuonbrain.sensors.battery_monitor import BatteryMonitor

monitor = BatteryMonitor(i2c_address=0x40, battery_capacity_mah=5000)
status = monitor.read_status()

print(f"Battery: {status.charge_percent:.1f}%")
print(f"Voltage: {status.voltage_v:.2f}V")
print(f"Charging: {status.is_charging}")
print(f"Time remaining: {status.time_to_empty_min:.0f} min")

if status.needs_charging():
    print("Battery low - initiate charging!")
```

### 2. Auto-Charge Behavior (`continuonbrain/behaviors/auto_charge.py`)
- State machine for autonomous docking
- Visual homing using AprilTag detection
- Alignment and charging verification
- Handles dock-not-found failures

**States:**
- `IDLE` → `LOW_BATTERY` (< 20%)
- `LOW_BATTERY` → `NAVIGATING_TO_DOCK`
- `NAVIGATING_TO_DOCK` → `ALIGNING_WITH_DOCK`
- `ALIGNING_WITH_DOCK` → `DOCKED`
- `DOCKED` → `CHARGING`
- `CHARGING` → `CHARGE_COMPLETE` (> 95%)
- `CHARGE_COMPLETE` → `IDLE` (undock)

**Usage:**
```python
from continuonbrain.sensors.battery_monitor import BatteryMonitor
from continuonbrain.behaviors.auto_charge import AutoChargeBehavior

battery = BatteryMonitor()
motors = MotorController()  # Your motor interface
charger = AutoChargeBehavior(battery, motors)

# Main control loop
while True:
    status = charger.update()
    print(f"State: {status.state.value}")
    time.sleep(0.5)
```

### 3. Robot Mode Integration (`continuonbrain/robot_modes.py`)
- Added `AUTO_CHARGING` mode to `RobotMode` enum
- Allows mode manager to track charging state
- Can interrupt other tasks when battery critical

### 4. System Health Check (`continuonbrain/system_health.py`)
- Battery check added to startup diagnostics
- Reports voltage, charge %, charging state
- Flags critical/warning battery levels

## Installation Steps

### Hardware Setup

1. **Install Battery Monitor:**
   ```bash
   # Connect INA219 to Pi GPIO:
   # VCC → 3.3V (Pin 1)
   # GND → GND (Pin 6)
   # SDA → GPIO 2 (Pin 3)
   # SCL → GPIO 3 (Pin 5)
   # VIN+ → Battery positive
   # VIN- → Through 0.1Ω shunt to load
   ```

2. **Wire Power Distribution:**
   ```
   Battery + → INA219 VIN+ → Buck Conv 1 IN+, Buck Conv 2 IN+
   Battery - → INA219 VIN- → Buck Conv 1 IN-, Buck Conv 2 IN-
   
   Buck Conv 1 OUT → Pi 5 USB-C (5.1V)
   Buck Conv 2 OUT → Servo power rail (6V)
   ```

3. **Mount Pogo Pins:**
   - Drill 4 holes in robot base (2 for +, 2 for -)
   - Install spring-loaded pogo pins
   - Wire to battery charging input (via BMS)

4. **Build Charging Dock:**
   - 3D print alignment guides (funnel shape)
   - Install mating contacts (flat copper pads)
   - Wire to 12V PSU through BMS/charging controller
   - Print AprilTag ID 42 (200mm square, laminate)

### Software Setup

1. **Install Python Dependencies:**
   ```bash
   pip install pi-ina219 adafruit-circuitpython-ina219
   ```

2. **Enable I2C:**
   ```bash
   sudo raspi-config
   # Interface Options → I2C → Enable
   sudo reboot
   ```

3. **Test Battery Monitor:**
   ```bash
   cd /home/craigm26/ContinuonXR/continuonbrain
   python sensors/battery_monitor.py
   ```

4. **Calibrate Voltage Curve** (optional):
   - Measure actual voltage at 100%, 75%, 50%, 25%, 10%
   - Update `BatteryMonitor._estimate_charge_percent()` with lookup table

5. **Test Auto-Charge (no motors):**
   ```bash
   python behaviors/auto_charge.py
   ```

6. **Integrate with Robot API:**
   Add to `robot_api_server.py`:
   ```python
   from continuonbrain.sensors.battery_monitor import BatteryMonitor
   from continuonbrain.behaviors.auto_charge import AutoChargeBehavior
   
   battery_monitor = BatteryMonitor()
   auto_charger = AutoChargeBehavior(battery_monitor, motor_controller)
   
   @app.get("/api/battery")
   def get_battery():
       status = battery_monitor.read_status()
       return status.__dict__ if status else {"error": "unavailable"}
   
   @app.get("/api/charge/status")
   def get_charge_status():
       return auto_charger._create_status(battery_monitor.read_status()).__dict__
   ```

## Integration with Existing Stack

### With VLA Policy (Autonomous Mode)
```python
# In autonomous control loop
battery_status = battery_monitor.read_status()

if auto_charger.should_interrupt_task(battery_status.charge_percent):
    # Battery critical - override policy
    mode_manager.set_mode(RobotMode.AUTO_CHARGING)
    while auto_charger.state != ChargingState.IDLE:
        auto_charger.update()
        time.sleep(0.5)
    mode_manager.set_mode(RobotMode.AUTONOMOUS)
```

### With Sleep Learning
```python
# Before entering sleep mode, check battery
if battery_status.charge_percent < 30:
    # Charge before training
    mode_manager.set_mode(RobotMode.AUTO_CHARGING)
    # Wait for full charge or sufficient level
    # Then proceed to SLEEP_LEARNING
```

### Startup Sequence
```python
# In startup_manager.py
health_checker.run_all_checks()  # Now includes battery check
battery_status = battery_monitor.read_status()

if battery_status.charge_percent < 15:
    logger.warning("Low battery on startup - charging first")
    mode_manager.set_mode(RobotMode.AUTO_CHARGING)
```

## Visual Homing Implementation

### AprilTag Detection (Recommended)
```python
# Add to auto_charge.py camera_detector
import cv2
from pupil_apriltags import Detector

class DockDetector:
    def __init__(self):
        self.detector = Detector(families='tag36h11')
        self.camera = cv2.VideoCapture(0)  # Or use OAK-D
    
    def detect(self):
        ret, frame = self.camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        
        results = []
        for det in detections:
            if det.tag_id == 42:  # Dock marker
                # Calculate distance and angle from tag pose
                distance_cm = self._estimate_distance(det)
                angle_deg = self._estimate_angle(det)
                lateral_cm = self._estimate_lateral_offset(det)
                
                results.append(Detection(
                    id=det.tag_id,
                    distance_cm=distance_cm,
                    angle_deg=angle_deg,
                    lateral_offset_cm=lateral_cm
                ))
        return results
```

### Fallback: Saved Waypoint
```python
# Save dock position when manually placed
auto_charger.save_dock_position(x=1.5, y=2.0, theta=0.0)

# Later, navigate using odometry
# Requires wheel encoders or visual odometry
```

## Safety Features

1. **Low Battery Shutdown**: At <5%, emergency stop and alert
2. **Charging Timeout**: If not charging after 30 min docked, alert
3. **Thermal Protection**: Monitor Pi temperature during charging
4. **Manual Override**: Allow manual undocking via API/button
5. **BMS Protection**: Use proper LiPo BMS to prevent overcharge/overdischarge

## Testing Checklist

- [ ] INA219 sensor reads correct voltage
- [ ] Battery percentage estimation accurate
- [ ] Charging detection works (current < 0)
- [ ] AprilTag detection reliable at 0.5-2m distance
- [ ] Robot navigates to dock within 50cm
- [ ] Alignment achieves <2cm lateral error
- [ ] Pogo pins make solid contact
- [ ] Charging begins automatically when docked
- [ ] Robot undocks when fully charged
- [ ] Low battery triggers charging behavior
- [ ] System health check reports battery status
- [ ] API endpoints return battery data

## Future Enhancements

1. **Multi-Dock Support**: Register multiple charging stations
2. **Predictive Charging**: Learn usage patterns, pre-charge
3. **Wireless Charging**: Qi coils instead of pogo pins
4. **Solar Assist**: Trickle charge from solar panel
5. **Battery Swapping**: Quick-release battery pack
6. **Fleet Management**: Coordinate charging across multiple robots

## Troubleshooting

**INA219 not detected:**
```bash
i2cdetect -y 1  # Should show 0x40
# If not, check wiring and I2C enable
```

**Battery percentage always 0%:**
- Check voltage range constants in `battery_monitor.py`
- Verify battery is actually 3S (11.1V nominal)
- Calibrate with actual voltage measurements

**Dock not found:**
- Check AprilTag is printed clearly (high contrast)
- Verify tag ID is 42
- Increase lighting
- Check camera exposure settings

**Won't align properly:**
- Reduce `DOCK_ALIGNMENT_TOLERANCE_CM` if too loose
- Check pogo pin spring strength
- Add wider alignment funnel

**Not charging when docked:**
- Verify pogo pin contact (multimeter continuity)
- Check BMS wiring
- Confirm 12V present at dock contacts
- Verify charging controller configuration

## References

- INA219 Datasheet: https://www.ti.com/lit/ds/symlink/ina219.pdf
- AprilTag Library: https://github.com/AprilRobotics/apriltag
- LiPo Safety Guide: https://rogershobbycenter.com/lipoguide
- Pogo Pin Selection: https://www.mill-max.com/products/spring-loaded-pins
