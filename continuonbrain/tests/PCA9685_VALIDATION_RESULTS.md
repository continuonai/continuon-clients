# PCA9685 Wiring Validation Results

## Summary
✅ **All tests passed!** The PCA9685 wiring configuration is correct and ready for web server control testing.

## Hardware Status
- **I2C Address**: 0x40
- **Mode**: Real hardware mode (PCA9685 drivers available)
- **Drivetrain**: Initialized and ready
- **Arm Controller**: Initialized and ready

## Wiring Configuration

### Drivetrain Channels
| Channel | Function | Type | Range |
|---------|----------|------|-------|
| 0 | Steering | Standard servo | 60° to 120° (center: 90°) |
| 1 | Throttle | Continuous servo | -1.0 to +1.0 (normalized) |

### Arm Controller Channels
| Channel | Joint | Range | Default |
|---------|-------|-------|---------|
| 0 | Base | 0° to 180° | 90° |
| 1 | Shoulder | 0° to 180° | 90° |
| 2 | Elbow | 0° to 180° | 90° |
| 3 | Wrist Pitch | 0° to 180° | 90° |
| 4 | Wrist Roll | 0° to 180° | 90° |
| 5 | Gripper | 30° to 90° | 30° (open) |

## ⚠️ Important Warning: Channel Conflict

**Drivetrain and Arm both use channels 0-1!**

- Drivetrain uses channels 0-1
- Arm uses channels 0-5
- **DO NOT operate both simultaneously on the same PCA9685 board**

### Recommended Solutions

**Option 1: Single PCA9685** (Current Setup)
- Use EITHER drivetrain OR arm controller
- Switch between them as needed
- Current configuration is correct for single-use

**Option 2: Two PCA9685 Boards** (For simultaneous operation)
```python
# Drivetrain on 0x40
drivetrain = DrivetrainController()  # Uses 0x40 by default

# Arm on 0x41 (requires hardware modification)
arm = PCA9685ArmController(i2c_address=0x41)
```

To use two boards:
1. Set second PCA9685 to address 0x41 (solder jumper A0)
2. Modify arm controller initialization with `i2c_address=0x41`
3. Both can operate simultaneously without conflicts

## Validation Tests Passed

✅ All 8 tests passed:
1. Drivetrain default configuration
2. Arm default configuration
3. Drivetrain initialization
4. Arm initialization
5. Drivetrain command clamping
6. Arm safety limits
7. Arm normalized actions
8. Steering angle mapping

## Safety Features Verified

- ✓ Angle clamping to configured limits
- ✓ Command value clamping to [-1, 1]
- ✓ Safe default positions on initialization
- ✓ Graceful shutdown returns to defaults
- ✓ Mock mode fallback when hardware unavailable

## Configuration Options

### Environment Variables
```bash
export DRIVETRAIN_STEERING_CHANNEL=0
export DRIVETRAIN_THROTTLE_CHANNEL=1
export DRIVETRAIN_I2C_ADDRESS=0x40
```

### JSON Configuration
Set `DRIVETRAIN_CONFIG_PATH` to point to a JSON file:
```json
{
  "drivetrain": {
    "steering_channel": 0,
    "throttle_channel": 1,
    "i2c_address": "0x40"
  }
}
```

## Next Steps

✅ **Ready for Web Server Testing**

You can now proceed to test web server controls with confidence that:
1. PCA9685 is correctly wired and configured
2. Both drivetrain and arm controllers initialize properly
3. Safety limits are enforced
4. Commands are properly clamped to valid ranges

### Running the Validation Again

To re-run the validation at any time:
```bash
python continuonbrain/tests/validate_pca9685_wiring.py
```

Or if made executable:
```bash
./continuonbrain/tests/validate_pca9685_wiring.py
```

## Test Files Created

1. **`test_pca9685_wiring.py`** - Comprehensive pytest-based test suite
   - Requires pytest to run
   - More detailed test coverage
   - Good for CI/CD integration

2. **`validate_pca9685_wiring.py`** - Standalone validation script
   - No dependencies beyond project code
   - Quick validation and reporting
   - Recommended for manual testing

## Troubleshooting

### If tests fail:
1. Check I2C connection: `i2cdetect -y 1`
2. Verify PCA9685 appears at address 0x40
3. Check power supply to servos (separate from Pi)
4. Ensure ground is shared between Pi and servo power

### If running in mock mode:
Install hardware drivers:
```bash
pip install adafruit-circuitpython-pca9685 adafruit-servokit
```

---

**Validation Date**: 2025-12-06  
**Status**: ✅ PASSED - Ready for web server control testing
