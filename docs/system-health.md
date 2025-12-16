# System Health & Startup Management

ContinuonBrain includes comprehensive health checking and startup management that automatically validates hardware and software whenever the robot wakes from sleep.

## Overview

The startup manager ensures system reliability by:
- **Detecting startup mode**: Cold boot, wake from sleep, or crash recovery
- **Running health checks**: Automatic validation on wake from sleep
- **Graceful degradation**: Continue with warnings, block on critical issues
- **State persistence**: Track sleep/wake cycles and crashes

## Health Checks

### Quick Mode (Wake from Sleep)
Runs essential checks in ~2 seconds:
- Hardware detection (cameras, servos, I2C, USB)
- Python environment and dependencies
- Disk space and memory
- Directory structure and permissions

### Full Mode (Recovery/Manual)
Includes all quick checks plus:
- AI accelerator availability
- Network connectivity
- Sensor data capture tests
- Model inference validation

### Health Check Components

| Component | Check | Severity |
|-----------|-------|----------|
| Hardware Detection | Auto-detect all devices | Critical |
| Camera | Depth camera availability | Warning |
| Servo Controller | PCA9685 I2C communication | Warning |
| I2C Bus | Bus accessible | Warning |
| USB Devices | Device enumeration | Warning |
| AI Accelerator | Hailo/Coral detection | Warning |
| Python | Version 3.9+ | Warning |
| Dependencies | numpy, depthai installed | Critical |
| Model Files | Base model or manifest exists | Warning |
| RLDS Storage | Episode directory accessible | Warning |
| Disk Space | <90% full | Critical if >90% |
| Memory | <90% used | Critical if >90% |
| CPU Temperature | <80°C | Critical if >80°C |
| Network | Internet connectivity | Warning |
| Directory Structure | Required dirs exist | Warning |
| Permissions | Write access to config | Critical |
| Safety Config | Safety manifest present | Warning |

## Usage

### Manual Health Check

```bash
# Quick health check
PYTHONPATH=$PWD python3 continuonbrain/system_health.py --quick

# Full health check
PYTHONPATH=$PWD python3 continuonbrain/system_health.py

# Save report to file
PYTHONPATH=$PWD python3 continuonbrain/system_health.py --save-report /tmp/health.json

# Use custom config directory
PYTHONPATH=$PWD python3 continuonbrain/system_health.py --config-dir /opt/continuonos/brain
```

### Startup Manager

```bash
# Normal startup (auto-detects mode)
PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py

# Force health check even on cold boot
PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py --force-health-check

# Prepare for sleep (records state)
PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py --prepare-sleep

# Prepare for shutdown
PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py --prepare-shutdown
```

### In Python Code

```python
from continuonbrain.startup_manager import StartupManager
from continuonbrain.system_health import SystemHealthChecker, HealthStatus

# Startup with automatic health checks
manager = StartupManager()
success = manager.startup()

if not success:
    print("Critical issues detected - cannot start")
    exit(1)

# Manual health check
checker = SystemHealthChecker()
overall_status, results = checker.check_all(quick_mode=True)

if overall_status == HealthStatus.CRITICAL:
    print("Critical hardware failure!")
    # Handle gracefully...

# Prepare for sleep
manager.prepare_sleep()
```

## Startup Modes

### Cold Boot
First power-on or clean shutdown → startup
- **Behavior**: Skips health check by default (fast startup)
- **Use Case**: Initial deployment, development
- **Override**: Use `--force-health-check`

### Wake from Sleep
Sleep → wake transition
- **Behavior**: Automatically runs quick health check
- **Use Case**: Robot sleeping between tasks
- **Duration**: ~2 seconds

### Recovery
Crash or error → restart
- **Behavior**: Runs full health check
- **Use Case**: Recovering from failures
- **Duration**: ~5-10 seconds

## Health Status Levels

### ✅ Healthy
All checks passed. System fully operational.
- **Exit Code**: 0
- **Action**: Continue normally

### ⚠️ Warning
Non-critical issues detected. System functional but degraded.
- **Exit Code**: 1
- **Action**: Log warnings, continue with degraded features
- **Examples**: 
  - No AI accelerator (fallback to CPU)
  - Optional dependencies missing
  - Disk space >75% but <90%

### ❌ Critical
Critical issues detected. System cannot function safely.
- **Exit Code**: 2
- **Action**: Block startup, require manual intervention
- **Examples**:
  - No write permissions
  - Disk space >90% full
  - Memory >90% used
  - CPU temperature >80°C
  - Critical dependencies missing

### ❓ Unknown
Check could not complete.
- **Exit Code**: Treated as Warning
- **Action**: Log unknown status, continue cautiously

## State Persistence

The startup manager tracks state in `.startup_state` file:

```json
{
  "last_shutdown": 1701446923.123,
  "shutdown_type": "sleep",
  "timestamp": "2025-12-01 15:48:33"
}
```

### Shutdown Types

| Type | Trigger | Next Boot Behavior |
|------|---------|-------------------|
| `sleep` | Normal sleep | Quick health check |
| `shutdown` | Clean shutdown | Skip health check |
| `crash` | Unhandled error | Full health check |

## Health Reports

Health checks save JSON reports to `logs/health_*.json`:

```json
{
  "timestamp": "2025-12-01 15:48:42",
  "timestamp_ns": 1701446922123456789,
  "duration_ms": 1967.7,
  "overall_status": "warning",
  "checks": [
    {
      "component": "Hardware Detection",
      "status": "healthy",
      "message": "Detected 3 device(s)",
      "details": {"device_count": 3}
    },
    {
      "component": "Camera",
      "status": "healthy",
      "message": "OAK-D Lite available",
      "details": {
        "name": "OAK-D Lite",
        "capabilities": ["rgb", "depth", "stereo", "ai"]
      }
    }
  ]
}
```

## Integration Examples

### Systemd Service

Create `/etc/systemd/system/continuonbrain.service`:

```ini
[Unit]
Description=ContinuonBrain Robot Control
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ContinuonXR
ExecStartPre=/usr/bin/python3 continuonbrain/startup_manager.py
ExecStart=/usr/bin/python3 continuonbrain/main.py
ExecStop=/usr/bin/python3 continuonbrain/startup_manager.py --prepare-shutdown
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Cron Wake Check

Run health check on wake from sleep (if using `rtcwake`):

```bash
# Add to crontab
@reboot sleep 10 && /usr/bin/python3 /home/pi/ContinuonXR/continuonbrain/startup_manager.py
```

### Robot Main Loop

```python
from continuonbrain.startup_manager import StartupManager, StartupMode

def main():
    # Initialize startup manager
    manager = StartupManager()
    
    # Run startup checks
    if not manager.startup():
        print("Startup failed - exiting")
        return 1
    
    try:
        # Main robot loop
        while True:
            # ... robot logic ...
            pass
    
    except KeyboardInterrupt:
        print("Shutting down...")
        manager.prepare_shutdown()
    
    except Exception as e:
        print(f"Crash: {e}")
        manager.record_crash(str(e))
        raise
    
    return 0

if __name__ == "__main__":
    exit(main())
```

## Troubleshooting

### Health Check Fails on Wake

**Symptom**: Critical errors on wake from sleep

**Solutions**:
1. Check hardware connections (USB, I2C)
2. Review health report in `logs/health_*.json`
3. Run manual health check: `python3 continuonbrain/system_health.py`
4. Check system logs: `journalctl -u continuonbrain`

### Slow Wake Times

**Symptom**: Takes >5 seconds to wake

**Solutions**:
1. Use `--quick` mode (already default for wake)
2. Disable network check (already skipped in quick mode)
3. Check CPU temperature (thermal throttling)
4. Optimize startup dependencies

### False Critical Errors

**Symptom**: Critical status but system works

**Solutions**:
1. Check disk space and clean up old episodes
2. Verify temperature sensors are accurate
3. Adjust thresholds in `system_health.py`
4. File issue if check is too strict

### State File Corruption

**Symptom**: Wrong startup mode detected

**Solutions**:
1. Delete `.startup_state` file
2. Restart with `--force-health-check`
3. Check filesystem integrity

## Performance

| Operation | Duration | Notes |
|-----------|----------|-------|
| Quick Health Check | ~2s | Wake from sleep |
| Full Health Check | ~5-10s | Recovery mode |
| State Recording | <10ms | Sleep/shutdown |
| Cold Boot (no check) | <100ms | Development |

## Best Practices

1. **Always prepare for sleep**: Call `prepare_sleep()` before sleeping
2. **Handle critical errors**: Don't ignore critical health status
3. **Monitor health reports**: Review logs for patterns
4. **Test recovery**: Simulate crashes to validate recovery
5. **Keep logs clean**: Rotate old health reports
6. **Calibrate thresholds**: Adjust for your hardware

## Future Enhancements

- [ ] Automatic recovery actions (disk cleanup, service restart)
- [ ] Predictive failure detection (trend analysis)
- [ ] Remote health monitoring (cloud reporting)
- [ ] Hardware-specific tests (camera capture, servo movement)
- [ ] Performance benchmarking (inference latency)
- [ ] Network diagnostics (connectivity, bandwidth)
- [ ] Battery health monitoring (if on UPS)
- [ ] Thermal management (active cooling control)

## See Also

- [Hardware Detection](hardware-detection.md) - Auto-detect cameras, HATs, servos
- [Pi5 Edge Brain Instructions](../continuonbrain/PI5_EDGE_BRAIN_INSTRUCTIONS.md) - Current Pi runbook
- [System Architecture](system-architecture.md) - Overall design
