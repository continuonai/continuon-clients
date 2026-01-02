"""
Safety Monitor - Continuous Safety Monitoring

The Safety Monitor runs as a Ring 0 component, continuously monitoring
sensor inputs and triggering safety responses.
"""

import os
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field

from .kernel import SafetyKernel, SafetyViolation, SafetyLevel

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """A sensor reading with timestamp."""
    name: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    valid: bool = True


@dataclass
class SafetyCheck:
    """A periodic safety check."""
    name: str
    check_fn: Callable[[], bool]
    interval_ms: int = 100
    last_check: Optional[datetime] = None
    last_result: bool = True
    failure_count: int = 0


class SafetyMonitor:
    """
    Continuous safety monitoring at Ring 0.
    
    Monitors:
    - Temperature (CPU, motors)
    - Power (voltage, current)
    - Force/torque sensors
    - Proximity/collision sensors
    - System health
    """
    
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Sensor readings
        self._readings: Dict[str, SensorReading] = {}
        self._reading_lock = threading.Lock()
        
        # Safety checks
        self._checks: List[SafetyCheck] = []
        
        # Thresholds
        self._thresholds: Dict[str, Any] = {
            'cpu_temp_warning': 75.0,
            'cpu_temp_critical': 85.0,
            'motor_temp_warning': 60.0,
            'motor_temp_critical': 80.0,
            'min_voltage': 11.0,
            'max_current': 10.0,
            'force_threshold': 50.0,
        }
        
        # Register standard checks
        self._register_standard_checks()
    
    def _register_standard_checks(self) -> None:
        """Register standard safety checks."""
        self.add_check(SafetyCheck(
            name="cpu_temperature",
            check_fn=self._check_cpu_temperature,
            interval_ms=1000,  # Check every second
        ))
        
        self.add_check(SafetyCheck(
            name="memory_usage",
            check_fn=self._check_memory_usage,
            interval_ms=5000,  # Check every 5 seconds
        ))
        
        self.add_check(SafetyCheck(
            name="watchdog_alive",
            check_fn=self._check_watchdog,
            interval_ms=100,  # Check 10Hz
        ))
    
    def add_check(self, check: SafetyCheck) -> None:
        """Add a safety check."""
        self._checks.append(check)
        logger.debug(f"Added safety check: {check.name}")
    
    def update_reading(self, name: str, value: Any) -> None:
        """Update a sensor reading."""
        with self._reading_lock:
            self._readings[name] = SensorReading(
                name=name,
                value=value,
                timestamp=datetime.now(),
            )
    
    def get_reading(self, name: str) -> Optional[SensorReading]:
        """Get a sensor reading."""
        with self._reading_lock:
            return self._readings.get(name)
    
    def start(self) -> None:
        """Start the safety monitor."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="SafetyMonitor",
            daemon=False,  # Not daemon - survives main thread (Ring 0)
        )
        self._thread.start()
        
        logger.info("Safety monitor started (Ring 0)")
    
    def stop(self) -> None:
        """Stop the safety monitor."""
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running and not self._stop_event.is_set():
            try:
                now = datetime.now()
                
                for check in self._checks:
                    # Check if it's time to run this check
                    if check.last_check is None:
                        run_check = True
                    else:
                        elapsed_ms = (now - check.last_check).total_seconds() * 1000
                        run_check = elapsed_ms >= check.interval_ms
                    
                    if run_check:
                        try:
                            result = check.check_fn()
                            check.last_check = now
                            check.last_result = result
                            
                            if not result:
                                check.failure_count += 1
                                self._handle_check_failure(check)
                            else:
                                check.failure_count = 0
                                
                        except Exception as e:
                            logger.error(f"Safety check '{check.name}' error: {e}")
                            check.failure_count += 1
                            self._handle_check_failure(check)
                
                # Sleep briefly
                self._stop_event.wait(timeout=0.01)  # 10ms
                
            except Exception as e:
                logger.critical(f"Safety monitor error: {e}")
                # On critical error, trigger emergency stop
                SafetyKernel.emergency_stop(f"Monitor error: {e}")
    
    def _handle_check_failure(self, check: SafetyCheck) -> None:
        """Handle a failed safety check."""
        logger.warning(f"Safety check '{check.name}' failed (count: {check.failure_count})")
        
        # Trigger e-stop after 3 consecutive failures
        if check.failure_count >= 3:
            SafetyKernel.emergency_stop(f"Safety check '{check.name}' failed 3 times")
    
    # =========================================================================
    # STANDARD CHECKS
    # =========================================================================
    
    def _check_cpu_temperature(self) -> bool:
        """Check CPU temperature is within limits."""
        try:
            # Read Pi CPU temperature
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_milli = int(f.read().strip())
                temp_c = temp_milli / 1000.0
            
            self.update_reading('cpu_temp', temp_c)
            
            if temp_c >= self._thresholds['cpu_temp_critical']:
                logger.critical(f"CPU temperature critical: {temp_c:.1f}°C")
                return False
            
            if temp_c >= self._thresholds['cpu_temp_warning']:
                logger.warning(f"CPU temperature warning: {temp_c:.1f}°C")
            
            return True
            
        except FileNotFoundError:
            # Not on Pi, check psutil
            try:
                import psutil
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current >= self._thresholds['cpu_temp_critical']:
                                logger.critical(f"Temperature critical: {name} {entry.current:.1f}°C")
                                return False
                return True
            except ImportError:
                return True  # No temp monitoring available
            
        except Exception as e:
            logger.warning(f"Could not check CPU temperature: {e}")
            return True  # Assume safe if can't check
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage is within limits."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            self.update_reading('ram_percent', mem.percent)
            self.update_reading('swap_percent', swap.percent)
            
            # Warn if memory is very high
            if mem.percent >= 95:
                logger.warning(f"Memory critical: {mem.percent:.1f}%")
                return False
            
            if mem.percent >= 85:
                logger.warning(f"Memory high: {mem.percent:.1f}%")
            
            return True
            
        except ImportError:
            return True  # psutil not available
        except Exception as e:
            logger.warning(f"Could not check memory: {e}")
            return True
    
    def _check_watchdog(self) -> bool:
        """Check that safety kernel watchdog is alive."""
        state = SafetyKernel.get_state()
        return state.get('watchdog_alive', False)
    
    # =========================================================================
    # FORCE / TORQUE MONITORING
    # =========================================================================
    
    def check_force(self, force_reading: float) -> bool:
        """Check if force is within safe limits."""
        self.update_reading('contact_force', force_reading)
        
        if abs(force_reading) > self._thresholds['force_threshold']:
            logger.warning(f"Force limit exceeded: {force_reading:.1f}N")
            return False
        
        return True
    
    def check_torques(self, torques: List[float]) -> bool:
        """Check if joint torques are within limits."""
        from .bounds import DEFAULT_BOUNDS
        
        self.update_reading('joint_torques', torques)
        
        ok, msg = DEFAULT_BOUNDS.joints.check_torque(torques)
        if not ok:
            logger.warning(f"Torque limit: {msg}")
        
        return ok
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status."""
        with self._reading_lock:
            readings = {k: {'value': v.value, 'timestamp': v.timestamp.isoformat()} 
                       for k, v in self._readings.items()}
        
        checks = {
            c.name: {
                'last_result': c.last_result,
                'failure_count': c.failure_count,
                'last_check': c.last_check.isoformat() if c.last_check else None,
            }
            for c in self._checks
        }
        
        return {
            'running': self._running,
            'readings': readings,
            'checks': checks,
            'thresholds': self._thresholds,
        }


# Global monitor instance (starts automatically)
_monitor: Optional[SafetyMonitor] = None


def get_monitor() -> SafetyMonitor:
    """Get the global safety monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = SafetyMonitor()
        _monitor.start()
    return _monitor

