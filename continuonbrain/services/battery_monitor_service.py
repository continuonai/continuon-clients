"""
Background battery monitoring service.
Periodically logs battery status and triggers alerts/shutdowns on critical conditions.
"""

import time
import logging
import threading
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class BatteryMonitorService:
    """
    Background service that monitors battery status and logs voltage every 10 seconds.
    Implements voltage-based alerts and emergency shutdown on critical conditions.
    """
    
    def __init__(
        self,
        config_dir: str = "/opt/continuonos/brain",
        log_interval_seconds: float = 10.0,
        alert_voltage_v: float = 10.5,  # Alert at <10.5V (3.5V/cell)
        shutdown_voltage_v: float = 9.9,  # Emergency shutdown at <9.9V (3.3V/cell)
        shutdown_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize battery monitoring service.
        
        Args:
            config_dir: Configuration directory for logs
            log_interval_seconds: Interval between battery status logs (default 10s)
            alert_voltage_v: Voltage threshold for alerts (default 10.5V)
            shutdown_voltage_v: Voltage threshold for emergency shutdown (default 9.9V)
            shutdown_callback: Optional callback function to call on emergency shutdown
        """
        self.config_dir = Path(config_dir)
        self.log_interval_seconds = log_interval_seconds
        self.alert_voltage_v = alert_voltage_v
        self.shutdown_voltage_v = shutdown_voltage_v
        self.shutdown_callback = shutdown_callback
        
        self.monitor = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_status = None
        self.alert_sent = False
        
        # Initialize battery monitor
        try:
            from continuonbrain.sensors.battery_monitor import BatteryMonitor
            self.monitor = BatteryMonitor()
            logger.info("Battery monitor service initialized")
        except Exception as e:
            logger.warning(f"Battery monitor unavailable: {e}")
            self.monitor = None
    
    def start(self):
        """Start background monitoring thread."""
        if self.running:
            logger.warning("Battery monitor service already running")
            return
        
        if not self.monitor:
            logger.warning("Cannot start battery monitor service: monitor unavailable")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"Battery monitor service started (log interval: {self.log_interval_seconds}s)")
    
    def stop(self):
        """Stop background monitoring thread."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Battery monitor service stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        log_file = self.config_dir / "logs" / "battery.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        while self.running:
            try:
                status = self.monitor.read_status()
                self.last_status = status
                
                if status:
                    # Log battery status
                    self._log_status(status, log_file)
                    
                    # Check voltage thresholds
                    voltage_v = status.voltage_v
                    
                    # Emergency shutdown check
                    if voltage_v < self.shutdown_voltage_v:
                        logger.critical(
                            f"EMERGENCY SHUTDOWN: Battery voltage {voltage_v:.2f}V < {self.shutdown_voltage_v}V "
                            f"(3.3V/cell threshold)"
                        )
                        self._trigger_shutdown(status)
                        break  # Exit monitoring loop
                    
                    # Alert check
                    elif voltage_v < self.alert_voltage_v:
                        if not self.alert_sent:
                            logger.warning(
                                f"Battery voltage low: {voltage_v:.2f}V < {self.alert_voltage_v}V "
                                f"(3.5V/cell threshold)"
                            )
                            self.alert_sent = True
                    else:
                        # Reset alert flag if voltage recovered
                        if self.alert_sent and voltage_v >= self.alert_voltage_v + 0.2:
                            logger.info(f"Battery voltage recovered: {voltage_v:.2f}V")
                            self.alert_sent = False
                    
                    # Check for overcurrent conditions
                    is_safe, overcurrent_msg = self.monitor.check_overcurrent(status.current_ma)
                    if not is_safe:
                        logger.critical(f"EMERGENCY SHUTDOWN: {overcurrent_msg}")
                        self._trigger_shutdown(status)
                        break
                    elif overcurrent_msg:
                        logger.warning(overcurrent_msg)
                
                else:
                    # Battery monitor unavailable (may be tethered)
                    logger.debug("Battery monitor unavailable (may be tethered power)")
                
            except Exception as e:
                logger.error(f"Error in battery monitoring loop: {e}")
            
            # Sleep until next check
            time.sleep(self.log_interval_seconds)
    
    def _log_status(self, status, log_file: Path):
        """Log battery status to file."""
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            time_remaining = f"{status.time_to_empty_min:.1f}min" if status.time_to_empty_min else "N/A"
            log_line = (
                f"{timestamp} | "
                f"Voltage: {status.voltage_v:.2f}V | "
                f"Current: {status.current_ma:.1f}mA | "
                f"Power: {status.power_mw/1000:.2f}W | "
                f"Charge: {status.charge_percent:.1f}% | "
                f"Charging: {status.is_charging} | "
                f"Time remaining: {time_remaining}"
            )
            
            with open(log_file, 'a') as f:
                f.write(log_line + "\n")
            
        except Exception as e:
            logger.error(f"Failed to log battery status: {e}")
    
    def _trigger_shutdown(self, status):
        """Trigger emergency shutdown."""
        logger.critical("Triggering emergency shutdown due to battery condition")
        
        # Call shutdown callback if provided
        if self.shutdown_callback:
            try:
                self.shutdown_callback()
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
        
        # Log final status
        log_file = self.config_dir / "logs" / "battery.log"
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(log_file, 'a') as f:
                f.write(
                    f"{timestamp} | EMERGENCY SHUTDOWN TRIGGERED | "
                    f"Voltage: {status.voltage_v:.2f}V | "
                    f"Current: {status.current_ma:.1f}mA\n"
                )
        except Exception:
            pass
    
    def get_status(self) -> dict:
        """Get current battery status."""
        if not self.monitor:
            return {"available": False, "error": "Monitor unavailable"}
        
        if self.last_status:
            return {
                "available": True,
                "voltage_v": round(self.last_status.voltage_v, 2),
                "current_ma": round(self.last_status.current_ma, 1),
                "charge_percent": round(self.last_status.charge_percent, 1),
                "is_charging": self.last_status.is_charging,
                "alert_sent": self.alert_sent,
                "running": self.running,
            }
        else:
            diagnostics = self.monitor.get_diagnostics()
            return {
                **diagnostics,
                "running": self.running,
            }


if __name__ == "__main__":
    # Test battery monitor service
    import signal
    
    logging.basicConfig(level=logging.INFO)
    
    service = BatteryMonitorService()
    
    def shutdown_handler(signum, frame):
        print("\nShutting down battery monitor service...")
        service.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    print("Starting battery monitor service (Ctrl+C to stop)...")
    service.start()
    
    # Keep running
    while True:
        time.sleep(1)

