"""
Battery monitoring for self-charging capability.
Reads voltage/current from INA219/INA260 sensor via I2C.
"""
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    from ina219 import INA219
    INA219_AVAILABLE = True
except ImportError:
    INA219_AVAILABLE = False
    logger.warning("ina219 library not available - install with: pip install pi-ina219")


@dataclass
class BatteryStatus:
    """Battery status snapshot."""
    voltage_v: float
    current_ma: float
    power_mw: float
    charge_percent: float  # Estimated 0-100%
    is_charging: bool
    time_to_empty_min: Optional[float]  # Minutes remaining at current draw
    timestamp_ns: int
    
    def needs_charging(self, threshold_percent: float = 20.0) -> bool:
        """Check if battery needs charging."""
        return self.charge_percent < threshold_percent and not self.is_charging


class BatteryMonitor:
    """
    Monitor battery via INA219/INA260 I2C sensor.
    Estimates charge percentage and time remaining.
    """
    
    # Typical 3S LiPo voltage ranges
    CELL_VOLTAGE_FULL = 4.2  # Per cell fully charged
    CELL_VOLTAGE_EMPTY = 3.3  # Per cell safe minimum
    CELLS_IN_SERIES = 3  # 3S battery
    
    VOLTAGE_FULL = CELL_VOLTAGE_FULL * CELLS_IN_SERIES  # 12.6V
    VOLTAGE_EMPTY = CELL_VOLTAGE_EMPTY * CELLS_IN_SERIES  # 9.9V
    
    def __init__(
        self,
        i2c_address: int = 0x40,  # Default INA219 address
        shunt_ohms: float = 0.1,
        max_expected_amps: float = 5.0,
        battery_capacity_mah: float = 5000.0,
    ):
        """
        Initialize battery monitor.
        
        Args:
            i2c_address: I2C address of INA219 (default 0x40)
            shunt_ohms: Shunt resistor value (default 0.1Ω)
            max_expected_amps: Maximum expected current draw
            battery_capacity_mah: Battery capacity in mAh
        """
        self.i2c_address = i2c_address
        self.battery_capacity_mah = battery_capacity_mah
        self.ina: Optional[INA219] = None
        
        if INA219_AVAILABLE:
            try:
                self.ina = INA219(shunt_ohms, max_expected_amps, address=i2c_address)
                self.ina.configure(self.ina.RANGE_16V, self.ina.GAIN_AUTO)
                logger.info(f"INA219 initialized at address 0x{i2c_address:02x}")
            except Exception as e:
                logger.error(f"Failed to initialize INA219: {e}")
                self.ina = None
        else:
            logger.warning("INA219 library not available")
    
    def read_status(self) -> Optional[BatteryStatus]:
        """
        Read current battery status.
        
        Returns:
            BatteryStatus or None if sensor unavailable
        """
        if not self.ina:
            return None
        
        try:
            voltage_v = self.ina.voltage()
            current_ma = self.ina.current()
            power_mw = self.ina.power()
            
            # Estimate charge percentage from voltage (simple linear approximation)
            charge_percent = self._estimate_charge_percent(voltage_v)
            
            # Detect charging (current flowing into battery = negative)
            is_charging = current_ma < -50  # More than 50mA charging current
            
            # Estimate time to empty
            time_to_empty_min = None
            if current_ma > 0 and self.battery_capacity_mah > 0:
                # Time = Capacity / Current
                time_to_empty_min = (self.battery_capacity_mah * charge_percent / 100.0) / current_ma * 60
            
            return BatteryStatus(
                voltage_v=voltage_v,
                current_ma=current_ma,
                power_mw=power_mw,
                charge_percent=charge_percent,
                is_charging=is_charging,
                time_to_empty_min=time_to_empty_min,
                timestamp_ns=time.time_ns(),
            )
        
        except Exception as e:
            logger.error(f"Failed to read battery status: {e}")
            return None
    
    def _estimate_charge_percent(self, voltage_v: float) -> float:
        """
        Estimate battery charge percentage from voltage.
        Uses simple linear mapping (actual LiPo discharge curve is non-linear).
        
        For better accuracy, implement lookup table from discharge curve.
        """
        if voltage_v >= self.VOLTAGE_FULL:
            return 100.0
        elif voltage_v <= self.VOLTAGE_EMPTY:
            return 0.0
        else:
            return ((voltage_v - self.VOLTAGE_EMPTY) / 
                    (self.VOLTAGE_FULL - self.VOLTAGE_EMPTY) * 100.0)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        status = self.read_status()
        if not status:
            return {"available": False, "error": "Sensor not available"}
        
        return {
            "available": True,
            "voltage_v": round(status.voltage_v, 2),
            "current_ma": round(status.current_ma, 1),
            "power_w": round(status.power_mw / 1000.0, 2),
            "charge_percent": round(status.charge_percent, 1),
            "is_charging": status.is_charging,
            "time_remaining_min": round(status.time_to_empty_min, 1) if status.time_to_empty_min else None,
            "needs_charging": status.needs_charging(),
        }


if __name__ == "__main__":
    # Test battery monitor
    logging.basicConfig(level=logging.INFO)
    
    monitor = BatteryMonitor()
    
    print("Battery Monitor Test")
    print("=" * 50)
    
    for i in range(5):
        status = monitor.read_status()
        if status:
            print(f"\nReading {i+1}:")
            print(f"  Voltage: {status.voltage_v:.2f}V")
            print(f"  Current: {status.current_ma:.1f}mA")
            print(f"  Power: {status.power_mw/1000:.2f}W")
            print(f"  Charge: {status.charge_percent:.1f}%")
            print(f"  Charging: {status.is_charging}")
            if status.time_to_empty_min:
                print(f"  Time remaining: {status.time_to_empty_min:.1f} min")
            print(f"  Needs charging: {status.needs_charging()}")
        else:
            print(f"\nReading {i+1}: Sensor unavailable")
        
        time.sleep(2)
