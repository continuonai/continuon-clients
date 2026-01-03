"""
Hardware Abstraction Layer (HAL)

Provides unified interface for:
- Accessory discovery (USB, I2C, GPIO, Bluetooth)
- Device registration and capability reporting
- Sensor/actuator abstraction
- Hot-plug support
"""

from .accessory_registry import AccessoryRegistry, Accessory, AccessoryType
from .discovery import discover_accessories, scan_usb, scan_i2c, scan_gpio

__all__ = [
    'AccessoryRegistry',
    'Accessory', 
    'AccessoryType',
    'discover_accessories',
    'scan_usb',
    'scan_i2c',
    'scan_gpio',
]

