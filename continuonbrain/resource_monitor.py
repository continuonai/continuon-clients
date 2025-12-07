"""
Resource Monitor for ContinuonBrain

System-wide resource monitoring to ensure system responsiveness by:
- Tracking RAM, swap, and CPU usage
- Triggering cleanup callbacks at defined thresholds
- Implementing graceful degradation under memory pressure
- Providing resource status API for services
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


logger = logging.getLogger(__name__)


class ResourceLevel(Enum):
    """Resource usage level."""
    NORMAL = "normal"          # < 75% usage
    WARNING = "warning"        # 75-85% usage
    CRITICAL = "critical"      # 85-90% usage
    EMERGENCY = "emergency"    # > 90% usage


@dataclass
class ResourceStatus:
    """Current resource status."""
    timestamp: float
    
    # Memory (MB)
    total_memory_mb: int
    used_memory_mb: int
    available_memory_mb: int
    memory_percent: float
    
    # Swap (MB)
    total_swap_mb: int
    used_swap_mb: int
    swap_percent: float
    
    # Limits
    system_reserve_mb: int
    max_brain_mb: int
    
    # Status
    level: ResourceLevel
    can_allocate: bool
    message: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result['level'] = self.level.value
        return result


@dataclass
class ResourceLimits:
    """Resource limit configuration."""
    system_reserve_mb: int = 2000
    max_brain_mb: int = 2000
    warning_threshold_percent: float = 75.0
    critical_threshold_percent: float = 85.0
    emergency_threshold_percent: float = 90.0
    check_interval_sec: float = 5.0
    log_usage: bool = True


class ResourceMonitor:
    """
    System-wide resource monitor.
    
    Monitors RAM, swap, and CPU usage and triggers cleanup callbacks
    when thresholds are exceeded.
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None, config_dir: Optional[Path] = None):
        """
        Initialize resource monitor.
        
        Args:
            limits: Resource limits configuration
            config_dir: Configuration directory for loading/saving limits
        """
        self.limits = limits or ResourceLimits()
        self.config_dir = Path(config_dir) if config_dir else None
        
        # Load limits from config if available
        if self.config_dir:
            self._load_limits()
        
        # Cleanup callbacks: {threshold_percent: [callbacks]}
        self.cleanup_callbacks: Dict[float, List[Callable]] = {
            self.limits.warning_threshold_percent: [],
            self.limits.critical_threshold_percent: [],
            self.limits.emergency_threshold_percent: [],
        }
        
        # Track last status
        self.last_status: Optional[ResourceStatus] = None
        self.last_level: ResourceLevel = ResourceLevel.NORMAL
        
        logger.info(f"Resource Monitor initialized: "
                   f"System reserve {self.limits.system_reserve_mb}MB, "
                   f"Brain max {self.limits.max_brain_mb}MB")
    
    def _load_limits(self):
        """Load resource limits from config file."""
        if not self.config_dir:
            return
        
        config_file = self.config_dir / "config" / "resource_limits.json"
        if not config_file.exists():
            logger.info(f"No resource limits config found at {config_file}, using defaults")
            return
        
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            memory_config = config.get("memory", {})
            self.limits.system_reserve_mb = memory_config.get("system_reserve_mb", self.limits.system_reserve_mb)
            self.limits.max_brain_mb = memory_config.get("max_brain_mb", self.limits.max_brain_mb)
            self.limits.warning_threshold_percent = memory_config.get("warning_threshold_percent", self.limits.warning_threshold_percent)
            self.limits.critical_threshold_percent = memory_config.get("critical_threshold_percent", self.limits.critical_threshold_percent)
            self.limits.emergency_threshold_percent = memory_config.get("emergency_threshold_percent", self.limits.emergency_threshold_percent)
            
            monitoring_config = config.get("monitoring", {})
            self.limits.check_interval_sec = monitoring_config.get("check_interval_sec", self.limits.check_interval_sec)
            self.limits.log_usage = monitoring_config.get("log_usage", self.limits.log_usage)
            
            logger.info(f"Loaded resource limits from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load resource limits from {config_file}: {e}")
    
    def save_limits(self):
        """Save current resource limits to config file."""
        if not self.config_dir:
            logger.warning("No config_dir set, cannot save limits")
            return
        
        config_file = self.config_dir / "config" / "resource_limits.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "memory": {
                "system_reserve_mb": self.limits.system_reserve_mb,
                "max_brain_mb": self.limits.max_brain_mb,
                "warning_threshold_percent": self.limits.warning_threshold_percent,
                "critical_threshold_percent": self.limits.critical_threshold_percent,
                "emergency_threshold_percent": self.limits.emergency_threshold_percent,
            },
            "monitoring": {
                "check_interval_sec": self.limits.check_interval_sec,
                "log_usage": self.limits.log_usage,
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved resource limits to {config_file}")
    
    def _read_meminfo(self) -> Tuple[int, int, int, int]:
        """
        Read memory info from /proc/meminfo.
        
        Returns:
            (total_mb, available_mb, total_swap_mb, free_swap_mb)
        """
        try:
            with open('/proc/meminfo') as f:
                lines = f.readlines()
            
            mem_total = 0
            mem_available = 0
            swap_total = 0
            swap_free = 0
            
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1]) // 1024  # Convert kB to MB
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) // 1024
                elif line.startswith('SwapTotal:'):
                    swap_total = int(line.split()[1]) // 1024
                elif line.startswith('SwapFree:'):
                    swap_free = int(line.split()[1]) // 1024
            
            return mem_total, mem_available, swap_total, swap_free
        except Exception as e:
            logger.error(f"Failed to read meminfo: {e}")
            return 0, 0, 0, 0
    
    def check_resources(self) -> ResourceStatus:
        """
        Check current resource usage.
        
        Returns:
            ResourceStatus with current usage and level
        """
        total_mb, available_mb, swap_total_mb, swap_free_mb = self._read_meminfo()
        
        if total_mb == 0:
            # Fallback if meminfo read failed
            logger.error("Failed to read memory info")
            return ResourceStatus(
                timestamp=time.time(),
                total_memory_mb=0,
                used_memory_mb=0,
                available_memory_mb=0,
                memory_percent=0.0,
                total_swap_mb=0,
                used_swap_mb=0,
                swap_percent=0.0,
                system_reserve_mb=self.limits.system_reserve_mb,
                max_brain_mb=self.limits.max_brain_mb,
                level=ResourceLevel.EMERGENCY,
                can_allocate=False,
                message="Failed to read memory info"
            )
        
        used_mb = total_mb - available_mb
        memory_percent = (used_mb / total_mb) * 100
        
        swap_used_mb = swap_total_mb - swap_free_mb
        swap_percent = (swap_used_mb / swap_total_mb * 100) if swap_total_mb > 0 else 0.0
        
        # Determine resource level
        if memory_percent >= self.limits.emergency_threshold_percent:
            level = ResourceLevel.EMERGENCY
            message = f"EMERGENCY: Memory at {memory_percent:.1f}% - only {available_mb}MB available"
            can_allocate = False
        elif memory_percent >= self.limits.critical_threshold_percent:
            level = ResourceLevel.CRITICAL
            message = f"CRITICAL: Memory at {memory_percent:.1f}% - {available_mb}MB available"
            can_allocate = available_mb > self.limits.system_reserve_mb
        elif memory_percent >= self.limits.warning_threshold_percent:
            level = ResourceLevel.WARNING
            message = f"WARNING: Memory at {memory_percent:.1f}% - {available_mb}MB available"
            can_allocate = available_mb > self.limits.system_reserve_mb
        else:
            level = ResourceLevel.NORMAL
            message = f"Normal: {available_mb}MB available ({memory_percent:.1f}% used)"
            can_allocate = True
        
        status = ResourceStatus(
            timestamp=time.time(),
            total_memory_mb=total_mb,
            used_memory_mb=used_mb,
            available_memory_mb=available_mb,
            memory_percent=memory_percent,
            total_swap_mb=swap_total_mb,
            used_swap_mb=swap_used_mb,
            swap_percent=swap_percent,
            system_reserve_mb=self.limits.system_reserve_mb,
            max_brain_mb=self.limits.max_brain_mb,
            level=level,
            can_allocate=can_allocate,
            message=message
        )
        
        # Log if level changed or if logging enabled
        if level != self.last_level:
            logger.warning(f"Resource level changed: {self.last_level.value} -> {level.value}: {message}")
            self._trigger_callbacks(level)
        elif self.limits.log_usage and level != ResourceLevel.NORMAL:
            logger.info(message)
        
        self.last_status = status
        self.last_level = level
        
        return status
    
    def get_available_memory(self) -> int:
        """
        Get available memory in MB.
        
        Returns:
            Available memory in MB
        """
        status = self.check_resources()
        return status.available_memory_mb
    
    def is_safe_to_allocate(self, size_mb: int) -> bool:
        """
        Check if it's safe to allocate the given amount of memory.
        
        Args:
            size_mb: Size to allocate in MB
        
        Returns:
            True if safe to allocate
        """
        status = self.check_resources()
        
        # Must have system reserve + requested size available
        required = self.limits.system_reserve_mb + size_mb
        return status.available_memory_mb >= required
    
    def register_cleanup_callback(self, threshold: ResourceLevel, callback: Callable):
        """
        Register a cleanup callback for a threshold.
        
        Args:
            threshold: Resource level to trigger on
            callback: Function to call when threshold is reached
        """
        threshold_percent = {
            ResourceLevel.WARNING: self.limits.warning_threshold_percent,
            ResourceLevel.CRITICAL: self.limits.critical_threshold_percent,
            ResourceLevel.EMERGENCY: self.limits.emergency_threshold_percent,
        }.get(threshold)
        
        if threshold_percent is None:
            logger.warning(f"Cannot register callback for {threshold}")
            return
        
        if threshold_percent not in self.cleanup_callbacks:
            self.cleanup_callbacks[threshold_percent] = []
        
        self.cleanup_callbacks[threshold_percent].append(callback)
        logger.info(f"Registered cleanup callback for {threshold.value} threshold")
    
    def _trigger_callbacks(self, level: ResourceLevel):
        """
        Trigger cleanup callbacks for the given level.
        
        Args:
            level: Resource level that was reached
        """
        threshold_percent = {
            ResourceLevel.WARNING: self.limits.warning_threshold_percent,
            ResourceLevel.CRITICAL: self.limits.critical_threshold_percent,
            ResourceLevel.EMERGENCY: self.limits.emergency_threshold_percent,
        }.get(level)
        
        if threshold_percent is None:
            return
        
        callbacks = self.cleanup_callbacks.get(threshold_percent, [])
        logger.info(f"Triggering {len(callbacks)} cleanup callbacks for {level.value}")
        
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
    
    def emergency_cleanup(self):
        """
        Perform emergency cleanup to free memory.
        
        This is called when memory usage is critical.
        """
        logger.warning("Performing emergency cleanup")
        
        # Trigger all cleanup callbacks
        for callbacks in self.cleanup_callbacks.values():
            for callback in callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Emergency cleanup callback failed: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Emergency cleanup complete")
    
    def get_status_summary(self) -> Dict:
        """
        Get a summary of current resource status.
        
        Returns:
            Dict with status summary
        """
        status = self.check_resources()
        return {
            "level": status.level.value,
            "memory_percent": round(status.memory_percent, 1),
            "available_mb": status.available_memory_mb,
            "can_allocate": status.can_allocate,
            "message": status.message,
            "limits": {
                "system_reserve_mb": self.limits.system_reserve_mb,
                "max_brain_mb": self.limits.max_brain_mb,
            }
        }
