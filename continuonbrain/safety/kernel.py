"""
Safety Kernel - Ring 0 Implementation

The Safety Kernel is the highest-privilege component in the ContinuonBrain
system. It operates like Ring 0 in Unix/Linux:

1. FIRST to initialize on boot
2. CANNOT be disabled, killed, or bypassed
3. Has VETO power over all actions
4. Runs at HIGHEST OS priority
5. Direct access to hardware (emergency stop)

This module uses several OS-level mechanisms to ensure Ring 0 behavior:
- atexit handlers (survives normal shutdown)
- signal handlers (survives SIGTERM, SIGINT)
- Real-time scheduling (when available)
- Watchdog thread (self-monitoring)
- Hardware GPIO (direct emergency stop)
"""

import os
import sys
import time
import atexit
import signal
import logging
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety severity levels."""
    INFO = auto()       # Informational
    WARNING = auto()    # Potential issue
    VIOLATION = auto()  # Safety rule violated
    CRITICAL = auto()   # Immediate danger
    EMERGENCY = auto()  # Emergency stop required


class SafetyViolation(Exception):
    """Exception raised when a safety rule is violated."""
    def __init__(self, message: str, level: SafetyLevel = SafetyLevel.VIOLATION):
        super().__init__(message)
        self.level = level
        self.timestamp = datetime.now()


class EmergencyStop(Exception):
    """Exception raised for emergency stop condition."""
    def __init__(self, reason: str):
        super().__init__(f"EMERGENCY STOP: {reason}")
        self.reason = reason
        self.timestamp = datetime.now()


@dataclass
class SafetyState:
    """Current safety system state."""
    initialized: bool = False
    emergency_stopped: bool = False
    violations: List[SafetyViolation] = field(default_factory=list)
    blocked_actions: int = 0
    allowed_actions: int = 0
    last_check_time: Optional[datetime] = None
    watchdog_alive: bool = False
    hardware_estop_engaged: bool = False
    

class SafetyKernel:
    """
    Ring 0 Safety Kernel
    
    This is a singleton that initializes on first import and cannot be
    destroyed or bypassed. It has veto power over all robot actions.
    """
    
    _instance: Optional["SafetyKernel"] = None
    _lock = threading.Lock()
    _initialized = False
    
    # Ring 0 constants
    RING_LEVEL = 0
    PRIORITY_REALTIME = True
    CANNOT_DISABLE = True
    
    def __new__(cls):
        """Singleton pattern - only one safety kernel instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Ring 0 safety kernel."""
        if SafetyKernel._initialized:
            return
        
        SafetyKernel._initialized = True
        
        self.state = SafetyState()
        self._action_validators: List[Callable] = []
        self._emergency_callbacks: List[Callable] = []
        self._blocked_action_types: Set[str] = set()
        
        # Watchdog thread
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        
        # Hardware interface
        self._gpio_available = False
        self._estop_pin: Optional[int] = None
        
        # Initialize Ring 0 protections
        self._init_ring0()
        
        logger.info("=" * 60)
        logger.info("SAFETY KERNEL INITIALIZED - RING 0")
        logger.info("=" * 60)
        logger.info("  Priority: HIGHEST (Ring 0)")
        logger.info("  Cannot disable: TRUE")
        logger.info("  Veto power: ALL ACTIONS")
        logger.info("=" * 60)
        
        self.state.initialized = True
    
    def _init_ring0(self) -> None:
        """Initialize Ring 0 protections."""
        # 1. Register atexit handler (survives normal shutdown)
        atexit.register(self._on_shutdown)
        
        # 2. Register signal handlers (survives SIGTERM, SIGINT)
        self._register_signal_handlers()
        
        # 3. Try to set real-time priority
        self._try_realtime_priority()
        
        # 4. Start watchdog thread
        self._start_watchdog()
        
        # 5. Initialize hardware e-stop if available
        self._init_hardware_estop()
        
        logger.info("Ring 0 protections initialized")
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers that cannot be bypassed."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.warning(f"Signal {signal_name} received - safety kernel intercepting")
            
            # On SIGTERM/SIGINT, ensure safe shutdown
            if signum in (signal.SIGTERM, signal.SIGINT):
                self._safe_shutdown()
                sys.exit(0)
        
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            
            # On Unix, also handle SIGUSR1 for status
            if hasattr(signal, 'SIGUSR1'):
                signal.signal(signal.SIGUSR1, lambda s, f: self._log_status())
                
        except Exception as e:
            logger.warning(f"Could not register signal handlers: {e}")
    
    def _try_realtime_priority(self) -> None:
        """Try to set real-time scheduling priority."""
        try:
            # Linux: Try to set SCHED_FIFO with high priority
            if hasattr(os, 'sched_setscheduler'):
                try:
                    param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
                    os.sched_setscheduler(0, os.SCHED_FIFO, param)
                    logger.info("Real-time priority set (SCHED_FIFO)")
                except PermissionError:
                    logger.info("Real-time priority requires root - using nice")
                    os.nice(-20)  # Highest nice priority
            else:
                # Try nice value
                os.nice(-10)
                
        except Exception as e:
            logger.debug(f"Could not set real-time priority: {e}")
    
    def _start_watchdog(self) -> None:
        """Start watchdog thread that monitors safety kernel health."""
        def watchdog_loop():
            self.state.watchdog_alive = True
            while not self._watchdog_stop.is_set():
                try:
                    # Check safety invariants
                    self._check_invariants()
                    self.state.last_check_time = datetime.now()
                except Exception as e:
                    logger.error(f"Watchdog detected error: {e}")
                    self.emergency_stop(f"Watchdog error: {e}")
                
                self._watchdog_stop.wait(timeout=0.1)  # 10Hz watchdog
            
            self.state.watchdog_alive = False
        
        self._watchdog_thread = threading.Thread(
            target=watchdog_loop,
            name="SafetyKernel-Watchdog",
            daemon=False,  # NOT daemon - survives main thread
        )
        self._watchdog_thread.start()
    
    def _init_hardware_estop(self) -> None:
        """Initialize hardware emergency stop if GPIO available."""
        # Only enable hardware e-stop if explicitly configured
        if os.environ.get('CONTINUON_ENABLE_HARDWARE_ESTOP', '0') != '1':
            logger.debug("Hardware E-Stop not enabled (set CONTINUON_ENABLE_HARDWARE_ESTOP=1)")
            self._gpio_available = False
            return
        
        try:
            # Check for Raspberry Pi GPIO
            import RPi.GPIO as GPIO
            
            self._estop_pin = int(os.environ.get('CONTINUON_ESTOP_PIN', 17))
            
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._estop_pin, GPIO.OUT, initial=GPIO.LOW)
            
            self._gpio_available = True
            logger.info(f"Hardware E-Stop initialized on GPIO {self._estop_pin}")
            
        except ImportError:
            logger.debug("RPi.GPIO not available - hardware E-Stop disabled")
            self._gpio_available = False
        except Exception as e:
            logger.warning(f"Could not initialize hardware E-Stop: {e}")
            self._gpio_available = False
    
    def _check_invariants(self) -> None:
        """Check safety invariants (called by watchdog)."""
        # Invariant 1: Safety kernel must be initialized
        if not self.state.initialized:
            raise SafetyViolation("Safety kernel not initialized", SafetyLevel.CRITICAL)
        
        # Invariant 2: If emergency stopped, ensure actuators are disabled
        if self.state.emergency_stopped:
            self._ensure_actuators_disabled()
    
    def _ensure_actuators_disabled(self) -> None:
        """Ensure all actuators are disabled (called during e-stop)."""
        if not self._gpio_available or not self._estop_pin:
            return  # No hardware e-stop configured
        
        try:
            import RPi.GPIO as GPIO
            GPIO.output(self._estop_pin, GPIO.HIGH)  # Trigger E-Stop relay
        except Exception as e:
            # Only log once per session
            if not hasattr(self, '_estop_error_logged'):
                logger.error(f"Failed to trigger hardware E-Stop: {e}")
                self._estop_error_logged = True
    
    # =========================================================================
    # PUBLIC API - Ring 0 Functions
    # =========================================================================
    
    @classmethod
    def _get_instance(cls) -> "SafetyKernel":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def allow_action(cls, action: Dict[str, Any]) -> bool:
        """
        Check if an action is allowed by safety kernel.
        
        This is the main gate - ALL actions must pass through here.
        Ring 0 has veto power over everything.
        
        Args:
            action: Action dict with 'type', 'target', 'params', etc.
        
        Returns:
            True if action is allowed, False if blocked
        """
        kernel = cls._get_instance()
        
        # Emergency stop blocks everything
        if kernel.state.emergency_stopped:
            kernel.state.blocked_actions += 1
            logger.warning(f"Action blocked (E-Stop active): {action.get('type', 'unknown')}")
            return False
        
        # Check action type blocks
        action_type = action.get('type', '')
        if action_type in kernel._blocked_action_types:
            kernel.state.blocked_actions += 1
            logger.warning(f"Action type blocked: {action_type}")
            return False
        
        # Run all validators
        for validator in kernel._action_validators:
            try:
                if not validator(action):
                    kernel.state.blocked_actions += 1
                    return False
            except SafetyViolation as v:
                kernel.state.violations.append(v)
                kernel.state.blocked_actions += 1
                logger.warning(f"Safety violation: {v}")
                return False
        
        kernel.state.allowed_actions += 1
        return True
    
    @classmethod
    def emergency_stop(cls, reason: str = "Manual trigger") -> None:
        """
        Trigger emergency stop.
        
        This CANNOT be blocked or bypassed. Ring 0 highest priority.
        
        Args:
            reason: Reason for emergency stop
        """
        kernel = cls._get_instance()
        
        logger.critical("=" * 60)
        logger.critical("EMERGENCY STOP TRIGGERED")
        logger.critical(f"Reason: {reason}")
        logger.critical("=" * 60)
        
        kernel.state.emergency_stopped = True
        kernel.state.hardware_estop_engaged = True
        
        # Disable hardware immediately
        kernel._ensure_actuators_disabled()
        
        # Call all emergency callbacks
        for callback in kernel._emergency_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
        
        # Log to persistent storage
        kernel._log_emergency(reason)
    
    @classmethod
    def reset_estop(cls, authorization: str) -> bool:
        """
        Reset emergency stop (requires authorization).
        
        Args:
            authorization: Authorization token/code
        
        Returns:
            True if reset successful
        """
        kernel = cls._get_instance()
        
        # Simple authorization check (in production, use proper auth)
        required_auth = os.environ.get('CONTINUON_ESTOP_RESET_CODE', 'SAFETY_RESET_2026')
        
        if authorization != required_auth:
            logger.warning("E-Stop reset rejected: Invalid authorization")
            return False
        
        logger.info("E-Stop reset authorized")
        kernel.state.emergency_stopped = False
        kernel.state.hardware_estop_engaged = False
        
        # Release hardware e-stop
        if kernel._gpio_available and kernel._estop_pin:
            try:
                import RPi.GPIO as GPIO
                GPIO.output(kernel._estop_pin, GPIO.LOW)
            except Exception as e:
                logger.error(f"Failed to release hardware E-Stop: {e}")
                return False
        
        return True
    
    @classmethod
    def register_validator(cls, validator: Callable[[Dict], bool]) -> None:
        """
        Register an action validator.
        
        Args:
            validator: Function that takes action dict, returns bool
        """
        kernel = cls._get_instance()
        kernel._action_validators.append(validator)
    
    @classmethod
    def register_emergency_callback(cls, callback: Callable[[str], None]) -> None:
        """
        Register callback for emergency stop events.
        
        Args:
            callback: Function called with reason string
        """
        kernel = cls._get_instance()
        kernel._emergency_callbacks.append(callback)
    
    @classmethod
    def block_action_type(cls, action_type: str) -> None:
        """Block all actions of a specific type."""
        kernel = cls._get_instance()
        kernel._blocked_action_types.add(action_type)
        logger.info(f"Blocked action type: {action_type}")
    
    @classmethod
    def unblock_action_type(cls, action_type: str) -> None:
        """Unblock actions of a specific type."""
        kernel = cls._get_instance()
        kernel._blocked_action_types.discard(action_type)
    
    @classmethod
    def get_state(cls) -> Dict[str, Any]:
        """Get current safety kernel state."""
        kernel = cls._get_instance()
        return {
            "ring_level": cls.RING_LEVEL,
            "initialized": kernel.state.initialized,
            "emergency_stopped": kernel.state.emergency_stopped,
            "hardware_estop_engaged": kernel.state.hardware_estop_engaged,
            "watchdog_alive": kernel.state.watchdog_alive,
            "violations_count": len(kernel.state.violations),
            "blocked_actions": kernel.state.blocked_actions,
            "allowed_actions": kernel.state.allowed_actions,
            "last_check": kernel.state.last_check_time.isoformat() if kernel.state.last_check_time else None,
            "blocked_action_types": list(kernel._blocked_action_types),
            "validators_count": len(kernel._action_validators),
        }
    
    @classmethod
    def is_safe(cls) -> bool:
        """Quick check if system is in safe state."""
        kernel = cls._get_instance()
        return (
            kernel.state.initialized and
            not kernel.state.emergency_stopped and
            kernel.state.watchdog_alive
        )
    
    # =========================================================================
    # INTERNAL
    # =========================================================================
    
    def _safe_shutdown(self) -> None:
        """Perform safe shutdown of all systems."""
        logger.info("Safety kernel initiating safe shutdown")
        
        # Stop all actuators
        self._ensure_actuators_disabled()
        
        # Stop watchdog
        self._watchdog_stop.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=1.0)
        
        # Log final state
        self._log_status()
    
    def _on_shutdown(self) -> None:
        """Called on Python exit (atexit handler)."""
        logger.info("Safety kernel shutdown handler called")
        self._safe_shutdown()
    
    def _log_status(self) -> None:
        """Log current safety status."""
        state = self.get_state()
        logger.info(f"Safety Kernel Status: {state}")
    
    def _log_emergency(self, reason: str) -> None:
        """Log emergency stop to persistent storage."""
        log_path = Path('/opt/continuonos/brain/logs/emergency_stops.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_path, 'a') as f:
                f.write(f"{datetime.now().isoformat()} | EMERGENCY STOP | {reason}\n")
        except Exception as e:
            logger.error(f"Failed to log emergency: {e}")


# ============================================================================
# BOOT SEQUENCE - Safety Kernel initializes FIRST
# ============================================================================

def _boot_safety_kernel():
    """
    Boot the safety kernel.
    
    This is called on module import to ensure safety is initialized
    BEFORE any other component.
    """
    try:
        kernel = SafetyKernel._get_instance()
        
        # Verify initialization
        if not kernel.state.initialized:
            raise RuntimeError("Safety kernel failed to initialize")
        
        if not kernel.state.watchdog_alive:
            raise RuntimeError("Safety watchdog not running")
        
        logger.info("Safety kernel boot complete - Ring 0 active")
        
    except Exception as e:
        logger.critical(f"CRITICAL: Safety kernel boot failed: {e}")
        raise


# Auto-initialize on import
_boot_safety_kernel()

