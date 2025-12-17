"""
Autonomous charging behavior for self-charging robot.
Navigates to dock when battery is low and manages charging cycle.
"""
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChargingState(Enum):
    """States in the auto-charging state machine."""
    IDLE = "idle"  # Not charging, battery OK
    LOW_BATTERY = "low_battery"  # Battery low, need to find dock
    NAVIGATING_TO_DOCK = "navigating_to_dock"  # Moving towards charging station
    ALIGNING_WITH_DOCK = "aligning_with_dock"  # Fine-tuning position
    DOCKED = "docked"  # Successfully docked
    CHARGING = "charging"  # Charging in progress
    CHARGE_COMPLETE = "charge_complete"  # Fully charged
    DOCK_NOT_FOUND = "dock_not_found"  # Failed to locate dock
    ALIGNMENT_FAILED = "alignment_failed"  # Failed to dock properly


@dataclass
class ChargingStatus:
    """Current status of auto-charging behavior."""
    state: ChargingState
    battery_percent: float
    is_charging: bool
    dock_distance_cm: Optional[float]  # Distance to dock (if known)
    attempts: int  # Number of docking attempts
    timestamp_ns: int


class AutoChargeBehavior:
    """
    Autonomous charging behavior implementation.
    
    Uses AprilTags/ArUco markers on charging dock for visual homing,
    or alternatively odometry + saved waypoint navigation.
    """
    
    # Thresholds
    LOW_BATTERY_THRESHOLD = 20.0  # Start charging below this %
    FULL_BATTERY_THRESHOLD = 95.0  # Consider charged above this %
    CRITICAL_BATTERY_THRESHOLD = 10.0  # Emergency priority
    
    # Dock detection
    DOCK_MARKER_ID = 42  # AprilTag ID on charging dock
    DOCK_ALIGNMENT_TOLERANCE_CM = 2.0  # Position tolerance
    DOCK_ALIGNMENT_TOLERANCE_DEG = 5.0  # Angle tolerance
    
    def __init__(
        self,
        battery_monitor,  # BatteryMonitor instance
        motor_controller,  # Motor control interface
        camera_detector=None,  # Optional: AprilTag/ArUco detector
    ):
        """
        Initialize auto-charge behavior.
        
        Args:
            battery_monitor: BatteryMonitor instance
            motor_controller: Interface to robot motors (e.g., via PCA9685)
            camera_detector: Optional visual marker detector for dock homing
        """
        self.battery_monitor = battery_monitor
        self.motor_controller = motor_controller
        self.camera_detector = camera_detector
        
        self.state = ChargingState.IDLE
        self.docking_attempts = 0
        self.dock_position_saved: Optional[Tuple[float, float, float]] = None  # x, y, theta
        
        logger.info("AutoChargeBehavior initialized")
    
    def update(self) -> ChargingStatus:
        """
        Main update loop - call this periodically.
        Executes state machine for autonomous charging.
        
        Returns:
            Current charging status
        """
        battery_status = self.battery_monitor.read_status()
        if not battery_status:
            logger.error("Battery monitor unavailable")
            return self._create_status(battery_status)
        
        # State machine transitions
        if self.state == ChargingState.IDLE:
            if battery_status.charge_percent < self.LOW_BATTERY_THRESHOLD:
                logger.info(f"Battery low ({battery_status.charge_percent:.1f}%), initiating charge")
                self.state = ChargingState.LOW_BATTERY
                self.docking_attempts = 0
        
        elif self.state == ChargingState.LOW_BATTERY:
            # Begin navigation to dock
            self._start_dock_navigation()
            self.state = ChargingState.NAVIGATING_TO_DOCK
        
        elif self.state == ChargingState.NAVIGATING_TO_DOCK:
            # Navigate towards dock using vision or waypoint
            dock_found, distance = self._navigate_to_dock()
            if dock_found and distance and distance < 30:  # Within 30cm
                logger.info("Dock nearby, starting alignment")
                self.state = ChargingState.ALIGNING_WITH_DOCK
            elif not dock_found:
                logger.warning("Dock not found during navigation")
                self.state = ChargingState.DOCK_NOT_FOUND
        
        elif self.state == ChargingState.ALIGNING_WITH_DOCK:
            # Fine-tune position to dock
            aligned = self._align_with_dock()
            if aligned:
                logger.info("Successfully aligned with dock")
                self.state = ChargingState.DOCKED
                self.docking_attempts = 0
            else:
                self.docking_attempts += 1
                if self.docking_attempts > 5:
                    logger.error("Failed to align with dock after 5 attempts")
                    self.state = ChargingState.ALIGNMENT_FAILED
        
        elif self.state == ChargingState.DOCKED:
            # Verify charging has started
            if battery_status.is_charging:
                logger.info("Charging confirmed")
                self.state = ChargingState.CHARGING
            else:
                # Not charging - may need repositioning
                logger.warning("Docked but not charging")
                time.sleep(2)  # Wait briefly
                if not battery_status.is_charging:
                    self.state = ChargingState.ALIGNING_WITH_DOCK
        
        elif self.state == ChargingState.CHARGING:
            # Monitor charging progress
            if battery_status.charge_percent >= self.FULL_BATTERY_THRESHOLD:
                logger.info("Battery fully charged")
                self.state = ChargingState.CHARGE_COMPLETE
        
        elif self.state == ChargingState.CHARGE_COMPLETE:
            # Undock and return to normal operation
            self._undock()
            self.state = ChargingState.IDLE
        
        elif self.state in [ChargingState.DOCK_NOT_FOUND, ChargingState.ALIGNMENT_FAILED]:
            # Handle failures - retry or call for help
            if battery_status.charge_percent < self.CRITICAL_BATTERY_THRESHOLD:
                logger.critical("Critical battery with failed docking - emergency stop")
                self.motor_controller.emergency_stop()
            else:
                # Retry after delay
                time.sleep(10)
                self.state = ChargingState.LOW_BATTERY
        
        return self._create_status(battery_status)
    
    def _start_dock_navigation(self):
        """Begin navigation towards charging dock."""
        logger.info("Starting navigation to charging dock")
        # Implementation depends on navigation system:
        # - If using vision: scan for AprilTag marker
        # - If using odometry: navigate to saved waypoint
        # - If using SLAM: navigate to known dock location
        pass
    
    def _navigate_to_dock(self) -> Tuple[bool, Optional[float]]:
        """
        Navigate towards dock using available sensors.
        
        Returns:
            (dock_found, distance_cm) tuple
        """
        if self.camera_detector:
            # Use visual marker detection (AprilTag/ArUco)
            detections = self.camera_detector.detect()
            for detection in detections:
                if detection.id == self.DOCK_MARKER_ID:
                    # Calculate motor commands to approach marker
                    distance_cm = detection.distance_cm
                    angle_deg = detection.angle_deg
                    
                    # Simple proportional control
                    forward_speed = min(0.3, distance_cm / 100.0)  # Slow down as we approach
                    turn_rate = angle_deg * 0.01
                    
                    self.motor_controller.set_velocity(forward_speed, turn_rate)
                    return True, distance_cm
            
            # Marker not visible - rotate to search
            self.motor_controller.set_velocity(0.0, 0.2)
            return False, None
        else:
            # Fallback: navigate to saved waypoint using odometry
            if self.dock_position_saved:
                # Navigate to (x, y, theta)
                # This requires odometry/localization system
                logger.info("Navigating to saved dock position")
                # TODO: Implement waypoint navigation
                return True, 50.0  # Placeholder
            else:
                logger.error("No dock position saved and no camera detector")
                return False, None
    
    def _align_with_dock(self) -> bool:
        """
        Fine-tune alignment with charging dock.
        
        Returns:
            True if successfully aligned
        """
        if not self.camera_detector:
            logger.warning("No camera detector - cannot align precisely")
            return True  # Assume aligned if no vision
        
        detections = self.camera_detector.detect()
        for detection in detections:
            if detection.id == self.DOCK_MARKER_ID:
                distance_cm = detection.distance_cm
                angle_deg = detection.angle_deg
                lateral_offset_cm = detection.lateral_offset_cm
                
                # Check if within tolerances
                if (abs(lateral_offset_cm) < self.DOCK_ALIGNMENT_TOLERANCE_CM and
                    abs(angle_deg) < self.DOCK_ALIGNMENT_TOLERANCE_DEG and
                    distance_cm < 10):  # Very close
                    
                    self.motor_controller.stop()
                    return True
                
                # Make small adjustments
                forward = 0.1 if distance_cm > 5 else -0.05
                turn = lateral_offset_cm * 0.005
                self.motor_controller.set_velocity(forward, turn)
                return False
        
        # Lost sight of marker
        return False
    
    def _undock(self):
        """Back away from charging dock."""
        logger.info("Undocking from charging station")
        self.motor_controller.set_velocity(-0.2, 0.0)  # Reverse
        time.sleep(2)
        self.motor_controller.stop()
    
    def _create_status(self, battery_status) -> ChargingStatus:
        """Create ChargingStatus from current state."""
        return ChargingStatus(
            state=self.state,
            battery_percent=battery_status.charge_percent if battery_status else 0.0,
            is_charging=battery_status.is_charging if battery_status else False,
            dock_distance_cm=None,  # TODO: Get from vision system
            attempts=self.docking_attempts,
            timestamp_ns=time.time_ns(),
        )
    
    def should_interrupt_task(self, battery_percent: float) -> bool:
        """
        Check if current task should be interrupted for charging.
        
        Args:
            battery_percent: Current battery percentage
            
        Returns:
            True if robot should stop current task and charge
        """
        return battery_percent < self.CRITICAL_BATTERY_THRESHOLD
    
    def save_dock_position(self, x: float, y: float, theta: float):
        """
        Save current position as charging dock location.
        Call this when manually positioned at dock.
        
        Args:
            x, y: Position in meters (from odometry)
            theta: Heading in radians
        """
        self.dock_position_saved = (x, y, theta)
        logger.info(f"Dock position saved: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")


if __name__ == "__main__":
    # Test harness
    logging.basicConfig(level=logging.INFO)
    
    print("AutoCharge Behavior Test")
    print("=" * 50)
    print("This is a stub - requires battery monitor and motor controller")
    print("Integration example:")
    print("""
    from continuonbrain.sensors.battery_monitor import BatteryMonitor
    from continuonbrain.behaviors.auto_charge import AutoChargeBehavior
    
    battery = BatteryMonitor()
    motors = MotorController()  # Your motor interface
    charger = AutoChargeBehavior(battery, motors)
    
    # Main loop
    while True:
        status = charger.update()
        print(f"State: {status.state}, Battery: {status.battery_percent}%")
        time.sleep(0.5)
    """)
