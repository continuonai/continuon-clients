"""
Hardware Service

Domain service for hardware control functionality.
Extracted from BrainService hardware initialization and control methods.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from continuonbrain.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class HardwareService:
    """
    Hardware domain service implementing IHardwareService.

    Handles:
    - Actuator control (arm, drivetrain)
    - Sensor reading (camera, depth)
    - Hardware capability detection
    - Safety enforcement
    """

    def __init__(
        self,
        config_dir: str = "/opt/continuonos/brain",
        container: Optional["ServiceContainer"] = None,
        prefer_real_hardware: bool = True,
        auto_detect: bool = True,
        **kwargs,
    ):
        """
        Initialize hardware service.

        Args:
            config_dir: Configuration directory
            container: Service container for dependencies
            prefer_real_hardware: Prefer real hardware over mocks
            auto_detect: Auto-detect hardware on init
        """
        self.config_dir = Path(config_dir)
        self._container = container
        self.prefer_real_hardware = prefer_real_hardware

        # Hardware components
        self._arm_controller = None
        self._drivetrain_controller = None
        self._camera = None
        self._hardware_detector = None

        # Capabilities
        self._capabilities = {
            "has_arm": False,
            "has_drivetrain": False,
            "has_camera": False,
            "has_depth": False,
            "has_hailo": False,
        }

        self._initialized = False

        if auto_detect:
            self._detect_hardware()

    def _detect_hardware(self) -> None:
        """Detect available hardware."""
        if self._initialized:
            return

        self._initialized = True

        try:
            from continuonbrain.sensors.hardware_detector import HardwareDetector

            self._hardware_detector = HardwareDetector()
            detected = self._hardware_detector.detect_all()

            self._capabilities["has_arm"] = detected.get("arm", False)
            self._capabilities["has_drivetrain"] = detected.get("drivetrain", False)
            self._capabilities["has_camera"] = detected.get("camera", False)
            self._capabilities["has_depth"] = detected.get("depth", False)
            self._capabilities["has_hailo"] = detected.get("hailo", False)

            logger.info(f"Hardware detected: {self._capabilities}")

        except ImportError:
            logger.warning("HardwareDetector not available")
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")

        # Initialize detected components
        self._init_arm()
        self._init_drivetrain()
        self._init_camera()

    def _init_arm(self) -> None:
        """Initialize arm controller if available."""
        if not self._capabilities["has_arm"]:
            return

        try:
            from continuonbrain.actuators.pca9685_arm import PCA9685ArmController

            self._arm_controller = PCA9685ArmController()
            logger.info("Arm controller initialized")

        except ImportError:
            logger.debug("Arm controller not available")
        except Exception as e:
            logger.warning(f"Arm init failed: {e}")
            self._capabilities["has_arm"] = False

    def _init_drivetrain(self) -> None:
        """Initialize drivetrain controller if available."""
        if not self._capabilities["has_drivetrain"]:
            return

        try:
            from continuonbrain.actuators.drivetrain_controller import DrivetrainController

            self._drivetrain_controller = DrivetrainController()
            logger.info("Drivetrain controller initialized")

        except ImportError:
            logger.debug("Drivetrain controller not available")
        except Exception as e:
            logger.warning(f"Drivetrain init failed: {e}")
            self._capabilities["has_drivetrain"] = False

    def _init_camera(self) -> None:
        """Initialize camera if available."""
        if not self._capabilities["has_camera"]:
            return

        try:
            from continuonbrain.sensors.oak_depth import OAKDepthCapture

            self._camera = OAKDepthCapture()
            self._capabilities["has_depth"] = True
            logger.info("Camera initialized with depth")

        except ImportError:
            logger.debug("OAK-D camera not available")
        except Exception as e:
            logger.warning(f"Camera init failed: {e}")
            self._capabilities["has_camera"] = False

    def drive(
        self,
        steering: float,
        throttle: float,
        duration_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute drive command with safety checks."""
        if not self._capabilities["has_drivetrain"]:
            return {
                "success": False,
                "actual_steering": 0,
                "actual_throttle": 0,
                "safety_limited": True,
                "error": "Drivetrain not available",
            }

        # Apply safety limits
        steering = max(-1.0, min(1.0, steering))
        throttle = max(-1.0, min(1.0, throttle))

        safety_limited = False
        max_throttle = 0.5  # Safety limit

        if abs(throttle) > max_throttle:
            throttle = max_throttle if throttle > 0 else -max_throttle
            safety_limited = True

        try:
            self._drivetrain_controller.set_velocity(steering, throttle)

            if duration_ms:
                import time
                time.sleep(duration_ms / 1000.0)
                self._drivetrain_controller.set_velocity(0, 0)

            return {
                "success": True,
                "actual_steering": steering,
                "actual_throttle": throttle,
                "safety_limited": safety_limited,
            }

        except Exception as e:
            logger.error(f"Drive command failed: {e}")
            return {
                "success": False,
                "actual_steering": 0,
                "actual_throttle": 0,
                "safety_limited": True,
                "error": str(e),
            }

    def stop(self) -> Dict[str, Any]:
        """Stop all motion immediately."""
        results = {"success": True, "stopped": []}

        if self._drivetrain_controller:
            try:
                self._drivetrain_controller.set_velocity(0, 0)
                results["stopped"].append("drivetrain")
            except Exception as e:
                logger.error(f"Drivetrain stop failed: {e}")
                results["success"] = False

        if self._arm_controller:
            try:
                self._arm_controller.stop()
                results["stopped"].append("arm")
            except Exception as e:
                logger.error(f"Arm stop failed: {e}")
                results["success"] = False

        return results

    def move_arm(
        self,
        joint_positions: List[float],
        speed: float = 0.5,
    ) -> Dict[str, Any]:
        """Move arm to target joint positions."""
        if not self._capabilities["has_arm"]:
            return {
                "success": False,
                "reached_target": False,
                "actual_positions": [],
                "error": "Arm not available",
            }

        try:
            self._arm_controller.move_to(joint_positions, speed=speed)

            return {
                "success": True,
                "reached_target": True,
                "actual_positions": joint_positions,
            }

        except Exception as e:
            logger.error(f"Arm move failed: {e}")
            return {
                "success": False,
                "reached_target": False,
                "actual_positions": [],
                "error": str(e),
            }

    def capture_frame(self) -> Optional[Dict[str, Any]]:
        """Capture RGB+Depth frame from camera."""
        if not self._capabilities["has_camera"]:
            return None

        try:
            import time

            frame_data = self._camera.capture()

            if frame_data is None:
                return None

            return {
                "rgb": frame_data.get("rgb"),
                "depth": frame_data.get("depth"),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None

    def get_capabilities(self) -> Dict[str, bool]:
        """Get hardware capability flags."""
        return dict(self._capabilities)

    def get_status(self) -> Dict[str, Any]:
        """Get current hardware status."""
        return {
            "initialized": self._initialized,
            "capabilities": self._capabilities,
            "arm_connected": self._arm_controller is not None,
            "drivetrain_connected": self._drivetrain_controller is not None,
            "camera_connected": self._camera is not None,
        }

    def is_available(self) -> bool:
        """Check if hardware service is available."""
        return True  # Always available (may have no hardware though)

    def shutdown(self) -> None:
        """Shutdown hardware service."""
        self.stop()

        if self._camera:
            try:
                self._camera.close()
            except Exception:
                pass

        self._arm_controller = None
        self._drivetrain_controller = None
        self._camera = None
        self._initialized = False
