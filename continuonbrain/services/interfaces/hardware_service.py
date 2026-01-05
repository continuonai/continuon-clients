"""
Hardware Service Interface

Protocol definition for hardware control services.
"""
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class IHardwareService(Protocol):
    """
    Protocol for hardware control services.

    Implementations handle:
    - Actuator control (arm, drivetrain)
    - Sensor reading (camera, depth)
    - Hardware capability detection
    - Safety enforcement
    """

    def drive(
        self,
        steering: float,
        throttle: float,
        duration_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute drive command with safety checks.

        Args:
            steering: Steering value (-1.0 to 1.0, left to right)
            throttle: Throttle value (-1.0 to 1.0, reverse to forward)
            duration_ms: Optional duration in milliseconds

        Returns:
            Dictionary containing:
                - success: bool - Whether command was executed
                - actual_steering: float - Actual steering applied
                - actual_throttle: float - Actual throttle applied
                - safety_limited: bool - Whether safety limits were applied
        """
        ...

    def stop(self) -> Dict[str, Any]:
        """
        Stop all motion immediately.

        Returns:
            Dictionary with stop confirmation
        """
        ...

    def move_arm(
        self,
        joint_positions: List[float],
        speed: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Move arm to target joint positions.

        Args:
            joint_positions: List of joint angles/positions
            speed: Movement speed (0.0 to 1.0)

        Returns:
            Dictionary containing:
                - success: bool - Whether command was executed
                - reached_target: bool - Whether target was reached
                - actual_positions: list - Actual joint positions
        """
        ...

    def capture_frame(self) -> Optional[Dict[str, Any]]:
        """
        Capture RGB+Depth frame from camera.

        Returns:
            Dictionary containing:
                - rgb: ndarray - RGB image (H, W, 3)
                - depth: ndarray - Depth map (H, W)
                - timestamp: float - Capture timestamp
            Or None if camera not available
        """
        ...

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get hardware capability flags.

        Returns:
            Dictionary of capability name -> available boolean:
                - has_arm: bool
                - has_drivetrain: bool
                - has_camera: bool
                - has_depth: bool
                - has_hailo: bool
        """
        ...

    def get_status(self) -> Dict[str, Any]:
        """
        Get current hardware status.

        Returns:
            Dictionary containing status for each hardware component
        """
        ...

    def is_available(self) -> bool:
        """Check if hardware service is available."""
        ...
