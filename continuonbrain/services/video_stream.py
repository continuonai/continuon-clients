import asyncio
import time
from typing import AsyncIterator, Optional

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


class VideoStreamHelper:
    """Camera and state streaming helpers for RobotService."""

    def __init__(self, service) -> None:
        self.service = service

    async def stream_robot_state(self, client_id: str) -> AsyncIterator[dict]:
        """Yield robot state at ~20Hz."""
        print(f"Client {client_id} subscribed to robot state stream")

        while True:
            try:
                if not self.service.system_instructions:
                    yield {"success": False, "message": "System instructions unavailable"}
                    return

                normalized_state = (
                    self.service.arm.get_normalized_state() if self.service.arm else [0.0] * 6
                )
                gripper_open = normalized_state[5] < 0.0

                state = {
                    "timestamp_nanos": time.time_ns(),
                    "joint_positions": normalized_state,
                    "gripper_open": gripper_open,
                    "frame_id": f"state_{int(time.time())}",
                    "wall_time_millis": int(time.time() * 1000),
                }

                yield state
                await asyncio.sleep(0.05)

            except Exception as exc:  # noqa: BLE001
                print(f"Error streaming state: {exc}")
                break

    async def get_depth_frame(self) -> Optional[dict]:
        """Get latest depth camera frame metadata."""
        if not self.service.camera or cv2 is None:
            return None

        frame = self.service.camera.capture_frame()
        if not frame:
            return None

        return {
            "timestamp_nanos": frame["timestamp_ns"],
            "rgb_shape": frame["rgb"].shape,
            "depth_shape": frame["depth"].shape,
            "has_data": True,
        }

    async def get_camera_frame_jpeg(self) -> Optional[bytes]:
        """Get latest RGB frame as JPEG bytes."""
        if not self.service.camera:
            return None

        try:
            frame = self.service.camera.capture_frame()
            if not frame or "rgb" not in frame:
                return None

            rgb_frame = frame["rgb"]
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

            success, jpeg_bytes = cv2.imencode(".jpg", rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                return jpeg_bytes.tobytes()
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"Error encoding camera frame: {exc}")
            return None

