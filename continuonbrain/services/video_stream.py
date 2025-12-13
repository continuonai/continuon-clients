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
        if not self.service.camera:
            return None

        frame = self.service.camera.capture_frame()
        if not frame or frame.get("rgb") is None or frame.get("depth") is None:
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
            if not frame or frame.get("rgb") is None:
                return None

            img = frame["rgb"]  # OAKDepthCapture requests BGR888p

            # Fast path: OpenCV available
            if cv2 is not None:
                ok, jpeg_bytes = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return jpeg_bytes.tobytes() if ok else None

            # Fallback: Pillow (no OpenCV on minimal installs)
            try:
                from PIL import Image  # type: ignore
                import io

                # Convert BGR -> RGB
                rgb = img[..., ::-1]
                pil = Image.fromarray(rgb)
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=85)
                return buf.getvalue()
            except Exception as exc:  # noqa: BLE001
                print(f"Error encoding camera frame (no cv2): {exc}")
                return None
        except Exception as exc:  # noqa: BLE001
            print(f"Error encoding camera frame: {exc}")
            return None

