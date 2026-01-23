"""
OAK-D Lite camera integration for trainer UI.
Provides RGB and depth capture from Luxonis OAK-D cameras.
Uses DepthAI 2.x API for better OAK-D Lite compatibility.
"""

import base64
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import depthai as dai
    import cv2
    import numpy as np
    HAS_DEPTHAI = True
except ImportError:
    HAS_DEPTHAI = False
    dai = None
    cv2 = None
    np = None


@dataclass
class OakDFrame:
    """A captured frame from OAK-D."""
    rgb: Optional[bytes] = None
    depth: Optional[bytes] = None
    timestamp: float = 0.0


class OakDCamera:
    """OAK-D Lite camera manager using DepthAI 2.x API."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._capture_thread = None
        self._last_frame: Optional[OakDFrame] = None
        self._error: Optional[str] = None

    def get_available_devices(self) -> list:
        """List available OAK-D devices."""
        if not HAS_DEPTHAI:
            return []

        try:
            devices = dai.Device.getAllAvailableDevices()
            return [
                {
                    "name": d.getMxId() if hasattr(d, 'getMxId') else d.name,
                    "state": d.state.name,
                    "protocol": d.protocol.name if hasattr(d, "protocol") else "USB",
                }
                for d in devices
            ]
        except Exception as e:
            print(f"Error listing devices: {e}")
            return []

    def start(self) -> bool:
        """Start the OAK-D camera pipeline in background thread."""
        if not HAS_DEPTHAI:
            self._error = "DepthAI not available"
            return False

        with self._lock:
            if self._running:
                return True

            self._error = None

            # Start capture in background thread
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

            # Wait for startup
            for _ in range(50):  # Wait up to 5 seconds
                time.sleep(0.1)
                if self._running:
                    return True
                if self._error:
                    return False

            if not self._running:
                self._error = self._error or "Timeout starting camera"
                return False

            return True

    def _capture_loop(self):
        """Background capture loop using DepthAI 2.x API."""
        try:
            print("OAK-D: Creating pipeline (DepthAI 2.x)...")

            # Create pipeline
            pipeline = dai.Pipeline()

            # RGB Camera
            camRgb = pipeline.create(dai.node.ColorCamera)
            camRgb.setPreviewSize(1280, 720)
            camRgb.setInterleaved(False)
            camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            camRgb.setFps(15)

            # Mono cameras for depth
            monoLeft = pipeline.create(dai.node.MonoCamera)
            monoRight = pipeline.create(dai.node.MonoCamera)
            monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
            monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

            # Stereo depth
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            monoLeft.out.link(stereo.left)
            monoRight.out.link(stereo.right)

            # XLinkOut for RGB
            xoutRgb = pipeline.create(dai.node.XLinkOut)
            xoutRgb.setStreamName("rgb")
            camRgb.preview.link(xoutRgb.input)

            # XLinkOut for depth
            xoutDepth = pipeline.create(dai.node.XLinkOut)
            xoutDepth.setStreamName("depth")
            stereo.depth.link(xoutDepth.input)

            print("OAK-D: Connecting...")

            with dai.Device(pipeline) as device:
                print("OAK-D: Connected!")
                self._running = True

                rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

                while self._running:
                    try:
                        frame = OakDFrame(timestamp=time.time())

                        # Get RGB frame
                        inRgb = rgbQueue.tryGet()
                        if inRgb is not None:
                            rgbFrame = inRgb.getCvFrame()
                            _, rgbEncoded = cv2.imencode('.jpg', rgbFrame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            frame.rgb = rgbEncoded.tobytes()

                        # Get depth frame
                        inDepth = depthQueue.tryGet()
                        if inDepth is not None:
                            depthFrame = inDepth.getFrame()
                            # Create colorized depth visualization
                            depthColormap = cv2.applyColorMap(
                                cv2.convertScaleAbs(depthFrame, alpha=0.03),
                                cv2.COLORMAP_JET
                            )
                            _, depthEncoded = cv2.imencode('.png', depthColormap)
                            frame.depth = depthEncoded.tobytes()

                        if frame.rgb or frame.depth:
                            self._last_frame = frame

                        time.sleep(0.033)  # ~30 FPS

                    except Exception as e:
                        print(f"OAK-D capture error: {e}")
                        time.sleep(0.1)

        except Exception as e:
            error_msg = str(e)
            print(f"OAK-D error: {error_msg}")

            if "ALREADY_IN_USE" in error_msg:
                self._error = "Camera in use. Unplug and replug the OAK-D."
            else:
                self._error = error_msg

        finally:
            self._running = False
            print("OAK-D: Stopped")

    def stop(self):
        """Stop the camera."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=3.0)
            self._capture_thread = None
        self._last_frame = None
        print("OAK-D: Stop requested")

    def capture_base64(self, include_depth: bool = True) -> dict:
        """Get the last captured frame as base64."""
        if self._error:
            return {"error": self._error}

        if not self._running:
            return {"error": "Camera not running"}

        frame = self._last_frame
        if frame is None:
            return {"error": "No frame yet (camera warming up)"}

        result = {"timestamp": frame.timestamp}

        if frame.rgb:
            result["rgb"] = base64.b64encode(frame.rgb).decode('utf-8')
            result["rgb_mime"] = "image/jpeg"

        if include_depth and frame.depth:
            result["depth"] = base64.b64encode(frame.depth).decode('utf-8')
            result["depth_mime"] = "image/png"

        return result

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> Optional[str]:
        return self._error


# Singleton instance
_oakd_camera: Optional[OakDCamera] = None


def get_oakd_camera() -> OakDCamera:
    """Get the singleton OAK-D camera instance."""
    global _oakd_camera
    if _oakd_camera is None:
        _oakd_camera = OakDCamera()
    return _oakd_camera
