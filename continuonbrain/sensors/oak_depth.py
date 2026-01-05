"""
OAK-D Lite depth camera capture for RLDS episodes.
Integrates with Pi5 robot arm setup per PI5_EDGE_BRAIN_INSTRUCTIONS.md.
Updated for DepthAI v3.x API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional numeric dependency
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
    logger.warning("numpy not available; OAK-D utilities will operate in noop mode")

try:  # pragma: no cover - optional hardware dependency
    import depthai as dai
except ImportError:  # pragma: no cover
    dai = None
    logger.warning("depthai package not available; OAK-D support will be disabled")


@dataclass
class CameraConfig:
    """OAK-D camera configuration."""
    resolution: Tuple[int, int] = (640, 400)  # Width x Height for Pi5 bandwidth
    fps: int = 30
    depth_preset: str = "ROBOTICS"  # ROBOTICS, FAST_DENSITY, FAST_ACCURACY, HIGH_DETAIL
    median_filter: str = "KERNEL_7x7"
    align_to_rgb: bool = True


class OAKDepthCapture:
    """
    Manages OAK-D Lite depth camera for robot manipulation.
    Captures aligned depth + RGB frames with timestamps for RLDS.
    Uses DepthAI v3.x API.
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.device: Optional["dai.Device"] = None
        self.pipeline: Optional["dai.Pipeline"] = None
        self.calibration_data: Optional[Dict[str, Any]] = None
        self.rgb_queue = None
        self.depth_queue = None
        self._running = False

    @property
    def dependencies_ready(self) -> bool:
        """Return True when required hardware/software is available."""
        if np is None:
            logger.warning("numpy missing; cannot process OAK-D frames safely")
            return False
        if dai is None:
            logger.warning("DepthAI SDK missing; OAK-D support disabled")
            return False
        return True
        
    def initialize(self) -> bool:
        """Initialize OAK-D camera and verify connection (DepthAI v3 API)."""
        if not self.dependencies_ready:
            return False

        try:
            # V3 API: Create device first
            devices = dai.Device.getAllAvailableDevices()
            if not devices:
                print("  No OAK devices found (DepthAI reported zero devices)")
                return False
                
            self.device = dai.Device()
            print(f" Device connected: {self.device.getDeviceName()}")
            
            # V3 API: Create pipeline with device context
            self.pipeline = dai.Pipeline(self.device)
            
            # Create RGB camera node using v3 builder pattern
            rgb_cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            
            # Request RGB output at configured resolution
            rgb_output = rgb_cam.requestOutput(self.config.resolution, dai.ImgFrame.Type.BGR888p)
            self.rgb_queue = rgb_output.createOutputQueue()
            
            # Create stereo cameras for depth (use requestOutput to avoid stride mismatch)
            left_cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right_cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            
            # Request mono outputs at same resolution as RGB (for stereo alignment)
            left_output = left_cam.requestOutput(self.config.resolution, dai.ImgFrame.Type.GRAY8)
            right_output = right_cam.requestOutput(self.config.resolution, dai.ImgFrame.Type.GRAY8)
            
            # Create stereo depth node
            stereo = self.pipeline.create(dai.node.StereoDepth)
            
            # Configure stereo depth
            # V3 preset modes: DEFAULT, FACE, FAST_ACCURACY, FAST_DENSITY, HIGH_DETAIL, ROBOTICS
            preset_map = {
                "ROBOTICS": dai.node.StereoDepth.PresetMode.ROBOTICS,
                "FAST_DENSITY": dai.node.StereoDepth.PresetMode.FAST_DENSITY,
                "FAST_ACCURACY": dai.node.StereoDepth.PresetMode.FAST_ACCURACY,
                "HIGH_DETAIL": dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
            }
            preset_mode = preset_map.get(self.config.depth_preset, dai.node.StereoDepth.PresetMode.ROBOTICS)
            stereo.setDefaultProfilePreset(preset_mode)
            
            # Median filter for noise reduction
            filter_map = {
                "KERNEL_3x3": dai.MedianFilter.KERNEL_3x3,
                "KERNEL_5x5": dai.MedianFilter.KERNEL_5x5,
                "KERNEL_7x7": dai.MedianFilter.KERNEL_7x7,
            }
            stereo.initialConfig.setMedianFilter(filter_map[self.config.median_filter])
            
            # Set stereo output size to match our resolution (must be multiple of 16)
            width = (self.config.resolution[0] // 16) * 16
            height = (self.config.resolution[1] // 16) * 16
            stereo.setOutputSize(width, height)
            
            # Align depth to RGB for manipulation tasks
            if self.config.align_to_rgb:
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            
            # V3 API: Link requested outputs to stereo (these are already the right size)
            left_output.link(stereo.left)
            right_output.link(stereo.right)
            
            # Get depth output queue
            depth_output = stereo.depth
            self.depth_queue = depth_output.createOutputQueue()
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize OAK-D: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start(self) -> bool:
        """Start camera capture (DepthAI v3 API)."""
        if not self.dependencies_ready:
            return False
        if not self.pipeline or not self.device:
            print("ERROR: Pipeline or device not initialized")
            return False
            
        try:
            # V3 API: Start the pipeline
            self.pipeline.start()
            self._running = True
            
            # Get calibration data for episode metadata
            calib = self.device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A,
                1920, 1080
            ) if hasattr(calib, 'getCameraIntrinsics') else None
            
            self.calibration_data = {
                "baseline_cm": calib.getBaselineDistance(),
                "fov_deg": calib.getFov(dai.CameraBoardSocket.CAM_A),
                "intrinsics": intrinsics if intrinsics is None or isinstance(intrinsics, list) else intrinsics.tolist(),
            }
            
            print(f"OAK-D started: baseline={self.calibration_data['baseline_cm']:.2f}cm")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start OAK-D: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def capture_frame(self) -> Optional[Dict[str, Any]]:
        """
        Capture aligned RGB + depth frame with timestamp.
        
        Returns dict compatible with RLDS observation schema:
        {
            'rgb': np.ndarray,  # (H, W, 3) uint8
            'depth': np.ndarray,  # (H, W) uint16 millimeters
            'timestamp_ns': int,  # nanoseconds since epoch
        }
        """
        if not self.dependencies_ready:
            return None
        if not self.device or not self.rgb_queue or not self.depth_queue:
            return None
            
        try:
            # Get latest frames with blocking timeout for reliability
            # Non-blocking has() can miss frames; use tryGet() or blocking get()
            try:
                rgb_frame = self.rgb_queue.tryGet()
                if rgb_frame is None:
                    # Try blocking with short timeout if non-blocking fails
                    rgb_frame = self.rgb_queue.get()
            except Exception:
                rgb_frame = None

            try:
                depth_frame = self.depth_queue.tryGet()
                if depth_frame is None:
                    depth_frame = self.depth_queue.get()
            except Exception:
                depth_frame = None

            if rgb_frame is None:
                return None

            # Depth is optional - allow RGB-only capture
            if depth_frame is None:
                depth_frame = None
            
            # Extract frame data
            rgb_data = rgb_frame.getCvFrame()  # Already BGR order from getCvFrame
            depth_data = depth_frame.getFrame() if depth_frame is not None else None  # uint16 millimeters
            
            # Timestamp alignment (use depth timestamp as reference)
            timestamp_ns = int(time.time_ns())
            
            return {
                'rgb': rgb_data,
                'depth': depth_data,
                'timestamp_ns': timestamp_ns,
            }
            
        except Exception as e:
            print(f"ERROR: Failed to capture frame: {e}")
            return None
    
    def get_camera_metadata(self) -> Dict[str, Any]:
        """Get camera calibration metadata for episode_metadata."""
        return {
            "camera_type": "OAK-D-LITE",
            "depth_baseline_cm": self.calibration_data.get("baseline_cm", 7.5) if self.calibration_data else 7.5,
            "resolution": f"{self.config.resolution[0]}x{self.config.resolution[1]}",
            "fps": self.config.fps,
            "intrinsics": self.calibration_data.get("intrinsics") if self.calibration_data else None,
        }
    
    def stop(self):
        """Stop camera and release resources."""
        self._running = False
        if self.pipeline and self.pipeline.isRunning():
            # V3 API handles cleanup automatically
            pass
        if self.device:
            self.device.close()
            self.device = None
        print("OAK-D stopped")


def test_oak_capture():
    """Test OAK-D capture for development."""
    if np is None or dai is None:
        print("Dependencies missing; skipping OAK-D capture test")
        return

    camera = OAKDepthCapture()
    
    if not camera.initialize():
        print("Failed to initialize camera")
        return
    
    if not camera.start():
        print("Failed to start camera")
        return
    
    print(f"Camera metadata: {camera.get_camera_metadata()}")
    print("\nCapturing 10 test frames...")
    
    for i in range(10):
        frame = camera.capture_frame()
        if frame:
            rgb_shape = frame['rgb'].shape
            depth_shape = frame['depth'].shape
            depth_valid = np.sum(frame['depth'] > 0)
            coverage = (depth_valid / frame['depth'].size) * 100
            
            print(f"Frame {i+1}: RGB={rgb_shape}, Depth={depth_shape}, "
                  f"coverage={coverage:.1f}%, ts={frame['timestamp_ns']}")
        else:
            print(f"Frame {i+1}: No data")
        
        time.sleep(0.1)
    
    camera.stop()
    print("\n OAK-D capture test complete")


if __name__ == "__main__":
    test_oak_capture()
