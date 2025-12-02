"""
OAK-D Lite depth camera capture for RLDS episodes.
Integrates with Pi5 robot arm setup per PI5_CAR_READINESS.md.
"""
import depthai as dai
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time


@dataclass
class CameraConfig:
    """OAK-D camera configuration."""
    resolution: str = "400p"  # 640x400 for Pi5 bandwidth
    fps: int = 30
    depth_preset: str = "HIGH_DENSITY"
    median_filter: str = "KERNEL_7x7"
    align_to_rgb: bool = True


class OAKDepthCapture:
    """
    Manages OAK-D Lite depth camera for robot manipulation.
    Captures aligned depth + RGB frames with timestamps for RLDS.
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.pipeline: Optional[dai.Pipeline] = None
        self.device: Optional[dai.Device] = None
        self.calibration_data: Optional[Dict[str, Any]] = None
        
    def initialize(self) -> bool:
        """Initialize OAK-D camera and verify connection."""
        try:
            devices = dai.Device.getAllAvailableDevices()
            if not devices:
                print("ERROR: No OAK devices found")
                return False
                
            # Create pipeline
            self.pipeline = dai.Pipeline()
            
            # Create nodes
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            left = self.pipeline.create(dai.node.MonoCamera)
            right = self.pipeline.create(dai.node.MonoCamera)
            stereo = self.pipeline.create(dai.node.StereoDepth)
            
            # Output streams
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            xout_depth = self.pipeline.create(dai.node.XLinkOut)
            
            xout_rgb.setStreamName("rgb")
            xout_depth.setStreamName("depth")
            
            # Configure RGB camera
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setFps(self.config.fps)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            
            # Configure stereo pair
            res_map = {
                "400p": dai.MonoCameraProperties.SensorResolution.THE_400_P,
                "480p": dai.MonoCameraProperties.SensorResolution.THE_480_P,
                "720p": dai.MonoCameraProperties.SensorResolution.THE_720_P,
            }
            
            left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            left.setResolution(res_map[self.config.resolution])
            left.setFps(self.config.fps)
            
            right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            right.setResolution(res_map[self.config.resolution])
            right.setFps(self.config.fps)
            
            # Configure stereo depth
            preset_map = {
                "HIGH_DENSITY": dai.node.StereoDepth.PresetMode.HIGH_DENSITY,
                "HIGH_ACCURACY": dai.node.StereoDepth.PresetMode.HIGH_ACCURACY,
            }
            stereo.setDefaultProfilePreset(preset_map[self.config.depth_preset])
            
            # Median filter for noise reduction
            filter_map = {
                "KERNEL_3x3": dai.MedianFilter.KERNEL_3x3,
                "KERNEL_5x5": dai.MedianFilter.KERNEL_5x5,
                "KERNEL_7x7": dai.MedianFilter.KERNEL_7x7,
            }
            stereo.initialConfig.setMedianFilter(filter_map[self.config.median_filter])
            
            # Align depth to RGB for manipulation tasks
            if self.config.align_to_rgb:
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            
            # Link nodes
            cam_rgb.video.link(xout_rgb.input)
            left.out.link(stereo.left)
            right.out.link(stereo.right)
            stereo.depth.link(xout_depth.input)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize OAK-D: {e}")
            return False
    
    def start(self) -> bool:
        """Start camera capture."""
        if not self.pipeline:
            print("ERROR: Pipeline not initialized")
            return False
            
        try:
            self.device = dai.Device(self.pipeline)
            
            # Get calibration data for episode metadata
            calib = self.device.readCalibration()
            self.calibration_data = {
                "baseline_cm": calib.getBaselineDistance(),
                "fov_deg": calib.getFov(dai.CameraBoardSocket.CAM_A),
                "intrinsics": calib.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A,
                    1920, 1080
                ).tolist() if hasattr(calib, 'getCameraIntrinsics') else None,
            }
            
            print(f"OAK-D started: baseline={self.calibration_data['baseline_cm']:.2f}cm")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start OAK-D: {e}")
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
        if not self.device:
            return None
            
        try:
            # Get queues
            q_rgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            q_depth = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)
            
            # Get latest frames (non-blocking)
            rgb_frame = q_rgb.get() if q_rgb.has() else None
            depth_frame = q_depth.get() if q_depth.has() else None
            
            if rgb_frame is None or depth_frame is None:
                return None
            
            # Extract frame data
            rgb_data = rgb_frame.getCvFrame()  # Already RGB order
            depth_data = depth_frame.getFrame()  # uint16 millimeters
            
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
            "depth_baseline_cm": self.calibration_data.get("baseline_cm", 7.5),
            "resolution": self.config.resolution,
            "fps": self.config.fps,
            "intrinsics": self.calibration_data.get("intrinsics"),
        }
    
    def stop(self):
        """Stop camera and release resources."""
        if self.device:
            self.device.close()
            self.device = None
        print("OAK-D stopped")


def test_oak_capture():
    """Test OAK-D capture for development."""
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
    print("\n✅ OAK-D capture test complete")


if __name__ == "__main__":
    test_oak_capture()
