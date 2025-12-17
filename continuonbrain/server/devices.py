"""
Device management helpers split from robot_api_server.
"""

from dataclasses import dataclass
from typing import Optional, Dict

from continuonbrain.actuators.pca9685_arm import PCA9685ArmController, ArmConfig
from continuonbrain.actuators.drivetrain_controller import DrivetrainController
from continuonbrain.sensors.oak_depth import OAKDepthCapture, CameraConfig
from continuonbrain.recording.arm_episode_recorder import ArmEpisodeRecorder
from continuonbrain.sensors.hardware_detector import HardwareDetector


@dataclass
class DetectedHardware:
    config: Dict
    devices: list


def auto_detect_hardware() -> DetectedHardware:
    detector = HardwareDetector()
    devices = detector.detect_all()
    cfg = detector.generate_config() if devices else {}
    return DetectedHardware(config=cfg, devices=devices)


def init_recorder(config_dir: str, max_steps: int = 500) -> ArmEpisodeRecorder:
    return ArmEpisodeRecorder(
        episodes_dir=f"{config_dir}/episodes",
        max_steps=max_steps,
    )

