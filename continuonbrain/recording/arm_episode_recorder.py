"""
RLDS episode recorder for SO-ARM101 robot arm manipulation.
Records depth vision + arm state + human teleop actions.
Compatible with Pi5 setup per PI5_CAR_READINESS.md and docs/rlds-schema.md.
"""
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
import numpy as np

# Local imports (with guards for development)
try:
    from continuonbrain.sensors.oak_depth import OAKDepthCapture, CameraConfig
    from continuonbrain.actuators.pca9685_arm import PCA9685ArmController, ArmConfig
    from continuonbrain.sensors.hardware_detector import HardwareDetector
except ImportError:
    print("Warning: Running in standalone mode")
    OAKDepthCapture = None
    PCA9685ArmController = None
    HardwareDetector = None


@dataclass
class EpisodeMetadata:
    """Episode-level metadata per RLDS schema."""
    episode_id: str
    robot_type: str = "SO-ARM101"
    xr_mode: str = "trainer"  # or "inference" 
    action_source: str = "human_teleop_xr"  # or "vla_policy"
    camera_config: Dict[str, Any] = None
    start_timestamp_ns: int = 0
    end_timestamp_ns: int = 0
    total_steps: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.camera_config is None:
            self.camera_config = {}
        if self.tags is None:
            self.tags = ["pi5", "oak-d-lite", "pca9685"]


@dataclass
class StepData:
    """Single step in an episode per RLDS schema."""
    step_index: int
    timestamp_ns: int
    
    # Observation
    rgb_image: np.ndarray  # (H, W, 3) uint8
    depth_image: np.ndarray  # (H, W) uint16 millimeters
    robot_state: List[float]  # 6 joint angles normalized [-1, 1]
    
    # Action
    action: List[float]  # 6 target joint positions normalized [-1, 1]
    action_source: str  # "human_teleop_xr" or "vla_policy"
    
    # Metadata
    is_terminal: bool = False
    language_instruction: Optional[str] = None
    reward: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "step_index": self.step_index,
            "timestamp_ns": self.timestamp_ns,
            "observation": {
                "rgb_shape": self.rgb_image.shape,
                "depth_shape": self.depth_image.shape,
                "robot_state": self.robot_state,
            },
            "action": self.action,
            "action_source": self.action_source,
            "is_terminal": self.is_terminal,
            "language_instruction": self.language_instruction,
            "reward": self.reward,
        }


class ArmEpisodeRecorder:
    """
    Records robot arm manipulation episodes for RLDS training.
    Integrates OAK-D depth + PCA9685 arm control.
    """
    
    def __init__(
        self,
        episodes_dir: str = "/opt/continuonos/brain/rlds/episodes",
        max_steps: int = 500,
    ):
        self.episodes_dir = Path(episodes_dir)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_steps = max_steps
        
        # Hardware components (optional for testing)
        self.camera: Optional[OAKDepthCapture] = None
        self.arm: Optional[PCA9685ArmController] = None
        
        # Current episode state
        self.current_episode: Optional[str] = None
        self.episode_metadata: Optional[EpisodeMetadata] = None
        self.steps: List[StepData] = []
        self.step_counter: int = 0
        
    def initialize_hardware(self, use_mock: bool = False, auto_detect: bool = True) -> bool:
        """
        Initialize camera and arm hardware.
        
        Args:
            use_mock: Force mock mode (no real hardware)
            auto_detect: Auto-detect available hardware
        """
        if use_mock:
            print("Using MOCK hardware mode")
            return True
        
        success = True
        
        # Auto-detect hardware if requested
        detected_config = {}
        if auto_detect and HardwareDetector:
            print("🔍 Auto-detecting hardware...")
            detector = HardwareDetector()
            devices = detector.detect_all()
            detected_config = detector.generate_config()
            print()
        
        # Initialize camera (auto-select based on detection)
        if OAKDepthCapture:
            camera_type = None
            if detected_config.get("primary", {}).get("depth_camera_driver") == "depthai":
                camera_type = "OAK-D"
                print(f"Using detected {detected_config['primary']['depth_camera']}")
            
            self.camera = OAKDepthCapture(CameraConfig())
            if self.camera.initialize():
                self.camera.start()
                print(f"✅ {camera_type or 'Depth'} camera initialized")
            else:
                print("⚠️  Camera initialization failed (continuing without camera)")
                self.camera = None
                success = False
        
        # Initialize servo controller (auto-select based on detection)
        if PCA9685ArmController:
            servo_address = None
            if "servo_controller" in detected_config.get("devices", {}):
                servo_info = detected_config["devices"]["servo_controller"][0]
                servo_address = servo_info.get("address")
                print(f"Using detected {servo_info['name']} at {servo_address}")
            
            self.arm = PCA9685ArmController(ArmConfig())
            if self.arm.initialize():
                print("✅ Arm controller initialized")
            else:
                print("⚠️  Arm initialization failed (continuing without arm)")
                self.arm = None
                success = False
        
        return success
    
    def start_episode(
        self,
        episode_id: Optional[str] = None,
        language_instruction: Optional[str] = None,
        action_source: str = "human_teleop_xr",
    ) -> str:
        """
        Start a new episode recording.
        
        Args:
            episode_id: Optional custom episode ID
            language_instruction: Optional task description
            action_source: "human_teleop_xr" or "vla_policy"
        
        Returns:
            Episode ID
        """
        if self.current_episode:
            print(f"Warning: Episode {self.current_episode} still active, closing it")
            self.end_episode()
        
        # Generate episode ID
        if not episode_id:
            timestamp = int(time.time())
            episode_id = f"arm_episode_{timestamp}"
        
        self.current_episode = episode_id
        self.step_counter = 0
        self.steps = []
        
        # Create metadata
        camera_config = {}
        if self.camera:
            camera_config = self.camera.get_camera_metadata()
        
        self.episode_metadata = EpisodeMetadata(
            episode_id=episode_id,
            robot_type="SO-ARM101",
            xr_mode="trainer",
            action_source=action_source,
            camera_config=camera_config,
            start_timestamp_ns=time.time_ns(),
            tags=["pi5", "oak-d-lite", "pca9685", "manipulation"],
        )
        
        print(f"\n📹 Started episode: {episode_id}")
        if language_instruction:
            print(f"   Task: {language_instruction}")
        
        return episode_id
    
    def record_step(
        self,
        action: List[float],
        action_source: str = "human_teleop_xr",
        language_instruction: Optional[str] = None,
        is_terminal: bool = False,
    ) -> bool:
        """
        Record a single step (observation + action).
        
        Args:
            action: 6D normalized action vector [-1, 1]
            action_source: Source of the action
            language_instruction: Optional per-step instruction
            is_terminal: Whether this is the final step
        
        Returns:
            True if successful
        """
        if not self.current_episode:
            print("ERROR: No active episode")
            return False
        
        if self.step_counter >= self.max_steps:
            print(f"WARNING: Max steps ({self.max_steps}) reached")
            return False
        
        try:
            timestamp_ns = time.time_ns()
            
            # Capture observation
            rgb_image = None
            depth_image = None
            
            if self.camera:
                frame = self.camera.capture_frame()
                if frame:
                    rgb_image = frame['rgb']
                    depth_image = frame['depth']
            
            # Generate mock data if no camera
            if rgb_image is None:
                rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
                depth_image = np.zeros((480, 640), dtype=np.uint16)
            
            # Get current robot state
            robot_state = [0.0] * 6
            if self.arm:
                robot_state = self.arm.get_normalized_state()
            
            # Execute action on arm
            if self.arm and not is_terminal:
                self.arm.set_normalized_action(action)
            
            # Create step data
            step = StepData(
                step_index=self.step_counter,
                timestamp_ns=timestamp_ns,
                rgb_image=rgb_image,
                depth_image=depth_image,
                robot_state=robot_state,
                action=action,
                action_source=action_source,
                is_terminal=is_terminal,
                language_instruction=language_instruction,
                reward=1.0 if is_terminal else 0.0,  # Simple sparse reward
            )
            
            self.steps.append(step)
            self.step_counter += 1
            
            if self.step_counter % 10 == 0:
                print(f"   Step {self.step_counter}/{self.max_steps}")
            
            return True
            
        except Exception as e:
            print(f"ERROR recording step: {e}")
            return False
    
    def end_episode(self, success: bool = True) -> Optional[Path]:
        """
        End current episode and save to disk.
        
        Args:
            success: Whether episode completed successfully
        
        Returns:
            Path to saved episode file, or None if failed
        """
        if not self.current_episode:
            print("ERROR: No active episode")
            return None
        
        try:
            # Update metadata
            self.episode_metadata.end_timestamp_ns = time.time_ns()
            self.episode_metadata.total_steps = len(self.steps)
            
            if not success:
                self.episode_metadata.tags.append("failed")
            
            # Prepare episode data
            episode_data = {
                "metadata": asdict(self.episode_metadata),
                "steps": [step.to_dict() for step in self.steps],
            }
            
            # Save metadata JSON
            episode_path = self.episodes_dir / self.current_episode
            episode_path.mkdir(exist_ok=True)
            
            metadata_file = episode_path / "episode.json"
            with open(metadata_file, 'w') as f:
                json.dump(episode_data, f, indent=2)
            
            # Save numpy arrays separately (more efficient)
            for i, step in enumerate(self.steps):
                np.save(episode_path / f"step_{i:04d}_rgb.npy", step.rgb_image)
                np.save(episode_path / f"step_{i:04d}_depth.npy", step.depth_image)
            
            duration_s = (self.episode_metadata.end_timestamp_ns - 
                         self.episode_metadata.start_timestamp_ns) / 1e9
            
            print(f"\n✅ Episode saved: {self.current_episode}")
            print(f"   Steps: {len(self.steps)}")
            print(f"   Duration: {duration_s:.1f}s")
            print(f"   Path: {episode_path}")
            
            # Reset state
            self.current_episode = None
            self.episode_metadata = None
            self.steps = []
            self.step_counter = 0
            
            return episode_path
            
        except Exception as e:
            print(f"ERROR saving episode: {e}")
            return None
    
    def shutdown(self):
        """Shutdown hardware gracefully."""
        if self.current_episode:
            print("Saving active episode before shutdown...")
            self.end_episode(success=False)
        
        if self.camera:
            self.camera.stop()
        
        if self.arm:
            self.arm.shutdown()
        
        print("✅ Recorder shutdown complete")


def test_recorder():
    """Test episode recorder with mock data."""
    print("Testing Arm Episode Recorder...")
    
    recorder = ArmEpisodeRecorder(
        episodes_dir="/tmp/test_episodes",
        max_steps=20,
    )
    
    # Initialize in mock mode
    recorder.initialize_hardware(use_mock=True)
    
    # Record a test episode
    recorder.start_episode(
        episode_id="test_pick_and_place",
        language_instruction="Pick up the red cube and place it in the bin",
    )
    
    # Simulate 15 steps of manipulation
    for i in range(15):
        # Generate random action
        action = [np.random.uniform(-0.5, 0.5) for _ in range(6)]
        
        is_terminal = (i == 14)
        
        recorder.record_step(
            action=action,
            action_source="human_teleop_xr",
            is_terminal=is_terminal,
        )
        
        time.sleep(0.05)  # Simulate 20Hz control
    
    # Save episode
    episode_path = recorder.end_episode(success=True)
    
    if episode_path:
        print(f"\n✅ Test episode saved to: {episode_path}")
        
        # Verify saved data
        metadata_file = episode_path / "episode.json"
        with open(metadata_file) as f:
            data = json.load(f)
        
        print(f"\nEpisode metadata:")
        print(f"  Steps: {data['metadata']['total_steps']}")
        print(f"  Tags: {data['metadata']['tags']}")
        print(f"  Camera: {data['metadata']['camera_config']}")
    
    recorder.shutdown()
    print("\n✅ Recorder test complete")


if __name__ == "__main__":
    test_recorder()
