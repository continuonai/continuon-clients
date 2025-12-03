"""
RLDS episode recorder for SO-ARM101 robot arm manipulation.
"""
Records depth vision + arm state + human teleop actions.
Compatible with Pi5 setup per PI5_CAR_READINESS.md and docs/rlds-schema.md.
"""
import importlib
import importlib.util
import json
import threading
import shutil
import time
import wave
from collections import deque
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


def _load_sounddevice():
    """Load optional sounddevice dependency without hard failing."""
    sd_spec = importlib.util.find_spec("sounddevice")
    if not sd_spec:
        return None
    return importlib.import_module("sounddevice")


@dataclass
class EpisodeMetadata:
    """Episode-level metadata per RLDS schema."""
    episode_id: str
    robot_type: str = "SO-ARM101"
    xr_mode: str = "trainer"  # or "inference"
    control_role: str = "human_teleop"  # aligns with docs/rlds-schema.md
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

    def to_dict(self, language_instruction: Optional[str] = None) -> Dict[str, Any]:
        """Serialize metadata with continuon.* tags for downstream stratification."""
        metadata = asdict(self)
        # Namespaced continuon block mirrors docs/rlds-schema.md
        metadata["continuon"] = {
            "xr_mode": self.xr_mode,
            "control_role": self.control_role,
        }

        # Attach instruction text to tags for easier filtering.
        if language_instruction:
            metadata["tags"].append(f"instruction:{language_instruction}")

        # Mirror xr_mode into tags for lightweight queries.
        metadata["tags"].append(f"continuon.xr_mode:{self.xr_mode}")
        return metadata


@dataclass
class StepData:
    """Single step in an episode per RLDS schema."""
    step_index: int
    timestamp_ns: int

    frame_timestamp_ns: int
    video_frame_id: str
    depth_frame_id: str
    robot_state_timestamp_ns: int
    robot_state_frame_id: str
    action_timestamp_ns: int

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
    audio_buffer: Optional[np.ndarray] = None
    audio_uri: Optional[str] = None
    audio_sample_rate_hz: Optional[int] = None
    audio_num_channels: Optional[int] = None
    audio_format: Optional[str] = None
    audio_frame_id: Optional[str] = None
    audio_timestamp_ns: Optional[int] = None
    audio_delta_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        data = {
    step_metadata: Dict[str, str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        step_metadata = self.step_metadata or {}
        return {
            "step_index": self.step_index,
            "timestamp_ns": self.timestamp_ns,
            "observation": {
                "frame_timestamp_ns": self.frame_timestamp_ns,
                "video_frame_id": self.video_frame_id,
                "depth_frame_id": self.depth_frame_id,
                "rgb_shape": self.rgb_image.shape,
                "depth_shape": self.depth_image.shape,
                "robot_state": {
                    "joint_positions": self.robot_state,
                    "frame_id": self.robot_state_frame_id,
                    "timestamp_nanos": self.robot_state_timestamp_ns,
                },
            },
            "action": {
                "command": self.action,
                "source": self.action_source,
                "timestamp_nanos": self.action_timestamp_ns,
            },
            "is_terminal": self.is_terminal,
            "language_instruction": self.language_instruction,
            "reward": self.reward,
            "step_metadata": step_metadata,
        }

        if self.audio_buffer is not None or self.audio_uri:
            data["audio"] = {
                "uri": self.audio_uri,
                "sample_rate_hz": self.audio_sample_rate_hz,
                "num_channels": self.audio_num_channels,
                "format": self.audio_format,
                "frame_id": self.audio_frame_id,
                "timestamp_ns": self.audio_timestamp_ns,
                "delta_ms_to_frame": self.audio_delta_ms,
            }

        return data


class MicrophoneCapture:
    """Lightweight microphone sampler with optional mock fallback."""

    def __init__(
        self,
        sample_rate_hz: int = 16000,
        num_channels: int = 1,
        block_duration_ms: int = 20,
        use_mock: bool = False,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.num_channels = num_channels
        self.block_duration_ms = block_duration_ms
        self.block_frames = int(self.sample_rate_hz * self.block_duration_ms / 1000)
        self.audio_format = "wav_pcm16"
        self.file_extension = "wav"
        self.use_mock = use_mock

        self.stream = None
        self.buffer = deque(maxlen=100)
        self.frame_counter = 0
        self.lock = threading.Lock()
        self.active = False

        self.sd = _load_sounddevice()
        if self.sd is None:
            self.use_mock = True

    def initialize(self) -> bool:
        if self.use_mock:
            print("🎙️  Using mock microphone capture")
            self.active = True
            return True

        try:
            self.stream = self.sd.InputStream(
                samplerate=self.sample_rate_hz,
                channels=self.num_channels,
                blocksize=self.block_frames,
                callback=self._callback,
            )
            self.stream.start()
            self.active = True
            print(f"✅ Microphone initialized at {self.sample_rate_hz} Hz")
            return True
        except Exception as e:
            print(f"⚠️  Microphone init failed: {e}")
            self.active = False
            self.stream = None
            return False

    def _callback(self, indata, frames, callback_time, status):
        del status, frames  # unused
        timestamp_ns = time.time_ns()

        if callback_time and getattr(callback_time, "inputBufferAdcTime", None):
            timestamp_ns = int(callback_time.inputBufferAdcTime * 1e9)

        with self.lock:
            self.buffer.append((timestamp_ns, indata.copy(), self.frame_counter))
            self.frame_counter += 1

    def get_aligned_block(self, target_timestamp_ns: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve the audio block closest to the target timestamp.

        Finds the audio buffer with timestamp nearest to the target, enabling
        synchronization with vision frames. Reports alignment delta for QA.

        Args:
            target_timestamp_ns (int): Target timestamp in nanoseconds to align to.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing:
                - buffer: Audio data as numpy array
                - timestamp_ns: Actual timestamp of the audio block
                - frame_id: String identifier for the audio frame
                - delta_ms: Time difference from target in milliseconds
            Returns None if no audio is available or capture is inactive.
        """
        if not self.active:
            return None

        if self.use_mock:
            with self.lock:
                timestamp_ns = target_timestamp_ns
                mock_audio = np.zeros((self.block_frames, self.num_channels), dtype=np.float32)
            timestamp_ns = target_timestamp_ns
            mock_audio = np.zeros((self.block_frames, self.num_channels), dtype=np.float32)
            with self.lock:
                frame_id = self.frame_counter
                self.frame_counter += 1
            return {
                "buffer": mock_audio,
                "timestamp_ns": timestamp_ns,
                "frame_id": f"audio_{frame_id:04d}",
                "delta_ms": 0.0,
            }

        with self.lock:
            if not self.buffer:
                return None

            closest_ts, closest_buffer, frame_id = min(
                self.buffer, key=lambda entry: abs(entry[0] - target_timestamp_ns)
            )

        delta_ms = abs(closest_ts - target_timestamp_ns) / 1e6
        return {
            "buffer": closest_buffer,
            "timestamp_ns": closest_ts,
            "frame_id": f"audio_{frame_id:04d}",
            "delta_ms": delta_ms,
        }

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.active = False


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
        self.microphone: Optional["MicrophoneCapture"] = None

        # Current episode state
        self.current_episode: Optional[str] = None
        self.episode_metadata: Optional[EpisodeMetadata] = None
        self.steps: List[StepData] = []
        self.step_counter: int = 0
        self.audio_enabled: bool = False

    def _save_audio_file(
        self, file_path: Path, audio_buffer: np.ndarray, sample_rate_hz: int, num_channels: int
    ) -> None:
        """Persist audio buffer to a WAV file.
        
        Args:
            file_path: Path where the WAV file will be saved
            audio_buffer: Audio data as numpy array (float32 or int16)
            sample_rate_hz: Sample rate in Hz
            num_channels: Number of audio channels
            
        Raises:
            IOError: If file cannot be created or written
        """
        if audio_buffer.dtype != np.int16:
            clipped = np.clip(audio_buffer, -1.0, 1.0)
            audio_int16 = (clipped * 32767).astype(np.int16)
        else:
            audio_int16 = audio_buffer

        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate_hz)
            wav_file.writeframes(audio_int16.tobytes())
        self.episode_language_instruction: Optional[str] = None
        
    def initialize_hardware(self, use_mock: bool = False, auto_detect: bool = True) -> bool:
        """
        Initialize camera and arm hardware.
        
        Args:
            use_mock: Force mock mode (no real hardware)
            auto_detect: Auto-detect available hardware
        """
        success = True

        if use_mock:
            print("Using MOCK hardware mode")
        else:
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

        # Initialize microphone capture
        self.microphone = MicrophoneCapture(use_mock=use_mock)
        audio_ready = self.microphone.initialize()
        self.audio_enabled = audio_ready
        if audio_ready:
            print("✅ Microphone capture initialized")
        else:
            print("⚠️  Microphone capture unavailable (audio disabled)")
            self.microphone = None

        return success
    
    def start_episode(
        self,
        episode_id: Optional[str] = None,
        language_instruction: Optional[str] = None,
        action_source: str = "human_teleop_xr",
        xr_mode: str = "trainer",
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
        self.episode_language_instruction = language_instruction

        # Create metadata
        camera_config = {}
        if self.camera:
            camera_config = self.camera.get_camera_metadata()

        self.episode_metadata = EpisodeMetadata(
            episode_id=episode_id,
            robot_type="SO-ARM101",
            xr_mode=xr_mode,
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
        ball_reached: bool = False,
        safety_violations: Optional[List[str]] = None,
        step_metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Record a single step (observation + action).

        Args:
            action: 6D normalized action vector [-1, 1]
            action_source: Source of the action
            language_instruction: Optional per-step instruction
            is_terminal: Whether this is the final step
            ball_reached: Flag set when the ball target is reached (auto-terminal)
            safety_violations: Optional list of safety violations encountered
            step_metadata: Additional step-level metadata tags
        
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
            audio_sample = None
            frame_timestamp_ns = timestamp_ns

            if self.camera:
                frame = self.camera.capture_frame()
                if frame:
                    rgb_image = frame['rgb']
                    depth_image = frame['depth']
                    frame_timestamp_ns = frame.get('timestamp_ns', timestamp_ns)

            # Generate mock data if no camera
            if rgb_image is None:
                rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
                depth_image = np.zeros((480, 640), dtype=np.uint16)

            frame_id = f"frame_{frame_timestamp_ns}"

            # Get current robot state
            robot_state = [0.0] * 6
            robot_state_timestamp_ns = frame_timestamp_ns
            if self.arm:
                robot_state = self.arm.get_normalized_state()

            # Fetch aligned audio block
            audio_uri = None
            audio_buffer = None
            audio_frame_id = None
            audio_timestamp_ns = None
            audio_delta_ms = None
            audio_sample_rate_hz = None
            audio_num_channels = None
            audio_format = None

            if self.microphone:
                audio_sample = self.microphone.get_aligned_block(timestamp_ns)
                if audio_sample:
                    audio_buffer = audio_sample["buffer"]
                    audio_timestamp_ns = audio_sample.get("timestamp_ns", timestamp_ns)
                    audio_frame_id = audio_sample.get("frame_id")
                    audio_delta_ms = audio_sample.get("delta_ms")
                    audio_sample_rate_hz = self.microphone.sample_rate_hz
                    audio_num_channels = self.microphone.num_channels
                    audio_format = self.microphone.audio_format
                    audio_uri = f"step_{self.step_counter:04d}_audio.{self.microphone.file_extension}"

                    if audio_delta_ms is not None and audio_delta_ms > 5.0:
                        print(f"⚠️  Audio/vision misalignment: {audio_delta_ms:.1f} ms")
            
                robot_state_timestamp_ns = time.time_ns()

            # Execute action on arm
            if self.arm and not is_terminal:
                self.arm.set_normalized_action(action)

            # Aggregate step metadata and safety flags
            combined_metadata: Dict[str, str] = {}
            if step_metadata:
                combined_metadata.update({k: str(v) for k, v in step_metadata.items()})

            if ball_reached:
                combined_metadata["ball_reached"] = "true"
                is_terminal = True

            if safety_violations:
                combined_metadata["has_safety_violation"] = "true"
                combined_metadata["safety_violations"] = ";".join(safety_violations)

            # Ensure first step carries the language instruction if not provided inline
            effective_language_instruction = language_instruction
            if effective_language_instruction is None and self.step_counter == 0:
                effective_language_instruction = self.episode_language_instruction

            # Create step data
            step = StepData(
                step_index=self.step_counter,
                timestamp_ns=timestamp_ns,
                frame_timestamp_ns=frame_timestamp_ns,
                video_frame_id=frame_id,
                depth_frame_id=frame_id,
                robot_state_timestamp_ns=robot_state_timestamp_ns,
                robot_state_frame_id=frame_id,
                action_timestamp_ns=time.time_ns(),
                rgb_image=rgb_image,
                depth_image=depth_image,
                robot_state=robot_state,
                action=action,
                action_source=action_source,
                is_terminal=is_terminal,
                language_instruction=effective_language_instruction,
                reward=1.0 if is_terminal else 0.0,  # Simple sparse reward
                audio_buffer=audio_buffer,
                audio_uri=audio_uri,
                audio_sample_rate_hz=audio_sample_rate_hz,
                audio_num_channels=audio_num_channels,
                audio_format=audio_format,
                audio_frame_id=audio_frame_id,
                audio_timestamp_ns=audio_timestamp_ns,
                audio_delta_ms=audio_delta_ms,
                step_metadata=combined_metadata,
            )

            self.steps.append(step)
            self.step_counter += 1
            
            if self.step_counter % 10 == 0:
                print(f"   Step {self.step_counter}/{self.max_steps}")
            
            return True
            
        except Exception as e:
            print(f"ERROR recording step: {e}")
            return False

    def _validate_step_alignment(self, step: StepData, tolerance_ns: int = 5_000_000) -> bool:
        """Validate timestamp/frame_id alignment within tolerance."""
        if step.video_frame_id != step.depth_frame_id:
            print(
                f"ERROR: Frame ID mismatch video={step.video_frame_id} depth={step.depth_frame_id}"
            )
            return False

        if step.robot_state_frame_id != step.video_frame_id:
            print(
                f"ERROR: Robot state frame_id {step.robot_state_frame_id}"
                f" does not match video frame_id {step.video_frame_id}"
            )
            return False

        skew = abs(step.frame_timestamp_ns - step.robot_state_timestamp_ns)
        if skew > tolerance_ns:
            skew_ms = skew / 1_000_000
            print(
                f"ERROR: Timestamp skew {skew_ms:.2f}ms exceeds tolerance between"
                " vision and robot state"
            )
            return False

        return True

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
            # Validate alignment before persisting
            for step in self.steps:
                if not self._validate_step_alignment(step):
                    print("❌ Episode failed schema validation; discarding recording")
                    self.current_episode = None
                    self.episode_metadata = None
                    self.steps = []
                    self.step_counter = 0
                    self.episode_language_instruction = None
                    return None

            # Update metadata
            self.episode_metadata.end_timestamp_ns = time.time_ns()
            self.episode_metadata.total_steps = len(self.steps)

            if not success:
                self.episode_metadata.tags.append("failed")

            # Ensure audio URIs are populated before serialization
            for i, step in enumerate(self.steps):
                if step.audio_buffer is not None and not step.audio_uri:
                    step.audio_uri = f"step_{i:04d}_audio.wav"

            # Prepare episode data
            episode_data = {
                "metadata": self.episode_metadata.to_dict(
                    language_instruction=self.episode_language_instruction
                ),
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

                if (
                    step.audio_buffer is not None
                    and step.audio_sample_rate_hz is not None
                    and step.audio_num_channels is not None
                ):
                    audio_filename = step.audio_uri or f"step_{i:04d}_audio.wav"
                    audio_path = episode_path / audio_filename
                    self._save_audio_file(
                        audio_path,
                        step.audio_buffer,
                        sample_rate_hz=step.audio_sample_rate_hz,
                        num_channels=step.audio_num_channels,
                    )
            # Run schema validation on persisted data (frame_id/timestamp alignment)
            from continuonbrain.recording.episode_upload import EpisodeUploadPipeline

            pipeline = EpisodeUploadPipeline()
            if not pipeline.validate_episode(episode_path):
                print("❌ Episode failed on-disk validation; removing recording")
                shutil.rmtree(episode_path, ignore_errors=True)
                self.current_episode = None
                self.episode_metadata = None
                self.steps = []
                self.step_counter = 0
                self.episode_language_instruction = None
                return None

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
            self.episode_language_instruction = None
            
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

        if self.microphone:
            self.microphone.stop()
            self.audio_enabled = False

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
