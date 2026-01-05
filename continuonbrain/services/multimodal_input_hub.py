"""
Multi-Modal Input Hub for Agent Harness Architecture

This module provides unified input processing for all sensory modalities:
- Vision: OAK-D RGB+Depth, SAM3 segmentation, Hailo pose estimation
- Audio: Speech-to-text (Whisper), voice activity detection
- Text: Chat interface, command parsing

The hub routes all inputs to the World Model Integration for fusion,
then injects the coherent world state into the HOPE Agent.

Architecture:
    Sensors (OAK-D, Microphone, Chat) → MultiModalInputHub → WorldModelIntegration
                                                            ↓
                                            HOPE Brain CMS + Agent
"""

import logging
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InputEvent:
    """A timestamped input event from any modality."""
    timestamp: float
    modality: str  # "vision", "audio", "text"
    event_type: str  # "frame", "speech", "command", "query"
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority = process first
    source: str = ""  # e.g., "oak_d", "microphone", "chat_ui"


@dataclass
class ProcessedInput:
    """Processed input ready for world model integration."""
    timestamp: float
    modality: str
    content: Any
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisionInputProcessor:
    """Processes vision inputs from cameras and vision services."""

    def __init__(self, oak_camera=None, sam_service=None, pose_service=None):
        self.oak_camera = oak_camera
        self.sam_service = sam_service
        self.pose_service = pose_service
        self._last_frame = None
        self._last_depth = None

    def capture_frame(self) -> Optional[Dict[str, Any]]:
        """Capture a frame from OAK-D camera."""
        if not self.oak_camera:
            return None

        try:
            rgb, depth = self.oak_camera.get_frame()
            if rgb is not None:
                self._last_frame = rgb
                self._last_depth = depth
                return {
                    "rgb": rgb,
                    "depth": depth,
                    "timestamp": time.time(),
                }
        except Exception as e:
            logger.debug(f"Vision capture error: {e}")
        return None

    def process_segmentation(self, rgb_frame) -> Optional[Dict[str, Any]]:
        """Run SAM3 segmentation on an RGB frame."""
        if not self.sam_service or rgb_frame is None:
            return None

        try:
            result = self.sam_service.segment(rgb_frame)
            return result
        except Exception as e:
            logger.debug(f"Segmentation error: {e}")
        return None

    def process_pose(self, rgb_frame) -> Optional[Dict[str, Any]]:
        """Run pose estimation on an RGB frame."""
        if not self.pose_service or rgb_frame is None:
            return None

        try:
            result = self.pose_service.estimate_pose(rgb_frame)
            return result
        except Exception as e:
            logger.debug(f"Pose estimation error: {e}")
        return None


class AudioInputProcessor:
    """Processes audio inputs for speech recognition."""

    def __init__(self, whisper_model=None):
        self.whisper_model = whisper_model
        self._audio_buffer = []
        self._sample_rate = 16000

    def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio to text using Whisper."""
        if not self.whisper_model or audio_data is None:
            return None

        try:
            # Whisper expects 16kHz float32 audio
            result = self.whisper_model.transcribe(audio_data)
            return result.get("text", "").strip()
        except Exception as e:
            logger.debug(f"Transcription error: {e}")
        return None

    def detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Simple voice activity detection based on energy."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return False

        energy = np.sqrt(np.mean(audio_chunk.astype(float) ** 2))
        threshold = 0.01  # Adjust based on environment
        return energy > threshold


class TextInputProcessor:
    """Processes text inputs from chat interface."""

    def __init__(self):
        self._command_patterns = {
            "move": ["move", "go", "drive", "navigate"],
            "pick": ["pick", "grab", "grasp", "lift"],
            "place": ["place", "put", "drop", "set"],
            "look": ["look", "see", "show", "view"],
            "stop": ["stop", "halt", "pause", "freeze"],
            "help": ["help", "what", "how", "explain"],
        }

    def parse_command(self, text: str) -> Dict[str, Any]:
        """Parse text to extract command intent."""
        text_lower = text.lower().strip()

        for command, keywords in self._command_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        "type": "command",
                        "command": command,
                        "raw_text": text,
                        "confidence": 0.8,
                    }

        # If no command detected, treat as query
        return {
            "type": "query",
            "query": text,
            "confidence": 1.0,
        }

    def is_question(self, text: str) -> bool:
        """Check if text is a question."""
        text = text.strip()
        return (
            text.endswith("?") or
            text.lower().startswith(("what", "where", "when", "why", "how", "is", "are", "can", "do"))
        )


class MultiModalInputHub:
    """
    Unified hub for processing all sensory inputs.

    Collects inputs from vision, audio, and text modalities,
    processes them, and routes to the world model for integration.
    """

    def __init__(
        self,
        world_model_integration=None,
        brain_service=None,
        max_queue_size: int = 100,
    ):
        """
        Initialize the multi-modal input hub.

        Args:
            world_model_integration: WorldModelIntegration for sensory fusion
            brain_service: BrainService for HOPE agent access
            max_queue_size: Maximum pending inputs before dropping
        """
        self.world_model = world_model_integration
        self.brain_service = brain_service

        # Input queue for async processing
        self._input_queue = queue.PriorityQueue(maxsize=max_queue_size)

        # Modality processors
        self._vision_processor = VisionInputProcessor()
        self._audio_processor = AudioInputProcessor()
        self._text_processor = TextInputProcessor()

        # Processing thread
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Callbacks for processed inputs
        self._callbacks: Dict[str, List[Callable]] = {
            "vision": [],
            "audio": [],
            "text": [],
            "all": [],
        }

        # Stats
        self._stats = {
            "vision_frames": 0,
            "audio_chunks": 0,
            "text_messages": 0,
            "total_processed": 0,
        }

        logger.info("MultiModalInputHub initialized")

    def configure_vision(self, oak_camera=None, sam_service=None, pose_service=None):
        """Configure vision processing components."""
        self._vision_processor.oak_camera = oak_camera
        self._vision_processor.sam_service = sam_service
        self._vision_processor.pose_service = pose_service
        logger.info("Vision processors configured")

    def configure_audio(self, whisper_model=None):
        """Configure audio processing components."""
        self._audio_processor.whisper_model = whisper_model
        logger.info("Audio processors configured")

    def register_callback(self, modality: str, callback: Callable):
        """
        Register a callback for processed inputs.

        Args:
            modality: "vision", "audio", "text", or "all"
            callback: Function to call with ProcessedInput
        """
        if modality in self._callbacks:
            self._callbacks[modality].append(callback)

    def start(self):
        """Start the input processing loop."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="multimodal_input_hub"
        )
        self._processing_thread.start()
        logger.info("MultiModalInputHub processing started")

    def stop(self):
        """Stop the input processing loop."""
        self._stop_event.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        self._running = False
        logger.info("MultiModalInputHub processing stopped")

    def submit_vision_frame(self, rgb: np.ndarray, depth: np.ndarray = None, source: str = "oak_d"):
        """Submit a vision frame for processing."""
        event = InputEvent(
            timestamp=time.time(),
            modality="vision",
            event_type="frame",
            data={"rgb": rgb, "depth": depth},
            source=source,
            priority=1,  # Vision is medium priority
        )
        self._enqueue(event)
        self._stats["vision_frames"] += 1

    def submit_audio_chunk(self, audio: np.ndarray, source: str = "microphone"):
        """Submit an audio chunk for processing."""
        event = InputEvent(
            timestamp=time.time(),
            modality="audio",
            event_type="speech",
            data={"audio": audio},
            source=source,
            priority=0,  # Audio is lower priority
        )
        self._enqueue(event)
        self._stats["audio_chunks"] += 1

    def submit_text(self, text: str, source: str = "chat_ui") -> ProcessedInput:
        """
        Submit text input for immediate processing.

        Text is processed synchronously since it's typically interactive.
        Returns the processed input immediately.
        """
        self._stats["text_messages"] += 1

        parsed = self._text_processor.parse_command(text)
        processed = ProcessedInput(
            timestamp=time.time(),
            modality="text",
            content=parsed,
            confidence=parsed.get("confidence", 1.0),
            metadata={"source": source, "is_question": self._text_processor.is_question(text)},
        )

        # Trigger callbacks
        self._trigger_callbacks("text", processed)
        self._trigger_callbacks("all", processed)

        # Inject into world model if available
        if self.world_model:
            self._inject_text_context(processed)

        self._stats["total_processed"] += 1
        return processed

    def _enqueue(self, event: InputEvent):
        """Add an event to the processing queue."""
        try:
            # Use negative priority for PriorityQueue (higher = first)
            self._input_queue.put_nowait((-event.priority, event.timestamp, event))
        except queue.Full:
            logger.warning(f"Input queue full, dropping {event.modality} event")

    def _processing_loop(self):
        """Background processing loop for queued inputs."""
        while not self._stop_event.is_set():
            try:
                # Wait for input with timeout
                try:
                    _, _, event = self._input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Process based on modality
                if event.modality == "vision":
                    self._process_vision(event)
                elif event.modality == "audio":
                    self._process_audio(event)

                self._stats["total_processed"] += 1

            except Exception as e:
                logger.error(f"Processing error: {e}")

    def _process_vision(self, event: InputEvent):
        """Process a vision event."""
        rgb = event.data.get("rgb")
        depth = event.data.get("depth")

        if rgb is None:
            return

        # Run segmentation if available
        seg_result = None
        if self._vision_processor.sam_service:
            seg_result = self._vision_processor.process_segmentation(rgb)

        # Run pose estimation if available
        pose_result = None
        if self._vision_processor.pose_service:
            pose_result = self._vision_processor.process_pose(rgb)

        processed = ProcessedInput(
            timestamp=event.timestamp,
            modality="vision",
            content={
                "segmentation": seg_result,
                "pose": pose_result,
                "has_depth": depth is not None,
            },
            confidence=0.9 if seg_result else 0.5,
            metadata={"source": event.source},
        )

        # Update world model
        if self.world_model and seg_result:
            # WorldModelIntegration expects specific format
            self._inject_vision_context(processed, seg_result, pose_result)

        # Update brain_service caches for backward compatibility
        if self.brain_service:
            if seg_result:
                self.brain_service.last_segmentation = seg_result
            if pose_result:
                self.brain_service.last_pose_result = pose_result

        # Trigger callbacks
        self._trigger_callbacks("vision", processed)
        self._trigger_callbacks("all", processed)

    def _process_audio(self, event: InputEvent):
        """Process an audio event."""
        audio = event.data.get("audio")
        if audio is None:
            return

        # Check for voice activity
        if not self._audio_processor.detect_voice_activity(audio):
            return

        # Transcribe speech
        text = self._audio_processor.transcribe(audio)
        if not text:
            return

        # Process as text
        parsed = self._text_processor.parse_command(text)
        processed = ProcessedInput(
            timestamp=event.timestamp,
            modality="audio",
            content={
                "text": text,
                "parsed": parsed,
            },
            confidence=0.7,  # Lower confidence for speech
            metadata={"source": event.source},
        )

        # Inject into world model
        if self.world_model:
            self._inject_text_context(processed)

        # Trigger callbacks
        self._trigger_callbacks("audio", processed)
        self._trigger_callbacks("all", processed)

    def _inject_vision_context(self, processed: ProcessedInput, seg_result: Dict, pose_result: Dict = None):
        """Inject vision context into world model."""
        try:
            from continuonbrain.services.world_model_integration import SensoryFrame

            frame = SensoryFrame(
                timestamp=processed.timestamp,
                segmentation=seg_result,
                objects=seg_result.get("objects", []) if seg_result else [],
                pose_estimation=pose_result,
            )

            # Let world model integration handle the frame
            if hasattr(self.world_model, '_integrate_frame'):
                self.world_model._integrate_frame(frame)
        except Exception as e:
            logger.debug(f"Vision context injection error: {e}")

    def _inject_text_context(self, processed: ProcessedInput):
        """Inject text context into HOPE agent."""
        try:
            if self.brain_service and self.brain_service.hope_agent:
                # Text queries can provide context for HOPE
                content = processed.content
                if content.get("type") == "query":
                    # Store as potential learning context
                    pass
        except Exception as e:
            logger.debug(f"Text context injection error: {e}")

    def _trigger_callbacks(self, modality: str, processed: ProcessedInput):
        """Trigger registered callbacks for a modality."""
        for callback in self._callbacks.get(modality, []):
            try:
                callback(processed)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self._stats,
            "queue_size": self._input_queue.qsize(),
            "running": self._running,
        }

    def capture_vision_snapshot(self) -> Optional[Dict[str, Any]]:
        """Capture and process a single vision snapshot synchronously."""
        frame_data = self._vision_processor.capture_frame()
        if not frame_data:
            return None

        rgb = frame_data.get("rgb")

        result = {
            "timestamp": frame_data["timestamp"],
            "has_rgb": rgb is not None,
            "has_depth": frame_data.get("depth") is not None,
        }

        if rgb is not None:
            seg_result = self._vision_processor.process_segmentation(rgb)
            pose_result = self._vision_processor.process_pose(rgb)

            result["segmentation"] = seg_result
            result["pose"] = pose_result

            # Update caches
            if self.brain_service:
                if seg_result:
                    self.brain_service.last_segmentation = seg_result
                if pose_result:
                    self.brain_service.last_pose_result = pose_result

        return result


def create_multimodal_hub(brain_service=None) -> MultiModalInputHub:
    """
    Factory function to create a configured multi-modal input hub.

    Args:
        brain_service: BrainService instance

    Returns:
        Configured MultiModalInputHub
    """
    # Get or create world model integration
    world_model = None
    if brain_service:
        if hasattr(brain_service, '_world_model_integration') and brain_service._world_model_integration:
            world_model = brain_service._world_model_integration
        else:
            from continuonbrain.services.world_model_integration import create_world_model_integration
            world_model, teacher = create_world_model_integration(brain_service)
            brain_service._world_model_integration = world_model
            brain_service._teacher_interface = teacher

    hub = MultiModalInputHub(
        world_model_integration=world_model,
        brain_service=brain_service,
    )

    # Configure vision if cameras available
    if brain_service:
        hub.configure_vision(
            oak_camera=getattr(brain_service, 'camera', None),
            sam_service=getattr(brain_service, 'sam_service', None),
            pose_service=getattr(brain_service, 'pose_service', None),
        )

    return hub
