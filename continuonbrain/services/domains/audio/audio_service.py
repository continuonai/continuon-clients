"""
Audio Service Implementation

Domain service implementing IAudioService protocol.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from continuonbrain.services.audio import AudioManager
from continuonbrain.services.interfaces import IAudioService

logger = logging.getLogger(__name__)


class AudioService:
    """
    Audio service for microphone input, speaker output, and speech processing.

    Implements IAudioService protocol using AudioManager for backend orchestration.
    """

    def __init__(
        self,
        audio_manager: Optional[AudioManager] = None,
        whisper_model: str = "tiny",
    ):
        """
        Initialize audio service.

        Args:
            audio_manager: Optional AudioManager instance (creates one if not provided)
            whisper_model: Whisper model size for STT
        """
        self._manager = audio_manager or AudioManager(whisper_model=whisper_model)
        logger.info(f"AudioService initialized: {self._manager.get_capabilities()}")

    @property
    def manager(self) -> AudioManager:
        """Get the underlying AudioManager."""
        return self._manager

    def capture_audio(
        self,
        duration_ms: int = 1000,
        sample_rate: int = 16000,
    ) -> Optional[np.ndarray]:
        """
        Capture audio from microphone.

        Args:
            duration_ms: Duration to capture in milliseconds
            sample_rate: Sample rate in Hz

        Returns:
            Audio samples as float32 array, or None if capture failed
        """
        return self._manager.capture(
            duration_ms=duration_ms,
            sample_rate=sample_rate,
        )

    def play_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        blocking: bool = True,
    ) -> bool:
        """
        Play audio through speaker.

        Args:
            audio: Audio samples as float32 array
            sample_rate: Sample rate in Hz
            blocking: If True, wait for playback to complete

        Returns:
            True if playback started/completed successfully
        """
        return self._manager.play(
            audio=audio,
            sample_rate=sample_rate,
            blocking=blocking,
        )

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        blocking: bool = True,
    ) -> bool:
        """
        Convert text to speech and play.

        Args:
            text: Text to speak
            voice: Voice ID/name to use
            speed: Speech speed multiplier
            blocking: If True, wait for speech to complete

        Returns:
            True if speech was successful
        """
        return self._manager.speak(
            text=text,
            voice=voice,
            speed=speed,
            blocking=blocking,
        )

    def transcribe(
        self,
        audio: Optional[np.ndarray] = None,
        duration_ms: int = 5000,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples. If None, captures from microphone.
            duration_ms: Duration to capture if audio is None

        Returns:
            Dictionary containing:
                - text: str - Transcribed text
                - confidence: float - Transcription confidence
                - language: str - Detected language
                - segments: list - Word/phrase segments with timestamps
        """
        return self._manager.transcribe(
            audio=audio,
            duration_ms=duration_ms,
        )

    def listen_for_wake_word(
        self,
        wake_words: List[str] = None,
        timeout_ms: int = 10000,
    ) -> Optional[str]:
        """
        Listen for wake word activation.

        Args:
            wake_words: List of wake words to detect
            timeout_ms: Timeout in milliseconds

        Returns:
            Detected wake word, or None if timeout
        """
        if wake_words is None:
            wake_words = ["hey robot", "continuon"]
        return self._manager.listen_for_wake_word(
            wake_words=wake_words,
            timeout_ms=timeout_ms,
        )

    def extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Extract audio features for model input.

        Args:
            audio: Raw audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Feature vector suitable for model input (mel spectrogram)
        """
        return self._manager.extract_features(
            audio=audio,
            sample_rate=sample_rate,
        )

    def get_audio_level(self) -> float:
        """
        Get current audio input level (0-1).

        Returns:
            RMS audio level from microphone
        """
        return self._manager.get_audio_level()

    def is_speaking(self) -> bool:
        """Check if TTS is currently playing."""
        return self._manager.is_speaking()

    def stop_speaking(self) -> None:
        """Stop any ongoing TTS playback."""
        self._manager.stop_speaking()

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get audio capability flags.

        Returns:
            Dictionary of capability name -> available boolean
        """
        return self._manager.get_capabilities()

    def get_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get available audio devices.

        Returns:
            Dictionary with input_devices and output_devices
        """
        return self._manager.get_devices()

    def is_available(self) -> bool:
        """Check if audio service is available."""
        return self._manager.is_available()

    def get_status(self) -> Dict[str, Any]:
        """Get audio service status."""
        return {
            "available": self.is_available(),
            **self._manager.get_status(),
        }

    def shutdown(self) -> None:
        """Release resources."""
        self._manager.shutdown()
