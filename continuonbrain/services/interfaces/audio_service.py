"""
Audio Service Interface

Protocol definition for audio services (microphone, speaker, TTS, STT).
"""
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class IAudioService(Protocol):
    """
    Protocol for audio services.

    Implementations handle:
    - Microphone input capture
    - Speaker output
    - Speech-to-text transcription
    - Text-to-speech synthesis
    - Audio feature extraction for model input
    """

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
        ...

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
        ...

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
        ...

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
        ...

    def listen_for_wake_word(
        self,
        wake_words: List[str] = ["hey robot", "continuon"],
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
        ...

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
            Feature vector suitable for model input
        """
        ...

    def get_audio_level(self) -> float:
        """
        Get current audio input level (0-1).

        Returns:
            RMS audio level from microphone
        """
        ...

    def is_speaking(self) -> bool:
        """Check if TTS is currently playing."""
        ...

    def stop_speaking(self) -> None:
        """Stop any ongoing TTS playback."""
        ...

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get audio capability flags.

        Returns:
            Dictionary of capability name -> available boolean:
                - has_microphone: bool
                - has_speaker: bool
                - has_tts: bool
                - has_stt: bool
                - has_wake_word: bool
        """
        ...

    def get_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get available audio devices.

        Returns:
            Dictionary with:
                - input_devices: list of microphone devices
                - output_devices: list of speaker devices
        """
        ...

    def is_available(self) -> bool:
        """Check if audio service is available."""
        ...
