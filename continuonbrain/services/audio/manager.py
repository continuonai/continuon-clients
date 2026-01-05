"""
Audio Manager

Unified orchestrator for all audio operations.
"""
import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from .backend import (
    AudioBackend,
    AudioBackendType,
    AudioCapability,
    AudioDevice,
    AudioResult,
)
from .microphone_backend import PyAudioBackend, SoundDeviceBackend
from .tts_backend import PiperBackend, ESpeakBackend, Pyttsx3Backend
from .stt_backend import WhisperBackend, VoskBackend

logger = logging.getLogger(__name__)


class AudioManager:
    """
    Unified audio manager with automatic backend selection.

    Orchestrates:
    - Audio capture (microphone)
    - Audio playback (speaker)
    - Text-to-speech synthesis
    - Speech-to-text transcription
    - Audio feature extraction for model input
    """

    def __init__(
        self,
        preferred_capture: Optional[AudioBackendType] = None,
        preferred_tts: Optional[AudioBackendType] = None,
        preferred_stt: Optional[AudioBackendType] = None,
        whisper_model: str = "tiny",
        piper_model: Optional[str] = None,
    ):
        self._lock = threading.Lock()
        self._is_speaking = False
        self._speak_thread: Optional[threading.Thread] = None

        # Initialize backends
        self._capture_backends: List[AudioBackend] = []
        self._tts_backends: List[AudioBackend] = []
        self._stt_backends: List[AudioBackend] = []

        # Store preferences
        self._preferred_capture = preferred_capture
        self._preferred_tts = preferred_tts
        self._preferred_stt = preferred_stt

        # Initialize capture backends (try sounddevice first, then pyaudio)
        self._capture_backends = [
            SoundDeviceBackend(),
            PyAudioBackend(),
        ]

        # Initialize TTS backends (try Piper first, then eSpeak, then pyttsx3)
        self._tts_backends = [
            PiperBackend(model_path=piper_model),
            ESpeakBackend(),
            Pyttsx3Backend(),
        ]

        # Initialize STT backends (try Whisper first, then Vosk)
        self._stt_backends = [
            WhisperBackend(model_size=whisper_model),
            VoskBackend(),
        ]

        # Reorder based on preferences
        self._apply_preferences()

        logger.info(
            f"AudioManager initialized: "
            f"capture={self.capture_backend}, "
            f"tts={self.tts_backend}, "
            f"stt={self.stt_backend}"
        )

    def _apply_preferences(self) -> None:
        """Reorder backends based on preferences."""
        if self._preferred_capture:
            self._capture_backends.sort(
                key=lambda b: 0 if b.backend_type == self._preferred_capture else 1
            )

        if self._preferred_tts:
            self._tts_backends.sort(
                key=lambda b: 0 if b.backend_type == self._preferred_tts else 1
            )

        if self._preferred_stt:
            self._stt_backends.sort(
                key=lambda b: 0 if b.backend_type == self._preferred_stt else 1
            )

    @property
    def capture_backend(self) -> Optional[str]:
        """Get the active capture backend name."""
        for backend in self._capture_backends:
            if backend.is_available():
                return backend.backend_type.value
        return None

    @property
    def tts_backend(self) -> Optional[str]:
        """Get the active TTS backend name."""
        for backend in self._tts_backends:
            if backend.is_available():
                return backend.backend_type.value
        return None

    @property
    def stt_backend(self) -> Optional[str]:
        """Get the active STT backend name."""
        for backend in self._stt_backends:
            if backend.is_available():
                return backend.backend_type.value
        return None

    def _get_capture_backend(self) -> Optional[AudioBackend]:
        """Get an available capture backend."""
        for backend in self._capture_backends:
            if backend.is_available() and AudioCapability.CAPTURE in backend.capabilities:
                return backend
        return None

    def _get_playback_backend(self) -> Optional[AudioBackend]:
        """Get an available playback backend."""
        for backend in self._capture_backends:
            if backend.is_available() and AudioCapability.PLAYBACK in backend.capabilities:
                return backend
        return None

    def _get_tts_backend(self) -> Optional[AudioBackend]:
        """Get an available TTS backend."""
        for backend in self._tts_backends:
            if backend.is_available():
                return backend
        return None

    def _get_stt_backend(self) -> Optional[AudioBackend]:
        """Get an available STT backend."""
        for backend in self._stt_backends:
            if backend.is_available():
                return backend
        return None

    def capture(
        self,
        duration_ms: int = 1000,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> Optional[np.ndarray]:
        """
        Capture audio from microphone.

        Args:
            duration_ms: Duration to capture in milliseconds
            sample_rate: Sample rate in Hz
            channels: Number of audio channels

        Returns:
            Audio samples as float32 array, or None if capture failed
        """
        backend = self._get_capture_backend()
        if backend is None:
            logger.warning("No capture backend available")
            return None

        result = backend.capture(
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            channels=channels,
        )

        if result.ok:
            return result.audio
        else:
            logger.error(f"Capture failed: {result.error}")
            return None

    def play(
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
        backend = self._get_playback_backend()
        if backend is None:
            logger.warning("No playback backend available")
            return False

        result = backend.play(
            audio=audio,
            sample_rate=sample_rate,
            blocking=blocking,
        )

        return result.ok

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        blocking: bool = True,
    ) -> bool:
        """
        Synthesize text to speech and play.

        Args:
            text: Text to speak
            voice: Voice ID/name to use
            speed: Speech speed multiplier
            blocking: If True, wait for speech to complete

        Returns:
            True if speech was successful
        """
        if not text:
            return True

        tts_backend = self._get_tts_backend()
        playback_backend = self._get_playback_backend()

        if tts_backend is None:
            logger.warning("No TTS backend available")
            return False

        if playback_backend is None:
            logger.warning("No playback backend available")
            return False

        def _do_speak():
            with self._lock:
                self._is_speaking = True

            try:
                # Synthesize
                tts_result = tts_backend.synthesize(
                    text=text,
                    voice=voice,
                    speed=speed,
                )

                if not tts_result.ok:
                    logger.error(f"TTS failed: {tts_result.error}")
                    return False

                # Play
                sample_rate = tts_result.metadata.get("sample_rate", 16000)
                play_result = playback_backend.play(
                    audio=tts_result.audio,
                    sample_rate=sample_rate,
                    blocking=True,
                )

                return play_result.ok

            finally:
                with self._lock:
                    self._is_speaking = False

        if blocking:
            return _do_speak()
        else:
            self._speak_thread = threading.Thread(target=_do_speak)
            self._speak_thread.start()
            return True

    def transcribe(
        self,
        audio: Optional[np.ndarray] = None,
        duration_ms: int = 5000,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples. If None, captures from microphone.
            duration_ms: Duration to capture if audio is None
            language: Language hint for transcription

        Returns:
            Dictionary with text, confidence, language, segments
        """
        # Capture if no audio provided
        if audio is None:
            audio = self.capture(duration_ms=duration_ms)
            if audio is None:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "error": "Failed to capture audio",
                }

        stt_backend = self._get_stt_backend()
        if stt_backend is None:
            logger.warning("No STT backend available")
            return {
                "text": "",
                "confidence": 0.0,
                "error": "No STT backend available",
            }

        result = stt_backend.transcribe(
            audio=audio,
            sample_rate=16000,
            language=language,
        )

        if result.ok:
            return {
                "text": result.text or "",
                "confidence": result.confidence,
                "language": result.metadata.get("language"),
                "segments": result.metadata.get("segments", []),
                "duration_ms": result.duration_ms,
                "backend": result.backend,
            }
        else:
            return {
                "text": "",
                "confidence": 0.0,
                "error": result.error,
            }

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
            wake_words = ["hey robot", "continuon", "hey brain"]

        import time

        start_time = time.time()
        chunk_ms = 2000  # Listen in 2-second chunks

        while (time.time() - start_time) * 1000 < timeout_ms:
            result = self.transcribe(duration_ms=chunk_ms)
            text = result.get("text", "").lower()

            for wake_word in wake_words:
                if wake_word.lower() in text:
                    logger.info(f"Wake word detected: {wake_word}")
                    return wake_word

        return None

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
            Feature array suitable for model input
        """
        # Try STT backends first (they may have better feature extraction)
        stt_backend = self._get_stt_backend()
        if stt_backend is not None:
            features = stt_backend.extract_features(audio, sample_rate)
            if features is not None:
                return features

        # Fall back to basic feature extraction
        from .backend import AudioBackend
        base = AudioBackend.__new__(AudioBackend)
        features = base.extract_features(audio, sample_rate)

        if features is not None:
            return features

        # Return zeros if all else fails
        n_frames = max(1, len(audio) // (sample_rate // 100))
        return np.zeros((n_frames, 80), dtype=np.float32)

    def get_audio_level(self, duration_ms: int = 100) -> float:
        """
        Get current audio input level (0-1).

        Args:
            duration_ms: Duration to sample

        Returns:
            RMS audio level from microphone
        """
        audio = self.capture(duration_ms=duration_ms)
        if audio is None or len(audio) == 0:
            return 0.0

        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))
        return float(min(1.0, rms))

    def is_speaking(self) -> bool:
        """Check if TTS is currently playing."""
        with self._lock:
            return self._is_speaking

    def stop_speaking(self) -> None:
        """Stop any ongoing TTS playback."""
        with self._lock:
            self._is_speaking = False
        # Note: Actually stopping playback would require backend support

    def get_capabilities(self) -> Dict[str, bool]:
        """Get audio capability flags."""
        return {
            "has_microphone": self._get_capture_backend() is not None,
            "has_speaker": self._get_playback_backend() is not None,
            "has_tts": self._get_tts_backend() is not None,
            "has_stt": self._get_stt_backend() is not None,
            "has_wake_word": self._get_stt_backend() is not None,
        }

    def get_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available audio devices."""
        result = {"input_devices": [], "output_devices": []}

        for backend in self._capture_backends:
            if backend.is_available():
                devices = backend.get_devices()
                for dev in devices.get("input_devices", []):
                    result["input_devices"].append(dev.to_dict())
                for dev in devices.get("output_devices", []):
                    result["output_devices"].append(dev.to_dict())
                break  # Use first available backend

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get audio system status."""
        return {
            "capture_backend": self.capture_backend,
            "tts_backend": self.tts_backend,
            "stt_backend": self.stt_backend,
            "capabilities": self.get_capabilities(),
            "is_speaking": self.is_speaking(),
            "backends": {
                "capture": [
                    {
                        "type": b.backend_type.value,
                        "available": b.is_available(),
                    }
                    for b in self._capture_backends
                ],
                "tts": [
                    {
                        "type": b.backend_type.value,
                        "available": b.is_available(),
                    }
                    for b in self._tts_backends
                ],
                "stt": [
                    {
                        "type": b.backend_type.value,
                        "available": b.is_available(),
                    }
                    for b in self._stt_backends
                ],
            },
        }

    def is_available(self) -> bool:
        """Check if any audio capability is available."""
        caps = self.get_capabilities()
        return any(caps.values())

    def shutdown(self) -> None:
        """Release all resources."""
        for backend in self._capture_backends + self._tts_backends + self._stt_backends:
            try:
                backend.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down {backend.backend_type}: {e}")
