"""
Audio Backend Interface

Base classes and data structures for audio backends.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class AudioBackendType(Enum):
    """Audio backend types."""
    PYAUDIO = "pyaudio"
    SOUNDDEVICE = "sounddevice"
    ALSA = "alsa"
    PULSE = "pulse"
    WHISPER = "whisper"
    VOSK = "vosk"
    PYTTSX3 = "pyttsx3"
    ESPEAK = "espeak"
    PIPER = "piper"


class AudioCapability(Enum):
    """Capabilities that audio backends can provide."""
    CAPTURE = "capture"
    PLAYBACK = "playback"
    TTS = "tts"
    STT = "stt"
    WAKE_WORD = "wake_word"
    FEATURE_EXTRACTION = "feature_extraction"


@dataclass
class AudioResult:
    """Result from an audio operation."""
    ok: bool
    audio: Optional[np.ndarray] = None
    text: Optional[str] = None
    confidence: float = 0.0
    duration_ms: float = 0.0
    backend: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ok": self.ok,
            "has_audio": self.audio is not None,
            "text": self.text,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "backend": self.backend,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AudioDevice:
    """Information about an audio device."""
    id: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False
    is_input: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "is_default": self.is_default,
            "is_input": self.is_input,
        }


class AudioBackend(ABC):
    """Abstract base class for audio backends."""

    @property
    @abstractmethod
    def backend_type(self) -> AudioBackendType:
        """Return the backend type."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[AudioCapability]:
        """Return supported capabilities."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and ready."""
        pass

    def capture(
        self,
        duration_ms: int = 1000,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> AudioResult:
        """
        Capture audio from microphone.

        Default implementation returns error - override in capture-capable backends.
        """
        return AudioResult(
            ok=False,
            backend=self.backend_type.value,
            error="Capture not supported by this backend",
        )

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        blocking: bool = True,
    ) -> AudioResult:
        """
        Play audio through speaker.

        Default implementation returns error - override in playback-capable backends.
        """
        return AudioResult(
            ok=False,
            backend=self.backend_type.value,
            error="Playback not supported by this backend",
        )

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> AudioResult:
        """
        Synthesize speech from text.

        Default implementation returns error - override in TTS backends.
        """
        return AudioResult(
            ok=False,
            backend=self.backend_type.value,
            error="TTS not supported by this backend",
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> AudioResult:
        """
        Transcribe audio to text.

        Default implementation returns error - override in STT backends.
        """
        return AudioResult(
            ok=False,
            backend=self.backend_type.value,
            error="STT not supported by this backend",
        )

    def extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[np.ndarray]:
        """
        Extract audio features for model input.

        Default implementation returns mel spectrogram features.
        """
        try:
            # Simple mel-like features without librosa
            # Normalize audio
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0

            # Frame parameters
            frame_length = int(0.025 * sample_rate)  # 25ms
            hop_length = int(0.010 * sample_rate)    # 10ms
            n_mels = 80

            # Simple energy-based features
            n_frames = (len(audio) - frame_length) // hop_length + 1
            features = np.zeros((n_frames, n_mels), dtype=np.float32)

            for i in range(n_frames):
                start = i * hop_length
                frame = audio[start:start + frame_length]

                # Simple FFT-based features
                fft = np.fft.rfft(frame * np.hanning(len(frame)))
                power = np.abs(fft) ** 2

                # Bin into mel-like bands (simplified)
                n_bins = len(power)
                mel_bins = np.linspace(0, n_bins, n_mels + 1, dtype=int)
                for j in range(n_mels):
                    features[i, j] = np.mean(power[mel_bins[j]:mel_bins[j+1]] + 1e-10)

            # Log scale
            features = np.log(features + 1e-10)

            return features

        except Exception:
            return None

    def get_devices(self) -> Dict[str, List[AudioDevice]]:
        """Get available audio devices."""
        return {"input_devices": [], "output_devices": []}

    def get_status(self) -> Dict[str, Any]:
        """Get backend status."""
        return {
            "type": self.backend_type.value,
            "available": self.is_available(),
            "capabilities": [c.value for c in self.capabilities],
        }

    def shutdown(self) -> None:
        """Release resources (optional)."""
        pass
