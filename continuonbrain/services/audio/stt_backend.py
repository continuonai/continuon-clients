"""
STT Backend

Speech-to-text backends for audio transcription.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .backend import (
    AudioBackend,
    AudioBackendType,
    AudioCapability,
    AudioResult,
)

logger = logging.getLogger(__name__)


class WhisperBackend(AudioBackend):
    """STT backend using OpenAI Whisper (local or API)."""

    def __init__(
        self,
        model_size: str = "tiny",
        use_api: bool = False,
        api_key: Optional[str] = None,
    ):
        self._model_size = model_size
        self._use_api = use_api
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = None
        self._available: Optional[bool] = None

    @property
    def backend_type(self) -> AudioBackendType:
        return AudioBackendType.WHISPER

    @property
    def capabilities(self) -> List[AudioCapability]:
        return [AudioCapability.STT, AudioCapability.FEATURE_EXTRACTION]

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        if self._use_api:
            self._available = bool(self._api_key)
        else:
            try:
                import whisper
                self._available = True
            except ImportError:
                # Try faster-whisper
                try:
                    from faster_whisper import WhisperModel
                    self._available = True
                except ImportError:
                    self._available = False

        return self._available

    def _load_model(self):
        """Lazy load the whisper model."""
        if self._model is not None:
            return self._model

        if self._use_api:
            return None

        try:
            # Try faster-whisper first (better performance on CPU)
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info(f"Loaded faster-whisper model: {self._model_size}")
        except ImportError:
            # Fall back to regular whisper
            import whisper
            self._model = whisper.load_model(self._model_size)
            logger.info(f"Loaded whisper model: {self._model_size}")

        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> AudioResult:
        if not self.is_available():
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error="Whisper not available",
            )

        try:
            import time

            start_time = time.time()

            # Ensure audio is float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0

            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                audio = self._resample(audio, sample_rate, 16000)

            if self._use_api:
                result = self._transcribe_api(audio, language)
            else:
                result = self._transcribe_local(audio, language)

            elapsed_ms = (time.time() - start_time) * 1000
            result.duration_ms = elapsed_ms

            return result

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    def _transcribe_local(
        self,
        audio: np.ndarray,
        language: Optional[str],
    ) -> AudioResult:
        """Transcribe using local Whisper model."""
        model = self._load_model()

        try:
            # faster-whisper
            from faster_whisper import WhisperModel
            if isinstance(model, WhisperModel):
                segments, info = model.transcribe(
                    audio,
                    language=language,
                    beam_size=5,
                )
                segments_list = list(segments)
                text = " ".join(seg.text for seg in segments_list)

                return AudioResult(
                    ok=True,
                    text=text.strip(),
                    confidence=0.9,  # faster-whisper doesn't provide confidence
                    backend=self.backend_type.value,
                    metadata={
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "segments": [
                            {
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text,
                            }
                            for seg in segments_list
                        ],
                    },
                )
        except:
            pass

        # Regular whisper
        result = model.transcribe(
            audio,
            language=language,
            fp16=False,
        )

        return AudioResult(
            ok=True,
            text=result["text"].strip(),
            confidence=0.9,
            backend=self.backend_type.value,
            metadata={
                "language": result.get("language"),
                "segments": result.get("segments", []),
            },
        )

    def _transcribe_api(
        self,
        audio: np.ndarray,
        language: Optional[str],
    ) -> AudioResult:
        """Transcribe using OpenAI Whisper API."""
        import tempfile
        import wave
        import requests

        # Save audio to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        # Convert to int16 for WAV
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        try:
            with open(temp_path, "rb") as f:
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={
                        "model": "whisper-1",
                        "language": language or "",
                        "response_format": "verbose_json",
                    },
                    timeout=30,
                )

            os.unlink(temp_path)

            if response.status_code != 200:
                return AudioResult(
                    ok=False,
                    backend=self.backend_type.value,
                    error=f"API error: {response.text}",
                )

            data = response.json()

            return AudioResult(
                ok=True,
                text=data.get("text", "").strip(),
                confidence=0.95,
                backend=self.backend_type.value,
                metadata={
                    "language": data.get("language"),
                    "duration": data.get("duration"),
                    "segments": data.get("segments", []),
                },
            )

        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio

        # Simple linear interpolation resampling
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[np.ndarray]:
        """Extract Whisper encoder features for multimodal input."""
        if not self.is_available() or self._use_api:
            return super().extract_features(audio, sample_rate)

        try:
            model = self._load_model()

            # Resample if needed
            if sample_rate != 16000:
                audio = self._resample(audio, sample_rate, 16000)

            # Pad or trim to 30 seconds
            import whisper
            audio = whisper.pad_or_trim(audio)

            # Get mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).numpy()

            return mel

        except Exception as e:
            logger.warning(f"Whisper feature extraction failed: {e}")
            return super().extract_features(audio, sample_rate)


class VoskBackend(AudioBackend):
    """STT backend using Vosk (offline, lightweight)."""

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._model = None
        self._available: Optional[bool] = None

    @property
    def backend_type(self) -> AudioBackendType:
        return AudioBackendType.VOSK

    @property
    def capabilities(self) -> List[AudioCapability]:
        return [AudioCapability.STT]

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            from vosk import Model, KaldiRecognizer

            # Check if model exists
            model_path = self._model_path or self._find_model()
            self._available = model_path is not None and os.path.isdir(model_path)

        except ImportError:
            self._available = False

        return self._available

    def _find_model(self) -> Optional[str]:
        """Find a Vosk model in common locations."""
        search_paths = [
            "/usr/share/vosk",
            "/opt/vosk",
            os.path.expanduser("~/.local/share/vosk"),
            os.path.expanduser("~/vosk-model"),
        ]

        for base_path in search_paths:
            if os.path.isdir(base_path):
                # Look for any model directory
                for item in os.listdir(base_path):
                    model_dir = os.path.join(base_path, item)
                    if os.path.isdir(model_dir) and os.path.exists(
                        os.path.join(model_dir, "am", "final.mdl")
                    ):
                        return model_dir

        return None

    def _load_model(self):
        """Lazy load the Vosk model."""
        if self._model is not None:
            return self._model

        from vosk import Model

        model_path = self._model_path or self._find_model()
        if not model_path:
            raise RuntimeError("No Vosk model found")

        self._model = Model(model_path)
        logger.info(f"Loaded Vosk model from: {model_path}")

        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> AudioResult:
        if not self.is_available():
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error="Vosk not available",
            )

        try:
            import json
            import time
            from vosk import KaldiRecognizer

            start_time = time.time()

            model = self._load_model()
            recognizer = KaldiRecognizer(model, sample_rate)
            recognizer.SetWords(True)

            # Convert to int16
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio.astype(np.int16)

            # Process audio
            recognizer.AcceptWaveform(audio_int16.tobytes())
            result = json.loads(recognizer.FinalResult())

            elapsed_ms = (time.time() - start_time) * 1000

            text = result.get("text", "")
            words = result.get("result", [])

            # Calculate average confidence
            if words:
                confidence = sum(w.get("conf", 0) for w in words) / len(words)
            else:
                confidence = 0.0

            return AudioResult(
                ok=True,
                text=text,
                confidence=confidence,
                duration_ms=elapsed_ms,
                backend=self.backend_type.value,
                metadata={
                    "words": words,
                },
            )

        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )
