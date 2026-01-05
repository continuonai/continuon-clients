"""
TTS Backend

Text-to-speech backends for audio synthesis.
"""
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

from .backend import (
    AudioBackend,
    AudioBackendType,
    AudioCapability,
    AudioResult,
)

logger = logging.getLogger(__name__)


class PiperBackend(AudioBackend):
    """TTS backend using Piper (fast local neural TTS)."""

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._available: Optional[bool] = None
        self._piper_path: Optional[str] = None

    @property
    def backend_type(self) -> AudioBackendType:
        return AudioBackendType.PIPER

    @property
    def capabilities(self) -> List[AudioCapability]:
        return [AudioCapability.TTS]

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        # Check for piper executable
        try:
            result = subprocess.run(
                ["piper", "--help"],
                capture_output=True,
                timeout=5,
            )
            self._piper_path = "piper"
            self._available = True
        except (subprocess.SubprocessError, FileNotFoundError):
            # Try common paths
            common_paths = [
                "/usr/bin/piper",
                "/usr/local/bin/piper",
                os.path.expanduser("~/.local/bin/piper"),
            ]
            for path in common_paths:
                if os.path.isfile(path):
                    self._piper_path = path
                    self._available = True
                    break
            else:
                self._available = False

        return self._available

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> AudioResult:
        if not self.is_available():
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error="Piper TTS not available",
            )

        try:
            import time

            # Use provided model or default
            model_path = voice or self._model_path
            if not model_path:
                # Try to find a default model
                model_dirs = [
                    "/usr/share/piper/voices",
                    os.path.expanduser("~/.local/share/piper/voices"),
                    "/opt/piper/voices",
                ]
                for model_dir in model_dirs:
                    if os.path.isdir(model_dir):
                        models = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]
                        if models:
                            model_path = os.path.join(model_dir, models[0])
                            break

            if not model_path or not os.path.exists(model_path):
                return AudioResult(
                    ok=False,
                    backend=self.backend_type.value,
                    error="No Piper voice model found",
                )

            start_time = time.time()

            # Generate audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            cmd = [
                self._piper_path,
                "--model", model_path,
                "--output_file", temp_path,
            ]

            if speed != 1.0:
                cmd.extend(["--length_scale", str(1.0 / speed)])

            process = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )

            if process.returncode != 0:
                os.unlink(temp_path)
                return AudioResult(
                    ok=False,
                    backend=self.backend_type.value,
                    error=f"Piper failed: {process.stderr.decode()}",
                )

            # Read the WAV file
            import wave
            with wave.open(temp_path, "rb") as wf:
                sample_rate = wf.getframerate()
                audio_data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio = audio / 32768.0

            os.unlink(temp_path)
            elapsed_ms = (time.time() - start_time) * 1000

            return AudioResult(
                ok=True,
                audio=audio,
                text=text,
                duration_ms=elapsed_ms,
                backend=self.backend_type.value,
                metadata={"sample_rate": sample_rate, "voice": model_path},
            )

        except Exception as e:
            logger.error(f"Piper TTS failed: {e}")
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )


class ESpeakBackend(AudioBackend):
    """TTS backend using eSpeak (lightweight, widely available)."""

    def __init__(self):
        self._available: Optional[bool] = None

    @property
    def backend_type(self) -> AudioBackendType:
        return AudioBackendType.ESPEAK

    @property
    def capabilities(self) -> List[AudioCapability]:
        return [AudioCapability.TTS]

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            result = subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True,
                timeout=5,
            )
            self._available = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                result = subprocess.run(
                    ["espeak", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                self._available = result.returncode == 0
            except (subprocess.SubprocessError, FileNotFoundError):
                self._available = False

        return self._available

    def _get_espeak_cmd(self) -> str:
        """Get the espeak command (espeak-ng or espeak)."""
        try:
            subprocess.run(["espeak-ng", "--version"], capture_output=True, timeout=2)
            return "espeak-ng"
        except:
            return "espeak"

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> AudioResult:
        if not self.is_available():
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error="eSpeak not available",
            )

        try:
            import time

            espeak_cmd = self._get_espeak_cmd()
            start_time = time.time()

            # Generate audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            # Calculate words per minute (default is ~175)
            wpm = int(175 * speed)

            cmd = [espeak_cmd, "-w", temp_path, "-s", str(wpm)]

            if voice:
                cmd.extend(["-v", voice])

            cmd.append(text)

            process = subprocess.run(cmd, capture_output=True, timeout=30)

            if process.returncode != 0:
                os.unlink(temp_path)
                return AudioResult(
                    ok=False,
                    backend=self.backend_type.value,
                    error=f"eSpeak failed: {process.stderr.decode()}",
                )

            # Read the WAV file
            import wave
            with wave.open(temp_path, "rb") as wf:
                sample_rate = wf.getframerate()
                audio_data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio = audio / 32768.0

            os.unlink(temp_path)
            elapsed_ms = (time.time() - start_time) * 1000

            return AudioResult(
                ok=True,
                audio=audio,
                text=text,
                duration_ms=elapsed_ms,
                backend=self.backend_type.value,
                metadata={"sample_rate": sample_rate, "voice": voice or "default"},
            )

        except Exception as e:
            logger.error(f"eSpeak TTS failed: {e}")
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )


class Pyttsx3Backend(AudioBackend):
    """TTS backend using pyttsx3 (cross-platform Python TTS)."""

    def __init__(self):
        self._available: Optional[bool] = None
        self._engine = None

    @property
    def backend_type(self) -> AudioBackendType:
        return AudioBackendType.PYTTSX3

    @property
    def capabilities(self) -> List[AudioCapability]:
        return [AudioCapability.TTS]

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            self._available = True
        except Exception as e:
            logger.debug(f"pyttsx3 not available: {e}")
            self._available = False

        return self._available

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
        return self._engine

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> AudioResult:
        if not self.is_available():
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error="pyttsx3 not available",
            )

        try:
            import time

            engine = self._get_engine()
            start_time = time.time()

            # Set voice if specified
            if voice:
                voices = engine.getProperty("voices")
                for v in voices:
                    if voice.lower() in v.name.lower() or voice == v.id:
                        engine.setProperty("voice", v.id)
                        break

            # Set rate (default is ~200 words/min)
            rate = int(200 * speed)
            engine.setProperty("rate", rate)

            # Generate to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            # Read the WAV file
            import wave
            with wave.open(temp_path, "rb") as wf:
                sample_rate = wf.getframerate()
                audio_data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio = audio / 32768.0

            os.unlink(temp_path)
            elapsed_ms = (time.time() - start_time) * 1000

            return AudioResult(
                ok=True,
                audio=audio,
                text=text,
                duration_ms=elapsed_ms,
                backend=self.backend_type.value,
                metadata={"sample_rate": sample_rate},
            )

        except Exception as e:
            logger.error(f"pyttsx3 TTS failed: {e}")
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.stop()
            self._engine = None
