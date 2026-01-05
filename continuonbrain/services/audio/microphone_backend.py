"""
Microphone Backend

Audio capture backends for microphone input.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .backend import (
    AudioBackend,
    AudioBackendType,
    AudioCapability,
    AudioDevice,
    AudioResult,
)

logger = logging.getLogger(__name__)


class PyAudioBackend(AudioBackend):
    """Microphone backend using PyAudio."""

    def __init__(self):
        self._pyaudio = None
        self._stream = None
        self._available: Optional[bool] = None

    @property
    def backend_type(self) -> AudioBackendType:
        return AudioBackendType.PYAUDIO

    @property
    def capabilities(self) -> List[AudioCapability]:
        return [AudioCapability.CAPTURE, AudioCapability.PLAYBACK]

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            # Check if there's at least one input device
            device_count = pa.get_device_count()
            has_input = False
            for i in range(device_count):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    has_input = True
                    break
            pa.terminate()
            self._available = has_input
        except Exception as e:
            logger.debug(f"PyAudio not available: {e}")
            self._available = False

        return self._available

    def _get_pyaudio(self):
        if self._pyaudio is None:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio

    def capture(
        self,
        duration_ms: int = 1000,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> AudioResult:
        if not self.is_available():
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error="PyAudio not available",
            )

        try:
            import pyaudio
            import time

            pa = self._get_pyaudio()
            frames_per_buffer = 1024
            format_type = pyaudio.paInt16

            stream = pa.open(
                format=format_type,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=frames_per_buffer,
            )

            start_time = time.time()
            frames = []
            num_frames = int(sample_rate * duration_ms / 1000)

            while len(frames) * frames_per_buffer < num_frames:
                data = stream.read(frames_per_buffer, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            elapsed_ms = (time.time() - start_time) * 1000

            # Convert to numpy array
            audio_data = b"".join(frames)
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Normalize to [-1, 1]

            return AudioResult(
                ok=True,
                audio=audio,
                duration_ms=elapsed_ms,
                backend=self.backend_type.value,
                metadata={"sample_rate": sample_rate, "channels": channels},
            )

        except Exception as e:
            logger.error(f"PyAudio capture failed: {e}")
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        blocking: bool = True,
    ) -> AudioResult:
        if not self.is_available():
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error="PyAudio not available",
            )

        try:
            import pyaudio
            import time

            pa = self._get_pyaudio()

            # Convert to int16
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio.astype(np.int16)

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
            )

            start_time = time.time()
            stream.write(audio_int16.tobytes())

            if blocking:
                stream.stop_stream()
                stream.close()

            elapsed_ms = (time.time() - start_time) * 1000

            return AudioResult(
                ok=True,
                duration_ms=elapsed_ms,
                backend=self.backend_type.value,
            )

        except Exception as e:
            logger.error(f"PyAudio playback failed: {e}")
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    def get_devices(self) -> Dict[str, List[AudioDevice]]:
        if not self.is_available():
            return {"input_devices": [], "output_devices": []}

        try:
            pa = self._get_pyaudio()
            input_devices = []
            output_devices = []

            default_input = pa.get_default_input_device_info()
            default_output = pa.get_default_output_device_info()

            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)

                if info.get("maxInputChannels", 0) > 0:
                    input_devices.append(AudioDevice(
                        id=i,
                        name=info.get("name", f"Device {i}"),
                        channels=info.get("maxInputChannels", 1),
                        sample_rate=int(info.get("defaultSampleRate", 16000)),
                        is_default=(i == default_input.get("index")),
                        is_input=True,
                    ))

                if info.get("maxOutputChannels", 0) > 0:
                    output_devices.append(AudioDevice(
                        id=i,
                        name=info.get("name", f"Device {i}"),
                        channels=info.get("maxOutputChannels", 2),
                        sample_rate=int(info.get("defaultSampleRate", 16000)),
                        is_default=(i == default_output.get("index")),
                        is_input=False,
                    ))

            return {"input_devices": input_devices, "output_devices": output_devices}

        except Exception as e:
            logger.error(f"Failed to get devices: {e}")
            return {"input_devices": [], "output_devices": []}

    def shutdown(self) -> None:
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None


class SoundDeviceBackend(AudioBackend):
    """Microphone backend using sounddevice library."""

    def __init__(self):
        self._available: Optional[bool] = None

    @property
    def backend_type(self) -> AudioBackendType:
        return AudioBackendType.SOUNDDEVICE

    @property
    def capabilities(self) -> List[AudioCapability]:
        return [AudioCapability.CAPTURE, AudioCapability.PLAYBACK]

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import sounddevice as sd
            devices = sd.query_devices()
            self._available = len(devices) > 0
        except Exception as e:
            logger.debug(f"sounddevice not available: {e}")
            self._available = False

        return self._available

    def capture(
        self,
        duration_ms: int = 1000,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> AudioResult:
        if not self.is_available():
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error="sounddevice not available",
            )

        try:
            import sounddevice as sd
            import time

            duration_sec = duration_ms / 1000.0
            start_time = time.time()

            audio = sd.rec(
                int(duration_sec * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype=np.float32,
            )
            sd.wait()

            elapsed_ms = (time.time() - start_time) * 1000

            # Flatten if mono
            if channels == 1:
                audio = audio.flatten()

            return AudioResult(
                ok=True,
                audio=audio,
                duration_ms=elapsed_ms,
                backend=self.backend_type.value,
                metadata={"sample_rate": sample_rate, "channels": channels},
            )

        except Exception as e:
            logger.error(f"sounddevice capture failed: {e}")
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        blocking: bool = True,
    ) -> AudioResult:
        if not self.is_available():
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error="sounddevice not available",
            )

        try:
            import sounddevice as sd
            import time

            start_time = time.time()
            sd.play(audio, samplerate=sample_rate)

            if blocking:
                sd.wait()

            elapsed_ms = (time.time() - start_time) * 1000

            return AudioResult(
                ok=True,
                duration_ms=elapsed_ms,
                backend=self.backend_type.value,
            )

        except Exception as e:
            logger.error(f"sounddevice playback failed: {e}")
            return AudioResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    def get_devices(self) -> Dict[str, List[AudioDevice]]:
        if not self.is_available():
            return {"input_devices": [], "output_devices": []}

        try:
            import sounddevice as sd

            devices = sd.query_devices()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]

            input_devices = []
            output_devices = []

            for i, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    input_devices.append(AudioDevice(
                        id=i,
                        name=dev.get("name", f"Device {i}"),
                        channels=dev.get("max_input_channels", 1),
                        sample_rate=int(dev.get("default_samplerate", 16000)),
                        is_default=(i == default_input),
                        is_input=True,
                    ))

                if dev.get("max_output_channels", 0) > 0:
                    output_devices.append(AudioDevice(
                        id=i,
                        name=dev.get("name", f"Device {i}"),
                        channels=dev.get("max_output_channels", 2),
                        sample_rate=int(dev.get("default_samplerate", 16000)),
                        is_default=(i == default_output),
                        is_input=False,
                    ))

            return {"input_devices": input_devices, "output_devices": output_devices}

        except Exception as e:
            logger.error(f"Failed to get devices: {e}")
            return {"input_devices": [], "output_devices": []}
