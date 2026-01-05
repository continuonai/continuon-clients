"""
ContinuonBrain Audio System

Unified audio system for microphone input, speaker output, and speech processing:
- MicrophoneBackend: Audio capture from microphone
- SpeakerBackend: Audio playback through speakers
- TTSBackend: Text-to-speech synthesis
- STTBackend: Speech-to-text transcription
- AudioManager: Unified orchestrator

Usage:
    from continuonbrain.services.audio import AudioManager

    manager = AudioManager()

    # Capture audio
    audio = manager.capture(duration_ms=2000)

    # Transcribe
    result = manager.transcribe(audio)
    print(result["text"])

    # Speak response
    manager.speak("Hello, I heard you say: " + result["text"])
"""
from .backend import (
    AudioBackend,
    AudioBackendType,
    AudioCapability,
    AudioDevice,
    AudioResult,
)
from .manager import AudioManager
from .microphone_backend import PyAudioBackend, SoundDeviceBackend
from .tts_backend import PiperBackend, ESpeakBackend, Pyttsx3Backend
from .stt_backend import WhisperBackend, VoskBackend

__all__ = [
    # Base classes
    'AudioBackend',
    'AudioBackendType',
    'AudioCapability',
    'AudioDevice',
    'AudioResult',
    # Manager
    'AudioManager',
    # Capture backends
    'PyAudioBackend',
    'SoundDeviceBackend',
    # TTS backends
    'PiperBackend',
    'ESpeakBackend',
    'Pyttsx3Backend',
    # STT backends
    'WhisperBackend',
    'VoskBackend',
]
