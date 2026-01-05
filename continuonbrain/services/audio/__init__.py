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
    AudioResult,
)
from .manager import AudioManager

__all__ = [
    'AudioBackend',
    'AudioBackendType',
    'AudioCapability',
    'AudioResult',
    'AudioManager',
]
