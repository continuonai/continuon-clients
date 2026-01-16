"""
Audio management for Trainer UI.

Provides TTS functionality using espeak-ng or system alternatives.
"""

import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class AudioConfig:
    """Audio configuration."""
    available: bool = False
    backend: str = ""
    rate_wpm: int = 175
    voice: str = "en"


class AudioManager:
    """
    Audio manager for TTS output.
    Uses espeak-ng (preferred), espeak, or macOS say command.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio manager.

        Args:
            config: AudioConfig or None for auto-detect
        """
        self.config = config or self._detect_audio()

    def _detect_audio(self) -> AudioConfig:
        """Detect available audio backend."""
        config = AudioConfig()

        # Check for espeak-ng first (preferred)
        if shutil.which("espeak-ng"):
            config.available = True
            config.backend = "espeak-ng"
            return config

        # Fall back to espeak
        if shutil.which("espeak"):
            config.available = True
            config.backend = "espeak"
            return config

        # Check for macOS say command
        if shutil.which("say"):
            config.available = True
            config.backend = "say"
            return config

        return config

    def speak(
        self,
        text: str,
        rate_wpm: Optional[int] = None,
        voice: Optional[str] = None,
        timeout_s: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Speak text using TTS.

        Args:
            text: Text to speak
            rate_wpm: Words per minute (80-320, default 175)
            voice: Voice/language code (e.g., "en", "en-us")
            timeout_s: Timeout in seconds

        Returns:
            Dict with status and details
        """
        if not self.config.available:
            return {
                "status": "error",
                "message": "No TTS backend available (install espeak-ng)",
            }

        # Clean and truncate text
        text = (text or "").strip()
        if not text:
            return {"status": "error", "message": "No text provided"}
        text = text[:800]  # Limit length

        # Use defaults if not specified
        rate_wpm = rate_wpm or self.config.rate_wpm
        voice = voice or self.config.voice
        rate_wpm = max(80, min(320, rate_wpm))

        # Build command based on backend
        cmd = None
        if self.config.backend == "espeak-ng":
            cmd = ["espeak-ng", "-s", str(rate_wpm), "-v", voice, text]
        elif self.config.backend == "espeak":
            cmd = ["espeak", "-s", str(rate_wpm), "-v", voice, text]
        elif self.config.backend == "say":
            # macOS say command has different syntax
            cmd = ["say", "-r", str(rate_wpm), text]
        elif self.config.backend == "mock":
            # Mock mode - just pretend to speak
            print(f"MOCK TTS: {text}")
            return {
                "status": "ok",
                "backend": "mock",
                "text": text,
            }

        if not cmd:
            return {
                "status": "error",
                "message": f"Unknown audio backend: {self.config.backend}",
            }

        try:
            subprocess.run(cmd, check=True, timeout=timeout_s)
            return {
                "status": "ok",
                "backend": self.config.backend,
                "rate_wpm": rate_wpm,
                "voice": voice,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": f"TTS timed out after {timeout_s}s",
                "backend": self.config.backend,
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "message": f"TTS failed: {e}",
                "backend": self.config.backend,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "backend": self.config.backend,
            }

    def get_status(self) -> Dict[str, Any]:
        """Get audio system status."""
        return {
            "available": self.config.available,
            "backend": self.config.backend,
            "rate_wpm": self.config.rate_wpm,
            "voice": self.config.voice,
        }
