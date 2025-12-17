from __future__ import annotations

import io
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class AudioRecordResult:
    path: Path
    sample_rate_hz: int
    num_channels: int
    duration_s: float
    backend: str


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(value)
    except Exception:
        return default
    return max(lo, min(hi, v))


def speak_text(
    text: str,
    *,
    rate_wpm: int = 175,
    voice: str = "en",
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """Offline-first TTS using system utilities (espeak-ng/espeak).

    Returns a small status dict for API responses.
    """
    msg = (text or "").strip()
    if not msg:
        return {"status": "error", "message": "No text provided"}
    # Keep it bounded so the robot doesn't speak forever.
    msg = msg[:800]

    rate_wpm = _clamp_int(rate_wpm, 80, 320, 175)
    cmd = None
    if shutil.which("espeak-ng"):
        cmd = ["espeak-ng", "-s", str(rate_wpm), "-v", str(voice), msg]
    elif shutil.which("espeak"):
        cmd = ["espeak", "-s", str(rate_wpm), "-v", str(voice), msg]

    if not cmd:
        return {"status": "error", "message": "No TTS backend found (install espeak-ng or espeak)"}

    try:
        subprocess.run(cmd, check=True, timeout=timeout_s)
        return {"status": "ok", "backend": cmd[0], "rate_wpm": rate_wpm, "voice": voice}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": f"TTS timed out after {timeout_s}s", "backend": cmd[0]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc), "backend": cmd[0]}


def _record_with_sounddevice(
    *,
    out_path: Path,
    seconds: int,
    sample_rate_hz: int,
    num_channels: int,
    device: Optional[Any] = None,
) -> Optional[AudioRecordResult]:
    # Optional dependency; keep imports guarded.
    try:
        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore
    except Exception:
        return None

    frames = int(seconds * sample_rate_hz)
    audio = sd.rec(frames, samplerate=sample_rate_hz, channels=num_channels, dtype="int16", device=device)
    sd.wait(timeout=seconds + 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate_hz)
        wf.writeframes(audio.tobytes())
    duration_s = float(frames) / float(sample_rate_hz)
    return AudioRecordResult(
        path=out_path,
        sample_rate_hz=sample_rate_hz,
        num_channels=num_channels,
        duration_s=duration_s,
        backend="sounddevice",
    )


def _resolve_arecord() -> Optional[str]:
    path = shutil.which("arecord")
    if path:
        return path
    # systemd services sometimes run with a minimal PATH; fall back to common locations.
    for candidate in ("/usr/bin/arecord", "/bin/arecord", "/usr/local/bin/arecord"):
        if Path(candidate).exists():
            return candidate
    return None


def _record_with_arecord(
    *,
    out_path: Path,
    seconds: int,
    sample_rate_hz: int,
    num_channels: int,
    device: Optional[str] = None,
) -> Tuple[Optional[AudioRecordResult], Optional[str]]:
    arecord = _resolve_arecord()
    if not arecord:
        return None, None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        arecord,
    ]
    if device:
        cmd += ["-D", str(device)]
    cmd += [
        "-d",
        str(seconds),
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate_hz),
        "-c",
        str(num_channels),
        str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            timeout=float(seconds) + 3.0,
            capture_output=True,
            text=True,
        )
        return AudioRecordResult(
            path=out_path,
            sample_rate_hz=sample_rate_hz,
            num_channels=num_channels,
            duration_s=float(seconds),
            backend="arecord",
        ), None
    except subprocess.TimeoutExpired:
        return None, f"arecord timed out after {float(seconds) + 3.0}s"
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        return None, (err or f"arecord failed with exit code {exc.returncode}")
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def record_wav(
    *,
    seconds: int = 4,
    sample_rate_hz: int = 16000,
    num_channels: int = 1,
    device: Optional[str] = None,
    out_dir: Path = Path("/opt/continuonos/brain/audio"),
) -> Tuple[Optional[AudioRecordResult], Dict[str, Any]]:
    """Record a short WAV clip from the robot microphone, best-effort."""
    seconds = _clamp_int(seconds, 1, 15, 4)
    sample_rate_hz = _clamp_int(sample_rate_hz, 8000, 48000, 16000)
    num_channels = _clamp_int(num_channels, 1, 2, 1)

    ts = int(time.time())
    out_path = out_dir / f"mic_{ts}.wav"

    res = _record_with_sounddevice(
        out_path=out_path,
        seconds=seconds,
        sample_rate_hz=sample_rate_hz,
        num_channels=num_channels,
        device=device,
    )
    if res:
        return res, {"status": "ok"}

    res2, err = _record_with_arecord(
        out_path=out_path,
        seconds=seconds,
        sample_rate_hz=sample_rate_hz,
        num_channels=num_channels,
        device=device,
    )
    if res2:
        return res2, {"status": "ok"}
    if err:
        return None, {"status": "error", "backend": "arecord", "message": err}

    return None, {"status": "error", "message": "No microphone backend available (install sounddevice or arecord)"}


