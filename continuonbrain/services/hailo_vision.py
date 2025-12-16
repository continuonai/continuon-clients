from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class HailoVisionState:
    enabled: bool
    available: bool
    hef_path: str
    last_ok: Optional[bool] = None
    last_ts: Optional[float] = None
    last_error: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "hef_path": self.hef_path,
            "last_ok": self.last_ok,
            "last_ts": self.last_ts,
            "last_error": self.last_error,
            # Keep last_result bounded; callers may prune further.
            "last_result": self.last_result,
        }


class HailoVision:
    """
    Subprocess-isolated Hailo inference for vision signals.

    Why subprocess: Hailo SDK/Python bindings can crash the interpreter on certain
    errors. Running it out-of-process keeps the main runtime resilient.
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        hef_path: Path = Path("/opt/continuonos/brain/model/base_model/model.hef"),
        topk: int = 5,
        timeout_s: float = 3.0,
    ) -> None:
        if enabled is None:
            enabled = os.environ.get("CONTINUON_HAILO_OFFLOAD", "1").lower() in ("1", "true", "yes", "on")
        self.hef_path = hef_path
        self.topk = max(1, min(int(topk), 10))
        self.timeout_s = max(0.5, float(timeout_s))
        self.python_exe = self._pick_python_exe()
        self.state = HailoVisionState(
            enabled=bool(enabled),
            available=False,
            hef_path=str(hef_path),
        )
        self._refresh_availability()

    def _pick_python_exe(self) -> str:
        """
        Pick a Python executable that can import hailo_platform.

        The runtime often runs in a venv that may not have the system-wide Hailo SDK
        installed. We prefer the first interpreter that can import hailo_platform.
        """
        override = os.environ.get("CONTINUON_HAILO_PYTHON")
        candidates = [override] if override else []
        candidates += [sys.executable, "/usr/bin/python3", "python3"]
        for exe in candidates:
            if not exe:
                continue
            try:
                proc = subprocess.run(
                    [exe, "-c", "import hailo_platform; print('ok')"],
                    capture_output=True,
                    timeout=2.0,
                    check=False,
                )
                if proc.returncode == 0:
                    return exe
            except Exception:
                continue
        # Fallback: still return sys.executable; worker will report a clean error.
        return sys.executable

    def _refresh_availability(self) -> None:
        if not self.state.enabled:
            self.state.available = False
            return
        try:
            has_dev = any(Path("/dev").glob("hailo*"))
            has_hef = Path(self.hef_path).exists()
            # Test hailo_platform availability in the worker interpreter.
            try:
                proc = subprocess.run(
                    [self.python_exe, "-c", "import hailo_platform; print('ok')"],
                    capture_output=True,
                    timeout=2.0,
                    check=False,
                )
                has_sdk = proc.returncode == 0
            except Exception:
                has_sdk = False
            self.state.available = bool(has_dev and has_hef and has_sdk)
        except Exception:
            self.state.available = False

    def get_state(self) -> Dict[str, Any]:
        self._refresh_availability()
        # Avoid dumping huge payloads.
        out = self.state.to_dict()
        if isinstance(out.get("last_result"), dict):
            # Bound last_result size
            lr = out["last_result"]
            if "topk" in lr and isinstance(lr["topk"], list):
                lr["topk"] = lr["topk"][:10]
        return out

    def infer_jpeg(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        self._refresh_availability()
        self.state.last_ts = time.time()
        if not self.state.enabled:
            self.state.last_ok = False
            self.state.last_error = "disabled"
            self.state.last_result = None
            return {"ok": False, "error": "hailo disabled"}
        if not self.state.available:
            self.state.last_ok = False
            self.state.last_error = "unavailable"
            self.state.last_result = None
            return {"ok": False, "error": "hailo unavailable"}

        env = dict(os.environ)
        env["CONTINUON_HAILO_HEF"] = str(self.hef_path)
        env["CONTINUON_HAILO_TOPK"] = str(self.topk)

        try:
            proc = subprocess.run(
                [self.python_exe, "-m", "continuonbrain.services.hailo_vision_worker"],
                input=jpeg_bytes,
                capture_output=True,
                env=env,
                timeout=self.timeout_s,
                check=False,
            )
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", errors="ignore")[:4000]
                self.state.last_ok = False
                self.state.last_error = f"worker_exit={proc.returncode} {err}".strip()
                self.state.last_result = None
                return {"ok": False, "error": self.state.last_error}
            raw = (proc.stdout or b"").decode("utf-8", errors="ignore")
            res = json.loads(raw) if raw else {"ok": False, "error": "empty worker output"}
            self.state.last_ok = bool(res.get("ok", False))
            self.state.last_error = None if self.state.last_ok else str(res.get("error") or "unknown")
            # Store bounded
            if isinstance(res, dict) and "topk" in res and isinstance(res["topk"], list):
                res["topk"] = res["topk"][:10]
            self.state.last_result = res if isinstance(res, dict) else None
            return res if isinstance(res, dict) else {"ok": False, "error": "non-dict worker output"}
        except subprocess.TimeoutExpired:
            self.state.last_ok = False
            self.state.last_error = f"timeout>{self.timeout_s}s"
            self.state.last_result = None
            return {"ok": False, "error": self.state.last_error}
        except Exception as exc:  # noqa: BLE001
            self.state.last_ok = False
            self.state.last_error = str(exc)
            self.state.last_result = None
            return {"ok": False, "error": str(exc)}


