"""
Lightweight system event logger for ContinuonBrain runtime.
Persists events to JSONL so the Web UI can render a recent activity log
(startups, reboots, service activations, etc.).
"""
import json
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SystemEvent:
    """Represents a single runtime event."""
    timestamp: float
    event_type: str
    message: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.setdefault("data", {})
        payload["iso_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))
        return payload


class SystemEventLogger:
    """Append-only logger backed by JSONL for quick dashboard reads."""

    def __init__(self, config_dir: str = "/tmp/continuonbrain_demo") -> None:
        self.config_dir = Path(config_dir)
        self.log_path = self.config_dir / "logs" / "system_events.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        event = SystemEvent(
            timestamp=time.time(),
            event_type=event_type,
            message=message,
            data=data or {},
        )
        line = json.dumps(event.to_dict())
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def load_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        if not self.log_path.exists():
            return []

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return []

        events: List[Dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        # Ensure newest-first ordering for the UI
        events.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return events
