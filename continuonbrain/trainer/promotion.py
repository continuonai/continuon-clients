"""
Adapter promotion + rollback utilities (offline-first).

This module centralizes candidate -> current promotion so:
- promotions are atomic (as much as possible on local FS),
- prior current adapters are archived to history/,
- every decision emits an audit log + a small RLDS episode_dir for provenance,
- rollback is possible by restoring from history.

It is intentionally dependency-light and safe to import on Pi-class devices.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _touch(path: Path) -> None:
    try:
        now = time.time()
        os.utime(path, (now, now))
    except Exception:
        pass


@dataclass
class PromotionRecord:
    timestamp_unix_s: float
    promoted: bool
    reason: str
    candidate_path: str
    current_path: str
    archived_previous_current: Optional[str]
    sha256: Optional[str]
    size_bytes: Optional[int]
    audit_log_path: Optional[str] = None
    rlds_episode_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_unix_s": self.timestamp_unix_s,
            "promoted": self.promoted,
            "reason": self.reason,
            "candidate_path": self.candidate_path,
            "current_path": self.current_path,
            "archived_previous_current": self.archived_previous_current,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "audit_log_path": self.audit_log_path,
            "rlds_episode_dir": self.rlds_episode_dir,
        }


def write_promotion_audit(log_dir: Path, record: PromotionRecord) -> Path:
    _ensure_dir(log_dir)
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(record.timestamp_unix_s))
    path = log_dir / f"promotion_{ts}.json"
    path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_promotion_rlds_episode(
    rlds_root: Path,
    *,
    record: PromotionRecord,
    environment_id: str = "pi5-dev",
    xr_mode: str = "sleep_learning",
    control_role: str = "human_supervisor",
) -> Path:
    """
    Log a tiny canonical episode_dir for provenance. This is not “robotics strict” RLDS;
    it’s intended for audit trails and replay tooling.
    """
    episode_id = f"adapter_promotion_{int(record.timestamp_unix_s)}_{uuid.uuid4().hex[:8]}"
    episode_dir = rlds_root / episode_id
    steps_dir = episode_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "blobs").mkdir(parents=True, exist_ok=True)

    metadata = {
        "xr_mode": xr_mode,
        "control_role": control_role,
        "environment_id": environment_id,
        "tags": [
            "adapter_promotion",
            f"promoted:{str(record.promoted).lower()}",
            f"reason:{record.reason}",
        ],
        "software": {
            "xr_app": "n/a",
            "continuonbrain_os": "dev",
            "glove_firmware": "n/a",
        },
        "start_time_unix_ms": int(record.timestamp_unix_s * 1000),
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    step_metadata = {
        "timestamp": str(record.timestamp_unix_s),
        "promoted": str(record.promoted).lower(),
        "reason": record.reason,
        "candidate_path": record.candidate_path,
        "current_path": record.current_path,
    }
    if record.archived_previous_current:
        step_metadata["archived_previous_current"] = record.archived_previous_current
    if record.sha256:
        step_metadata["sha256"] = record.sha256
    if record.size_bytes is not None:
        step_metadata["size_bytes"] = str(record.size_bytes)

    step = {
        "observation": {"promotion": record.to_dict()},
        "action": {"kind": "promote_adapter", "source": "sleep_learning"},
        "reward": 0.0,
        "is_terminal": True,
        "step_metadata": step_metadata,
    }
    (steps_dir / "000000.jsonl").write_text(json.dumps(step, sort_keys=True) + "\n", encoding="utf-8")
    return episode_dir


def promote_candidate_adapter(
    *,
    candidate_path: Path,
    current_dir: Path,
    history_dir: Path,
    log_dir: Path,
    rlds_dir: Optional[Path] = None,
    reason: str,
    environment_id: str = "pi5-dev",
    manifest_path: Optional[Path] = Path("/opt/continuonos/brain/model/manifest.json"),
) -> PromotionRecord:
    """
    Promote candidate -> current and archive previous current -> history.

    This function always emits an audit record (and optionally an RLDS episode_dir),
    whether promotion succeeded or was rejected.
    """
    ts = time.time()
    candidate_path = candidate_path.expanduser()
    current_dir = current_dir.expanduser()
    history_dir = history_dir.expanduser()
    log_dir = log_dir.expanduser()

    current_path = current_dir / candidate_path.name
    archived_previous: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None

    promoted = False
    if candidate_path.exists() and reason == "approved":
        _ensure_dir(current_dir)
        _ensure_dir(history_dir)
        if current_path.exists():
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(ts))
            archived = history_dir / f"{candidate_path.stem}_{stamp}{candidate_path.suffix}"
            shutil.move(str(current_path), str(archived))
            archived_previous = str(archived)

        sha256 = _sha256(candidate_path)
        size_bytes = candidate_path.stat().st_size
        shutil.move(str(candidate_path), str(current_path))
        promoted = True

        # Hint the runtime to reload adapters if it watches manifest mtime.
        if manifest_path and manifest_path.exists():
            _touch(manifest_path)
        # Also create a marker file under current/.
        try:
            marker = current_dir / ".last_promotion.json"
            marker.write_text(
                json.dumps(
                    {
                        "timestamp_unix_s": ts,
                        "current_path": str(current_path),
                        "sha256": sha256,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    record = PromotionRecord(
        timestamp_unix_s=ts,
        promoted=promoted,
        reason=reason,
        candidate_path=str(candidate_path),
        current_path=str(current_path),
        archived_previous_current=archived_previous,
        sha256=sha256,
        size_bytes=size_bytes,
    )
    audit_path = write_promotion_audit(log_dir, record)
    record.audit_log_path = str(audit_path)

    if rlds_dir:
        try:
            _ensure_dir(rlds_dir)
            ep_dir = write_promotion_rlds_episode(rlds_dir, record=record, environment_id=environment_id)
            record.rlds_episode_dir = str(ep_dir)
        except Exception:
            record.rlds_episode_dir = None
    return record


def rollback_to_history(
    *,
    history_path: Path,
    current_dir: Path,
    log_dir: Path,
    reason: str = "manual_rollback",
) -> PromotionRecord:
    """
    Restore a historical adapter into current/.
    """
    ts = time.time()
    history_path = history_path.expanduser()
    current_dir = current_dir.expanduser()
    log_dir = log_dir.expanduser()

    current_path = current_dir / history_path.name
    archived_previous: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None

    promoted = False
    if history_path.exists():
        _ensure_dir(current_dir)
        if current_path.exists():
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(ts))
            archived = history_path.parent / f"{current_path.stem}_replaced_{stamp}{current_path.suffix}"
            shutil.move(str(current_path), str(archived))
            archived_previous = str(archived)
        sha256 = _sha256(history_path)
        size_bytes = history_path.stat().st_size
        shutil.copy2(str(history_path), str(current_path))
        promoted = True

    record = PromotionRecord(
        timestamp_unix_s=ts,
        promoted=promoted,
        reason=reason,
        candidate_path=str(history_path),
        current_path=str(current_path),
        archived_previous_current=archived_previous,
        sha256=sha256,
        size_bytes=size_bytes,
    )
    audit_path = write_promotion_audit(log_dir, record)
    record.audit_log_path = str(audit_path)
    return record


