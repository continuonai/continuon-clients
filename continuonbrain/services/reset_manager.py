from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class ResetProfile(str, Enum):
    """Reset profiles for seed brain lifecycle."""

    FACTORY = "factory"
    MEMORIES_ONLY = "memories_only"


@dataclass(frozen=True)
class ResetRequest:
    profile: ResetProfile
    dry_run: bool = False
    config_dir: Optional[Path] = None
    runtime_root: Path = Path("/opt/continuonos/brain")
    now_ts: Optional[float] = None


@dataclass
class ResetResult:
    success: bool
    profile: str
    dry_run: bool
    deleted: List[Dict[str, Any]]
    skipped: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    audit_path: Optional[str] = None
    elapsed_s: Optional[float] = None


def _as_path(p: Optional[Path | str]) -> Optional[Path]:
    if p is None:
        return None
    return p if isinstance(p, Path) else Path(p)


def _safe_rm(path: Path, *, dry_run: bool) -> Dict[str, Any]:
    """Best-effort delete of a file or directory."""
    try:
        if not path.exists():
            return {"path": str(path), "status": "missing"}
        if dry_run:
            return {"path": str(path), "status": "dry_run"}
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
        else:
            path.unlink()
        return {"path": str(path), "status": "deleted"}
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "status": "error", "error": str(exc)}


def _runtime_targets(runtime_root: Path, *, include_base_model: bool) -> List[Path]:
    """Canonical runtime paths under /opt/continuonos/brain."""
    targets: List[Path] = [
        runtime_root / "rlds" / "episodes",
        runtime_root / "rlds" / "compact",
        runtime_root / "rlds" / "tfrecord",
        runtime_root / "trainer" / "logs",
        runtime_root / "trainer" / "checkpoints",
        runtime_root / "model" / "adapters" / "current",
        runtime_root / "model" / "adapters" / "candidate",
    ]
    if include_base_model:
        targets.append(runtime_root / "model" / "base_model")
    return targets


def _config_targets(config_dir: Path) -> List[Path]:
    """Dev/runtime config-dir targets (BrainService(config_dir))."""
    return [
        config_dir / "experiences",
        config_dir / "memories" / "chat_logs",
        config_dir / "recordings" / "episodes",
    ]


def _resolve_audit_path(runtime_root: Path, config_dir: Optional[Path]) -> Path:
    """Prefer runtime audit location, fall back to config_dir, then /tmp."""
    candidates: List[Path] = [
        runtime_root / "trainer" / "logs" / "reset_audit.jsonl",
    ]
    if config_dir:
        candidates.append(config_dir / "trainer" / "logs" / "reset_audit.jsonl")
        candidates.append(config_dir / "reset_audit.jsonl")
    candidates.append(Path("/tmp/reset_audit.jsonl"))
    for c in candidates:
        try:
            c.parent.mkdir(parents=True, exist_ok=True)
            return c
        except Exception:
            continue
    return candidates[-1]


class ResetManager:
    """
    Seed-brain reset manager.

    Supports two profiles:
      - factory: wipe EVERYTHING including base model + current/candidate adapters.
      - memories_only: wipe RLDS/logs/experience/chat artifacts but keep model folders.
    """

    CONFIRM_FACTORY = "FACTORY RESET"
    CONFIRM_MEMORIES = "CLEAR MEMORIES"

    def __init__(self) -> None:
        # Admin token can be supplied by env or file.
        self._token_env = "CONTINUON_ADMIN_TOKEN"
        self._token_file_env = "CONTINUON_ADMIN_TOKEN_PATH"

    def _load_expected_token(self, runtime_root: Path, config_dir: Optional[Path]) -> Optional[str]:
        token = os.environ.get(self._token_env)
        if token:
            return token.strip()

        path_env = os.environ.get(self._token_file_env)
        if path_env:
            p = Path(path_env)
            if p.exists():
                try:
                    return p.read_text(encoding="utf-8").strip()
                except Exception:
                    return None

        # Default token file locations (opt-in)
        candidates: List[Path] = [
            runtime_root / "admin_token.txt",
        ]
        if config_dir:
            candidates.append(config_dir / "admin_token.txt")
        for p in candidates:
            if p.exists():
                try:
                    return p.read_text(encoding="utf-8").strip()
                except Exception:
                    continue
        return None

    def authorize(self, *, provided_token: Optional[str], runtime_root: Path, config_dir: Optional[Path]) -> bool:
        """
        Verify reset authorization.

        - If an expected token exists (env/file), it must match.
        - If no expected token exists, resets are denied unless CONTINUON_ALLOW_UNSAFE_RESET=1.
        """
        expected = self._load_expected_token(runtime_root, config_dir)
        if expected:
            return bool(provided_token) and provided_token.strip() == expected
        return os.environ.get("CONTINUON_ALLOW_UNSAFE_RESET", "0").lower() in ("1", "true", "yes")

    def build_targets(self, req: ResetRequest) -> Dict[str, List[str]]:
        include_base = req.profile == ResetProfile.FACTORY
        targets: List[Path] = []
        targets.extend(_runtime_targets(req.runtime_root, include_base_model=include_base))

        if req.profile == ResetProfile.MEMORIES_ONLY:
            # Remove adapters/checkpoints/logs/RLDS but keep base_model + current/candidate.
            # We already included adapters/current + candidate above; filter them out.
            keep_prefixes = [
                req.runtime_root / "model" / "adapters" / "current",
                req.runtime_root / "model" / "adapters" / "candidate",
                req.runtime_root / "model" / "base_model",
            ]
            targets = [p for p in targets if not any(str(p).startswith(str(k)) for k in keep_prefixes)]

        cfg_dir = _as_path(req.config_dir)
        if cfg_dir:
            targets.extend(_config_targets(cfg_dir))

        # Deduplicate while preserving order
        seen = set()
        out: List[str] = []
        for p in targets:
            s = str(p)
            if s not in seen:
                seen.add(s)
                out.append(s)
        return {"targets": out}

    def run(self, req: ResetRequest) -> ResetResult:
        start = time.time()
        now = req.now_ts if req.now_ts is not None else time.time()

        targets = self.build_targets(req)["targets"]
        deleted: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for t in targets:
            path = Path(t)
            res = _safe_rm(path, dry_run=req.dry_run)
            if res["status"] in ("deleted", "dry_run"):
                deleted.append(res)
            elif res["status"] == "missing":
                skipped.append(res)
            else:
                errors.append(res)

        audit_path = _resolve_audit_path(req.runtime_root, _as_path(req.config_dir))
        audit_entry = {
            "ts": now,
            "profile": req.profile.value,
            "dry_run": req.dry_run,
            "runtime_root": str(req.runtime_root),
            "config_dir": str(req.config_dir) if req.config_dir else None,
            "deleted": deleted,
            "skipped": skipped,
            "errors": errors,
        }
        try:
            if not req.dry_run:
                with audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(audit_entry) + "\n")
        except Exception:
            # Never fail reset purely due to audit write.
            pass

        elapsed = time.time() - start
        ok = len(errors) == 0
        return ResetResult(
            success=ok,
            profile=req.profile.value,
            dry_run=req.dry_run,
            deleted=deleted,
            skipped=skipped,
            errors=errors,
            audit_path=str(audit_path),
            elapsed_s=elapsed,
        )


