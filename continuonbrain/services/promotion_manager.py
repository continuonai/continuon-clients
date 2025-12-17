from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PromotionResult:
    success: bool
    dry_run: bool
    candidate_dir: str
    current_dir: str
    backup_dir: Optional[str]
    message: str
    audit_path: Optional[str] = None


def _resolve_audit_path(runtime_root: Path) -> Path:
    candidates = [
        runtime_root / "trainer" / "logs" / "promotion_audit.jsonl",
        Path("/tmp/promotion_audit.jsonl"),
    ]
    for c in candidates:
        try:
            c.parent.mkdir(parents=True, exist_ok=True)
            return c
        except Exception:
            continue
    return candidates[-1]


class PromotionManager:
    """
    Promote a candidate seed export into the current adapter slot.

    This is filesystem-only and intentionally conservative:
    - Candidate must exist and contain a manifest
    - Current is backed up with a timestamp before swap
    - Operation is atomic-ish via rename when possible
    """

    def __init__(self) -> None:
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

        candidates = [runtime_root / "admin_token.txt"]
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
        expected = self._load_expected_token(runtime_root, config_dir)
        if expected:
            return bool(provided_token) and provided_token.strip() == expected
        return os.environ.get("CONTINUON_ALLOW_UNSAFE_RESET", "0").lower() in ("1", "true", "yes")

    def promote(
        self,
        *,
        runtime_root: Path = Path("/opt/continuonos/brain"),
        candidate_rel: Path = Path("model/adapters/candidate/core_model_seed"),
        current_rel: Path = Path("model/adapters/current/core_model_seed"),
        dry_run: bool = False,
    ) -> PromotionResult:
        candidate_dir = runtime_root / candidate_rel
        current_dir = runtime_root / current_rel
        manifest = candidate_dir / "model_manifest.json"

        if not candidate_dir.exists() or not candidate_dir.is_dir():
            return PromotionResult(
                success=False,
                dry_run=dry_run,
                candidate_dir=str(candidate_dir),
                current_dir=str(current_dir),
                backup_dir=None,
                message="candidate dir missing",
            )
        if not manifest.exists():
            return PromotionResult(
                success=False,
                dry_run=dry_run,
                candidate_dir=str(candidate_dir),
                current_dir=str(current_dir),
                backup_dir=None,
                message="candidate manifest missing",
            )

        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = current_dir.parent / f"{current_dir.name}_backup_{ts}"

        audit_path = _resolve_audit_path(runtime_root)
        entry: Dict[str, Any] = {
            "ts": time.time(),
            "dry_run": dry_run,
            "candidate_dir": str(candidate_dir),
            "current_dir": str(current_dir),
            "backup_dir": str(backup_dir),
        }

        try:
            if dry_run:
                entry["status"] = "dry_run"
                return PromotionResult(
                    success=True,
                    dry_run=True,
                    candidate_dir=str(candidate_dir),
                    current_dir=str(current_dir),
                    backup_dir=str(backup_dir),
                    message="dry run ok",
                    audit_path=str(audit_path),
                )

            current_dir.parent.mkdir(parents=True, exist_ok=True)

            # Backup current if present
            if current_dir.exists():
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                shutil.move(str(current_dir), str(backup_dir))

            # Promote by move (atomic within filesystem). Recreate candidate parent afterward.
            shutil.move(str(candidate_dir), str(current_dir))
            candidate_dir.parent.mkdir(parents=True, exist_ok=True)

            entry["status"] = "ok"
            entry["result"] = "promoted"
            ok = True
            msg = "promoted candidate to current"
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "error"
            entry["error"] = str(exc)
            ok = False
            msg = str(exc)

        try:
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        return PromotionResult(
            success=ok,
            dry_run=dry_run,
            candidate_dir=str(candidate_dir),
            current_dir=str(current_dir),
            backup_dir=str(backup_dir) if backup_dir else None,
            message=msg,
            audit_path=str(audit_path),
        )


