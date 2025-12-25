from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class PairingSession:
    token: str
    confirm_code: str
    created_unix_s: int
    expires_unix_s: int
    base_url: str

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/pair?token={self.token}"


class PairingManager:
    """Offline-first local pairing for ownership claim via QR + 6-digit confirm code.

    Security model:
    - Intended for local-network usage.
    - Requires physical access to the robot UI to see the 6-digit code.
    - Sessions are short-lived and stored on disk (config_dir).
    """

    def __init__(self, config_dir: str) -> None:
        self.config_dir = str(config_dir)
        self._pairing_dir = Path(self.config_dir) / "pairing"
        self._pending_path = self._pairing_dir / "pending.json"
        self._ownership_path = Path(self.config_dir) / "ownership.json"

    def start(self, *, base_url: str, ttl_s: int = 300) -> PairingSession:
        now = int(time.time())
        ttl_s = max(60, min(int(ttl_s), 900))
        token = secrets.token_urlsafe(24)
        confirm_code = f"{secrets.randbelow(1_000_000):06d}"
        session = PairingSession(
            token=token,
            confirm_code=confirm_code,
            created_unix_s=now,
            expires_unix_s=now + ttl_s,
            base_url=str(base_url or "").strip() or "http://127.0.0.1:8080",
        )
        self._pairing_dir.mkdir(parents=True, exist_ok=True)
        self._pending_path.write_text(json.dumps(session.__dict__, indent=2, default=str))
        return session

    def get_pending(self) -> Optional[PairingSession]:
        try:
            raw = json.loads(self._pending_path.read_text())
            if not isinstance(raw, dict):
                return None
            session = PairingSession(
                token=str(raw.get("token") or ""),
                confirm_code=str(raw.get("confirm_code") or ""),
                created_unix_s=int(raw.get("created_unix_s") or 0),
                expires_unix_s=int(raw.get("expires_unix_s") or 0),
                base_url=str(raw.get("base_url") or ""),
            )
            if not session.token or not session.confirm_code:
                return None
            return session
        except Exception:
            return None

    def confirm(self, *, token: str, confirm_code: str, owner_id: str, account_id: Optional[str] = None, account_type: Optional[str] = "local_pairing") -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        session = self.get_pending()
        if not session:
            return False, "No pending pairing session", None
        now = int(time.time())
        if now > session.expires_unix_s:
            return False, "Pairing session expired", None
        if str(token or "").strip() != session.token:
            return False, "Invalid pairing token", None
        if str(confirm_code or "").strip() != session.confirm_code:
            return False, "Invalid confirm code", None

        owner_id = str(owner_id or "").strip()
        if not owner_id:
            return False, "owner_id is required", None

        payload = {
            "owned": True,
            "subscription_active": False,
            "seed_installed": False,
            "account_id": account_id,
            "account_type": account_type or "local_pairing",
            "owner_id": owner_id,
            "paired_unix_s": now,
        }
        try:
            self._ownership_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception as exc:
            return False, f"Failed to write ownership.json: {exc}", None

        # Invalidate pending session after success.
        try:
            self._pending_path.unlink(missing_ok=True)  # py3.8+: ok on 3.11+
        except Exception:
            pass

        return True, "paired", payload

    def ownership_status(self) -> Dict[str, Any]:
        out = {"owned": False, "owner_id": None}
        try:
            if self._ownership_path.exists():
                data = json.loads(self._ownership_path.read_text())
                if isinstance(data, dict):
                    out["owned"] = bool(data.get("owned", False))
                    out["owner_id"] = data.get("owner_id")
                    out["account_type"] = data.get("account_type")
                    out["paired_unix_s"] = data.get("paired_unix_s")
        except Exception:
            pass
        return out


