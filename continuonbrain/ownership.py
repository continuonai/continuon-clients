"""
Ownership and access control metadata for ContinuonBrain instances.
Provides simple persistence of owner/renter/lease records in config_dir.
Initial creator defaults to the OEM owner unless overridden.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict


DEFAULT_OWNER = {
    "owner_id": "craig.merry",
    "owner_name": "Craig Michael Merry",
    "role": "creator",
}


@dataclass
class Lease:
    user_id: str
    user_name: str
    permissions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OwnershipRecord:
    owner_id: str
    owner_name: str
    creator_id: str
    creator_name: str
    renters: List[Lease] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "owner_id": self.owner_id,
            "owner_name": self.owner_name,
            "creator_id": self.creator_id,
            "creator_name": self.creator_name,
            "renters": [r.to_dict() for r in self.renters],
        }

    @staticmethod
    def from_dict(data: Dict) -> "OwnershipRecord":
        renters = [Lease(**r) for r in data.get("renters", [])]
        return OwnershipRecord(
            owner_id=data.get("owner_id", DEFAULT_OWNER["owner_id"]),
            owner_name=data.get("owner_name", DEFAULT_OWNER["owner_name"]),
            creator_id=data.get("creator_id", DEFAULT_OWNER["owner_id"]),
            creator_name=data.get("creator_name", DEFAULT_OWNER["owner_name"]),
            renters=renters,
        )


class OwnershipStore:
    def __init__(self, config_dir: str = "/tmp/continuonbrain_demo"):
        self.config_dir = Path(config_dir)
        self.path = self.config_dir / "ownership.json"
        self.record: OwnershipRecord = self._load()

    def _load(self) -> OwnershipRecord:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                return OwnershipRecord.from_dict(data)
            except Exception:
                pass
        # Default to creator as owner
        return OwnershipRecord(
            owner_id=DEFAULT_OWNER["owner_id"],
            owner_name=DEFAULT_OWNER["owner_name"],
            creator_id=DEFAULT_OWNER["owner_id"],
            creator_name=DEFAULT_OWNER["owner_name"],
            renters=[],
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.record.to_dict(), indent=2))

    def register_owner(self, owner_id: str, owner_name: str, actor_id: Optional[str] = None) -> bool:
        """Set a new owner; only creator or current owner can change it."""
        if actor_id and actor_id not in [self.record.owner_id, self.record.creator_id]:
            return False
        self.record.owner_id = owner_id
        self.record.owner_name = owner_name
        self.save()
        return True

    def add_renter(self, user_id: str, user_name: str, permissions: Optional[List[str]] = None, actor_id: Optional[str] = None) -> bool:
        if actor_id and actor_id not in [self.record.owner_id, self.record.creator_id]:
            return False
        lease = Lease(user_id=user_id, user_name=user_name, permissions=permissions or [])
        self.record.renters.append(lease)
        self.save()
        return True

    def is_owner(self, user_id: str) -> bool:
        return user_id == self.record.owner_id or user_id == self.record.creator_id

    def to_dict(self) -> Dict:
        return self.record.to_dict()


__all__ = ["OwnershipStore", "OwnershipRecord", "Lease", "DEFAULT_OWNER"]
