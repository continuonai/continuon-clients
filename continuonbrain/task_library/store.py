"""Task Library index builder and lookup helpers.

This module provides a lightweight JSON index for RLDS episodes on disk. The
index groups episodes by XR mode, control role, and tags to make UI/robot task
selection easier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TaskCategory:
    """Represents the UI/robot selection category for an RLDS episode."""

    xr_mode: str
    control_role: str
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def matches(self, xr_mode: Optional[str], control_role: Optional[str], tag: Optional[str]) -> bool:
        if xr_mode and xr_mode != self.xr_mode:
            return False
        if control_role and control_role != self.control_role:
            return False
        if tag and tag not in self.tags:
            return False
        return True

    def as_dict(self) -> Dict[str, object]:
        return {
            "xr_mode": self.xr_mode,
            "control_role": self.control_role,
            "tags": list(self.tags),
        }


@dataclass
class TaskIndexEntry:
    """Single RLDS episode reference within the task library index."""

    episode_id: str
    task_name: str
    category: TaskCategory
    recency: datetime
    quality_flags: List[str] = field(default_factory=list)
    is_golden: bool = False
    eligible_for_autonomy: bool = False
    metadata_path: Path = field(default_factory=Path)
    steps_paths: List[Path] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "category": self.category.as_dict(),
            "recency": self.recency.isoformat(),
            "quality_flags": list(self.quality_flags),
            "is_golden": self.is_golden,
            "eligible_for_autonomy": self.eligible_for_autonomy,
            "metadata_path": str(self.metadata_path),
            "steps_paths": [str(path) for path in self.steps_paths],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TaskIndexEntry":
        recency_str = data.get("recency")
        if not isinstance(recency_str, str):
            raise ValueError("Index entries must include a recency timestamp.")
        recency = datetime.fromisoformat(recency_str)
        category_data = data.get("category", {})
        category = TaskCategory(
            xr_mode=str(category_data.get("xr_mode", "unknown")),
            control_role=str(category_data.get("control_role", "unknown")),
            tags=tuple(category_data.get("tags", []) or []),
        )
        return cls(
            episode_id=str(data.get("episode_id", "")),
            task_name=str(data.get("task_name", "")),
            category=category,
            recency=recency,
            quality_flags=list(data.get("quality_flags", []) or []),
            is_golden=bool(data.get("is_golden", False)),
            eligible_for_autonomy=bool(data.get("eligible_for_autonomy", False)),
            metadata_path=Path(str(data.get("metadata_path", ""))),
            steps_paths=[Path(path) for path in data.get("steps_paths", []) or []],
        )


class TaskLibrary:
    """Lightweight RLDS episode indexer for local task selection."""

    def __init__(self, episodes_dir: Path, index_path: Optional[Path] = None) -> None:
        self.episodes_dir = Path(episodes_dir)
        self.index_path = Path(index_path) if index_path else self.episodes_dir / "task_index.json"
        self._index: List[TaskIndexEntry] = []
        if self.index_path.exists():
            self._index = self._load_index()

    def build_index(self) -> List[TaskIndexEntry]:
        """Scan episodes on disk and refresh the JSON index."""

        entries: List[TaskIndexEntry] = []
        for metadata_path in self.episodes_dir.rglob("metadata.json"):
            episode_data = json.loads(metadata_path.read_text())
            episode_metadata = episode_data.get("episode_metadata", {})
            continuon_data = episode_metadata.get("continuon", {})
            category = TaskCategory(
                xr_mode=str(continuon_data.get("xr_mode", "unknown")),
                control_role=str(continuon_data.get("control_role", "unknown")),
                tags=tuple(continuon_data.get("tags", []) or []),
            )
            recency = self._parse_recency(episode_metadata, metadata_path)
            quality_flags = list(continuon_data.get("quality_flags", []) or [])
            is_golden = bool(continuon_data.get("is_golden", False))
            eligible = bool(continuon_data.get("eligible_for_autonomy", False))
            steps_dir = metadata_path.parent / "steps"
            steps_paths = sorted(steps_dir.glob("*.jsonl")) if steps_dir.exists() else []
            entries.append(
                TaskIndexEntry(
                    episode_id=str(episode_metadata.get("episode_id", "")),
                    task_name=str(episode_metadata.get("task_name", "")),
                    category=category,
                    recency=recency,
                    quality_flags=quality_flags,
                    is_golden=is_golden,
                    eligible_for_autonomy=eligible,
                    metadata_path=metadata_path,
                    steps_paths=steps_paths,
                )
            )

        self._index = sorted(entries, key=lambda entry: entry.recency, reverse=True)
        self._save_index()
        return self._index

    def list_categories(self) -> List[TaskCategory]:
        """Return a unique list of categories present in the index."""

        categories = {entry.category for entry in self._index}
        return sorted(categories, key=lambda category: (category.xr_mode, category.control_role, "|".join(category.tags)))

    def list_tasks_by_category(
        self,
        xr_mode: Optional[str] = None,
        control_role: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[TaskIndexEntry]:
        """Filter tasks by category components."""

        return [
            entry
            for entry in self._index
            if entry.category.matches(xr_mode=xr_mode, control_role=control_role, tag=tag)
        ]

    def latest_variant(
        self,
        task_name: str,
        xr_mode: Optional[str] = None,
        control_role: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Optional[TaskIndexEntry]:
        """Return the newest episode for a task within the given category filter."""

        candidates = self.list_tasks_by_category(xr_mode, control_role, tag)
        filtered = [entry for entry in candidates if entry.task_name == task_name]
        if not filtered:
            return None
        return sorted(filtered, key=lambda entry: entry.recency, reverse=True)[0]

    def golden_variant(
        self,
        task_name: str,
        xr_mode: Optional[str] = None,
        control_role: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Optional[TaskIndexEntry]:
        """Return the preferred golden episode variant for a task."""

        for entry in self.list_tasks_by_category(xr_mode, control_role, tag):
            if entry.task_name == task_name and entry.is_golden:
                return entry
        return None

    def set_autonomy_flag(self, episode_id: str, eligible: bool) -> bool:
        """Mark an episode as eligible for autonomous selection and persist the change."""

        updated = False
        for entry in self._index:
            if entry.episode_id == episode_id:
                entry.eligible_for_autonomy = eligible
                updated = True
        if updated:
            self._save_index()
        return updated

    def _parse_recency(self, episode_metadata: Dict[str, object], metadata_path: Path) -> datetime:
        recorded_at = episode_metadata.get("recorded_at")
        if isinstance(recorded_at, str):
            cleaned = recorded_at.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(cleaned)
            except ValueError:
                pass
        return datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc)

    def _save_index(self) -> None:
        serialized = [entry.as_dict() for entry in self._index]
        self.index_path.write_text(json.dumps(serialized, indent=2))

    def _load_index(self) -> List[TaskIndexEntry]:
        raw_entries = json.loads(self.index_path.read_text())
        return [TaskIndexEntry.from_dict(entry) for entry in raw_entries]

    @property
    def index(self) -> Iterable[TaskIndexEntry]:
        return tuple(self._index)
