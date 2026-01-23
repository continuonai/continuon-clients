"""
Filesystem Memory Manager.

Implements the three-level memory hierarchy with filesystem persistence.
Maps to CMS levels: Procedural (L2) > Semantic (L1) > Episodic (L0).
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import re

logger = logging.getLogger(__name__)


class MemoryLevel(Enum):
    """Memory hierarchy levels (CMS mapping)."""
    PROCEDURAL = "procedural"  # L2: Stable, rarely changes (decay=0.999)
    SEMANTIC = "semantic"      # L1: Accumulated knowledge (decay=0.99)
    EPISODIC = "episodic"      # L0: Session events (decay=0.9)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    key: str
    content: Union[str, dict]
    level: MemoryLevel
    timestamp: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "content": self.content,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Result from memory search."""
    entry: MemoryEntry
    score: float
    path: str


# Retention policy (simulates CMS decay)
RETENTION_POLICY = {
    MemoryLevel.EPISODIC: {
        "max_age_days": 30,
        "max_files": 100,
        "decay_rate": 0.9,
    },
    MemoryLevel.SEMANTIC: {
        "max_age_days": 365,
        "max_versions": 10,
        "decay_rate": 0.99,
    },
    MemoryLevel.PROCEDURAL: {
        "require_approval": True,
        "no_auto_delete": True,
        "decay_rate": 0.999,
    },
}


class FilesystemMemory:
    """
    Filesystem-based memory system with three-level hierarchy.

    Directory structure:
        data_dir/
        ├── procedural/     # L2: Agents, capabilities, guardrails
        ├── semantic/       # L1: Skills, knowledge, behaviors
        └── episodic/       # L0: Conversations, events, episodes
    """

    def __init__(self, data_dir: str = "./brain_b_data"):
        self.data_dir = Path(data_dir)
        self._init_directories()

    def _init_directories(self) -> None:
        """Initialize memory directory structure."""
        for level in MemoryLevel:
            level_dir = self.data_dir / level.value
            level_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        subdirs = {
            "procedural": ["skills", "agents"],
            "semantic": ["knowledge", "behaviors", "skills"],
            "episodic": ["conversations", "events", "rlds_episodes", "checkpoints"],
        }

        for level, dirs in subdirs.items():
            for subdir in dirs:
                (self.data_dir / level / subdir).mkdir(parents=True, exist_ok=True)

    # === Read Operations ===

    def read(self, level: MemoryLevel, path: str) -> Optional[MemoryEntry]:
        """Read a specific memory entry."""
        full_path = self.data_dir / level.value / path

        if not full_path.exists():
            return None

        content = self._read_file(full_path)
        if content is None:
            return None

        stat = full_path.stat()
        return MemoryEntry(
            key=path,
            content=content,
            level=level,
            timestamp=stat.st_mtime,
            metadata={"path": str(full_path)},
        )

    def search(
        self,
        query: str,
        levels: Optional[list[MemoryLevel]] = None,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """
        Search memory for matching entries.

        Uses simple keyword matching. Can be extended with
        semantic similarity for more advanced search.
        """
        if levels is None:
            levels = list(MemoryLevel)

        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for level in levels:
            level_dir = self.data_dir / level.value
            for file_path in level_dir.rglob("*"):
                if file_path.is_file():
                    score = self._score_file(file_path, query_lower, query_words)
                    if score > 0:
                        content = self._read_file(file_path)
                        if content is not None:
                            entry = MemoryEntry(
                                key=str(file_path.relative_to(level_dir)),
                                content=content,
                                level=level,
                                timestamp=file_path.stat().st_mtime,
                            )
                            results.append(SearchResult(entry, score, str(file_path)))

        # Sort by score (descending) and return top results
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    def _score_file(self, path: Path, query: str, query_words: set) -> float:
        """Score a file's relevance to query."""
        score = 0.0

        # Check filename
        filename = path.name.lower()
        if query in filename:
            score += 2.0
        for word in query_words:
            if word in filename:
                score += 0.5

        # Check content (for text files)
        if path.suffix in [".md", ".json", ".txt", ".jsonl"]:
            try:
                content = path.read_text()[:5000].lower()
                if query in content:
                    score += 1.0
                for word in query_words:
                    if word in content:
                        score += 0.3
            except Exception as e:
                logger.debug(f"Could not read {path} for scoring: {e}")

        return score

    def _read_file(self, path: Path) -> Optional[Union[str, dict]]:
        """Read file content based on extension."""
        try:
            if path.suffix == ".json":
                with open(path) as f:
                    return json.load(f)
            elif path.suffix == ".jsonl":
                lines = []
                with open(path) as f:
                    for line in f:
                        lines.append(json.loads(line))
                return lines
            else:
                return path.read_text()
        except Exception as e:
            logger.debug(f"Could not read file {path}: {e}")
            return None

    # === Write Operations ===

    def write(
        self,
        level: MemoryLevel,
        path: str,
        content: Union[str, dict],
        metadata: Optional[dict] = None,
    ) -> str:
        """Write content to memory at specified level."""
        full_path = self.data_dir / level.value / path

        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Check procedural write restrictions
        if level == MemoryLevel.PROCEDURAL:
            if not self._check_procedural_approval(path, content):
                raise PermissionError(
                    f"Procedural memory writes require approval: {path}"
                )

        # Write content
        self._write_file(full_path, content)

        # Write metadata if provided
        if metadata:
            meta_path = full_path.with_suffix(full_path.suffix + ".meta")
            self._write_file(meta_path, {
                "timestamp": time.time(),
                "metadata": metadata,
            })

        return str(full_path)

    def append(
        self,
        level: MemoryLevel,
        path: str,
        content: dict,
    ) -> None:
        """Append to a JSONL file (for event logs)."""
        full_path = self.data_dir / level.value / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "a") as f:
            f.write(json.dumps(content) + "\n")

    def _write_file(self, path: Path, content: Union[str, dict]) -> None:
        """Write content to file based on type."""
        if isinstance(content, dict) or isinstance(content, list):
            with open(path, "w") as f:
                json.dump(content, f, indent=2)
        else:
            path.write_text(str(content))

    def _check_procedural_approval(self, path: str, content: Union[str, dict]) -> bool:
        """Check if procedural write is approved."""
        # For now, allow all writes (in production, this would check approval)
        # This is where human-in-the-loop approval would be implemented
        return True

    # === Episodic Memory Helpers ===

    def log_event(self, event: dict) -> None:
        """Log an event to episodic memory."""
        event["timestamp"] = time.time()
        self.append(MemoryLevel.EPISODIC, "events/events.jsonl", event)

    def log_conversation_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log a conversation turn."""
        turn = {
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        self.append(
            MemoryLevel.EPISODIC,
            f"conversations/{session_id}.jsonl",
            turn
        )

    def save_episode(self, episode_id: str, metadata: dict, steps: list[dict]) -> str:
        """Save an RLDS episode."""
        episode_dir = self.data_dir / "episodic" / "rlds_episodes" / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        self.write(
            MemoryLevel.EPISODIC,
            f"rlds_episodes/{episode_id}/metadata.json",
            metadata
        )

        # Save steps
        steps_path = episode_dir / "steps.jsonl"
        with open(steps_path, "w") as f:
            for step in steps:
                f.write(json.dumps(step) + "\n")

        return str(episode_dir)

    # === Semantic Memory Helpers ===

    def update_knowledge(self, key: str, content: str, source: str = "unknown") -> None:
        """Update semantic knowledge."""
        # Add metadata header
        header = f"""---
updated: {time.strftime('%Y-%m-%d %H:%M:%S')}
source: {source}
---

"""
        self.write(
            MemoryLevel.SEMANTIC,
            f"knowledge/{key}.md",
            header + content
        )

    def save_behavior(self, name: str, actions: list[dict]) -> None:
        """Save a learned behavior."""
        behaviors_file = self.data_dir / "semantic" / "behaviors" / "behaviors.json"

        # Load existing behaviors
        behaviors = {}
        if behaviors_file.exists():
            with open(behaviors_file) as f:
                behaviors = json.load(f)

        # Add/update behavior
        behaviors[name] = {
            "actions": actions,
            "created_at": time.time(),
        }

        # Save
        self.write(MemoryLevel.SEMANTIC, "behaviors/behaviors.json", behaviors)

    # === Procedural Memory Helpers ===

    def register_skill(self, name: str, definition: str) -> None:
        """Register a skill definition."""
        self.write(
            MemoryLevel.PROCEDURAL,
            f"skills/{name}/SKILL.md",
            definition
        )

    def get_capabilities(self) -> dict:
        """Get available capabilities."""
        cap_file = self.data_dir / "procedural" / "capabilities.json"
        if cap_file.exists():
            with open(cap_file) as f:
                return json.load(f)
        return {"actions": [], "tools": []}

    # === Memory Stats ===

    def stats(self) -> dict:
        """Get memory statistics."""
        stats = {"levels": {}}

        for level in MemoryLevel:
            level_dir = self.data_dir / level.value
            files = list(level_dir.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            total_size = sum(f.stat().st_size for f in files if f.is_file())

            stats["levels"][level.value] = {
                "file_count": file_count,
                "total_size_bytes": total_size,
                "decay_rate": RETENTION_POLICY[level]["decay_rate"],
            }

        stats["total_files"] = sum(
            s["file_count"] for s in stats["levels"].values()
        )
        stats["total_size_bytes"] = sum(
            s["total_size_bytes"] for s in stats["levels"].values()
        )

        return stats

    def list_level(self, level: MemoryLevel) -> list[str]:
        """List all entries in a level."""
        level_dir = self.data_dir / level.value
        entries = []

        for path in level_dir.rglob("*"):
            if path.is_file() and not path.name.endswith(".meta"):
                entries.append(str(path.relative_to(level_dir)))

        return sorted(entries)
