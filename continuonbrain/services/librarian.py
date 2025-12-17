"""
Lightweight offline retriever for wiki/episodic shards.

Design goals:
- No heavy deps; numpy is optional and used when available.
- Manifest-driven: expects a manifest.json with shard entries.
- Read-only: caller is responsible for logging retrievals into RLDS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


@dataclass
class ShardSpec:
    id: str
    kind: str  # "wiki" | "episodic" | other read-only sources
    dim: int
    size: int
    embedding_path: Path
    metadata_path: Path
    license: Optional[str] = None
    domain: Optional[str] = None
    language: Optional[str] = None


class Librarian:
    """Minimal cosine-similarity retriever over local shards."""

    def __init__(
        self,
        manifest_path: Path,
        preload: bool = False,
        max_rows_per_shard: Optional[int] = None,
    ):
        self.manifest_path = manifest_path
        self.max_rows_per_shard = max_rows_per_shard
        self._shards: List[ShardSpec] = []
        self._embeddings: Dict[str, Optional["np.ndarray"]] = {}
        self._metadata: Dict[str, Optional[List[Dict[str, object]]]] = {}
        self._load_manifest()
        if preload:
            self._preload_all()

    # ---------------------------- Public API ---------------------------- #
    def available_shards(self) -> List[Dict[str, object]]:
        """Return shard metadata without loading data."""
        return [
            {
                "id": s.id,
                "kind": s.kind,
                "size": s.size,
                "dim": s.dim,
                "license": s.license,
                "domain": s.domain,
                "language": s.language,
                "embedding_path": str(s.embedding_path),
                "metadata_path": str(s.metadata_path),
            }
            for s in self._shards
        ]

    def retrieve(
        self,
        query_embedding: "np.ndarray",
        k: int = 5,
        kinds: Optional[Sequence[str]] = None,
        domains: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        """
        Retrieve top-k entries across shards filtered by kind/domain.

        Returns a list sorted by descending score with keys: score, shard_id, metadata.
        """
        if np is None:
            raise ImportError("numpy is required for retrieval but is not installed.")
        if query_embedding is None:
            return []
        query = self._normalize(query_embedding)
        results: List[Tuple[float, Dict[str, object]]] = []
        for shard in self._shards:
            if kinds and shard.kind not in kinds:
                continue
            if domains and shard.domain not in domains:
                continue
            embeddings = self._get_embeddings(shard.id)
            metas = self._get_metadata(shard.id)
            if embeddings is None or metas is None or embeddings.size == 0:
                continue
            scores = embeddings @ query
            topk_idx = np.argpartition(scores, -min(k, len(scores)))[-min(k, len(scores)) :]
            for idx in topk_idx:
                results.append(
                    (
                        float(scores[idx]),
                        {
                            "score": float(scores[idx]),
                            "shard_id": shard.id,
                            "kind": shard.kind,
                            "metadata": metas[idx] if idx < len(metas) else {},
                        },
                    )
                )
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:k]]

    # --------------------------- Internal helpers --------------------------- #
    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            self._shards = []
            return
        payload = json.loads(self.manifest_path.read_text())
        base_dir = self.manifest_path.parent
        shards = []
        for entry in payload.get("shards", []):
            try:
                shards.append(
                    ShardSpec(
                        id=entry["id"],
                        kind=entry.get("kind", "wiki"),
                        dim=int(entry["dim"]),
                        size=int(entry.get("size", 0)),
                        embedding_path=base_dir / entry["embedding_file"],
                        metadata_path=base_dir / entry["metadata_file"],
                        license=entry.get("license"),
                        domain=entry.get("domain"),
                        language=entry.get("language"),
                    )
                )
            except KeyError:
                # Skip malformed entries; keep manifest load tolerant.
                continue
        self._shards = shards

    def _preload_all(self) -> None:
        for shard in self._shards:
            self._get_embeddings(shard.id)
            self._get_metadata(shard.id)

    def _get_embeddings(self, shard_id: str) -> Optional["np.ndarray"]:
        if shard_id in self._embeddings:
            return self._embeddings[shard_id]
        shard = self._find_shard(shard_id)
        if shard is None or np is None:
            self._embeddings[shard_id] = None
            return None
        if not shard.embedding_path.exists():
            self._embeddings[shard_id] = None
            return None
        data = np.load(shard.embedding_path)
        if data.ndim != 2:
            self._embeddings[shard_id] = None
            return None
        if self.max_rows_per_shard is not None:
            data = data[: self.max_rows_per_shard]
        data = self._normalize_matrix(data)
        self._embeddings[shard_id] = data
        return data

    def _get_metadata(self, shard_id: str) -> Optional[List[Dict[str, object]]]:
        if shard_id in self._metadata:
            return self._metadata[shard_id]
        shard = self._find_shard(shard_id)
        if shard is None or not shard.metadata_path.exists():
            self._metadata[shard_id] = None
            return None
        items: List[Dict[str, object]] = []
        with shard.metadata_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if self.max_rows_per_shard is not None and idx >= self.max_rows_per_shard:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._metadata[shard_id] = items
        return items

    def _find_shard(self, shard_id: str) -> Optional[ShardSpec]:
        for shard in self._shards:
            if shard.id == shard_id:
                return shard
        return None

    @staticmethod
    def _normalize(vec: "np.ndarray") -> "np.ndarray":
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    @classmethod
    def _normalize_matrix(cls, mat: "np.ndarray") -> "np.ndarray":
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms

