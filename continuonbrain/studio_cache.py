"""Caching utilities for the Continuon Brain Studio server.

The Studio server surfaces Slow-loop artifacts to the frontend panels.
This module keeps an offline-friendly cache with per-artifact TTLs and
privacy-aware redaction to avoid persisting sensitive traces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

SnapshotFetcher = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class ArtifactPolicy:
    """Defines TTL and redaction strategy for an artifact type."""

    name: str
    ttl: timedelta
    redacted_fields: tuple[str, ...] = ()


@dataclass
class CachedArtifact:
    """Represents a cached artifact and its metadata."""

    payload: Any
    cached_at: datetime
    ttl: timedelta
    privacy_redactions: tuple[str, ...]

    def expires_at(self) -> datetime:
        return self.cached_at + self.ttl

    def is_stale(self, now: datetime) -> bool:
        return now >= self.expires_at()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SlowLoopCache:
    """Cache manager for Slow-loop artifacts used by the Studio server."""

    _DEFAULT_POLICIES: tuple[ArtifactPolicy, ...] = (
        ArtifactPolicy(
            name="policy_bundle_ids",
            ttl=timedelta(hours=24),
            redacted_fields=(),
        ),
        ArtifactPolicy(
            name="memory_plane_merge_state",
            ttl=timedelta(hours=6),
            redacted_fields=("spans", "utterances", "raw_payload"),
        ),
        ArtifactPolicy(
            name="drift_alerts",
            ttl=timedelta(hours=1),
            redacted_fields=("raw_trace", "feature_vector", "frames", "raw_frame"),
        ),
    )

    def __init__(
        self,
        config_dir: Path,
        fetcher: Optional[SnapshotFetcher] = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._fetcher = fetcher
        self._clock = clock
        self._policies: Dict[str, ArtifactPolicy] = {p.name: p for p in self._DEFAULT_POLICIES}
        cache_dir = Path(config_dir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_dir / "slow_loop.json"

    def fetch_snapshot(self, allow_network: bool = True) -> Dict[str, Any]:
        """Return Slow-loop artifacts with freshness metadata.

        When ``allow_network`` is ``False``, the cache is returned even if
        stale. When ``True``, stale or missing artifacts are refreshed from
        the cloud fetcher if available, and the cache is updated accordingly.
        """

        now = self._clock()
        cached_entries = self._load_cache()
        needs_refresh = allow_network and any(
            self._needs_refresh(cached_entries.get(name), policy, now)
            for name, policy in self._policies.items()
        )

        cloud_payload: Optional[Mapping[str, Any]] = None
        if needs_refresh and self._fetcher is not None:
            try:
                cloud_payload = self._fetcher()
            except Exception:
                cloud_payload = None

        snapshot: Dict[str, Any] = {}
        updated_cache: Dict[str, CachedArtifact] = {}

        for name, policy in self._policies.items():
            cached = cached_entries.get(name)
            data_from_cloud = cloud_payload.get(name) if cloud_payload else None

            if data_from_cloud is not None:
                sanitized_payload = self._sanitize_payload(data_from_cloud, policy.redacted_fields)
                cached = CachedArtifact(
                    payload=sanitized_payload,
                    cached_at=now,
                    ttl=policy.ttl,
                    privacy_redactions=policy.redacted_fields,
                )
            elif cached is not None:
                # keep cached as is
                pass
            else:
                cached = CachedArtifact(
                    payload=None,
                    cached_at=now,
                    ttl=policy.ttl,
                    privacy_redactions=policy.redacted_fields,
                )

            updated_cache[name] = cached
            snapshot[name] = {
                "data": cached.payload,
                "freshness": {
                    "source": "cloud" if data_from_cloud is not None else "cache",
                    "cached_at": cached.cached_at.isoformat(),
                    "expires_at": cached.expires_at().isoformat(),
                    "stale": cached.is_stale(now),
                    "privacy_redactions": list(cached.privacy_redactions),
                },
            }

        if cloud_payload is not None:
            self._save_cache(updated_cache)

        return snapshot

    def _needs_refresh(
        self, cached: Optional[CachedArtifact], policy: ArtifactPolicy, now: datetime
    ) -> bool:
        if cached is None:
            return True
        return cached.is_stale(now)

    def _load_cache(self) -> Dict[str, CachedArtifact]:
        if not self._cache_path.exists():
            return {}

        try:
            raw = json.loads(self._cache_path.read_text())
        except json.JSONDecodeError:
            return {}

        entries: Dict[str, CachedArtifact] = {}
        for name, payload in raw.items():
            try:
                cached_at = datetime.fromisoformat(payload["cached_at"])
                ttl_seconds = payload.get("ttl_seconds")
                expires_at_raw = payload.get("expires_at")
                if expires_at_raw:
                    expires_at = datetime.fromisoformat(expires_at_raw)
                    ttl = expires_at - cached_at
                elif ttl_seconds:
                    ttl = timedelta(seconds=ttl_seconds)
                else:
                    ttl = self._policies[name].ttl
                entries[name] = CachedArtifact(
                    payload=payload.get("payload"),
                    cached_at=cached_at,
                    ttl=ttl,
                    privacy_redactions=tuple(payload.get("privacy_redactions", ())),
                )
            except Exception:
                continue

        return entries

    def _save_cache(self, entries: Mapping[str, CachedArtifact]) -> None:
        serializable: Dict[str, MutableMapping[str, Any]] = {}
        for name, entry in entries.items():
            serializable[name] = {
                "payload": entry.payload,
                "cached_at": entry.cached_at.isoformat(),
                "ttl_seconds": int(entry.ttl.total_seconds()),
                "privacy_redactions": list(entry.privacy_redactions),
                "expires_at": entry.expires_at().isoformat(),
            }

        self._cache_path.write_text(json.dumps(serializable, indent=2))

    def _sanitize_payload(self, payload: Any, redacted_fields: tuple[str, ...]) -> Any:
        if not redacted_fields:
            return payload

        if isinstance(payload, dict):
            return {
                key: self._sanitize_payload(value, redacted_fields)
                for key, value in payload.items()
                if key not in redacted_fields
            }

        if isinstance(payload, list):
            return [self._sanitize_payload(item, redacted_fields) for item in payload]

        return payload
