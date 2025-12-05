"""Studio server utilities for Slow-loop artifact delivery.

This module bridges ``SlowLoopCache`` into the Studio server surface so
API handlers can persist ``cache/slow_loop.json`` with the TTL and
redaction rules in :mod:`continuonbrain.studio_cache` and return responses
annotated with freshness metadata. A lightweight client helper consumes
those responses to decide whether panels are rendering cached or live data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

from continuonbrain.studio_cache import SlowLoopCache, SnapshotFetcher


@dataclass
class SlowLoopResponse:
    """API response wrapper for Slow-loop artifacts.

    Attributes:
        artifacts: The redacted payload per artifact name.
        freshness: The per-artifact freshness block returned by the cache
            layer, including the cache source and TTL metadata.
    """

    artifacts: Dict[str, Any]
    freshness: Dict[str, Mapping[str, Any]]

    def to_dict(self) -> Dict[str, Mapping[str, Any]]:
        """Expose the response as a JSON-serializable mapping."""

        return {"artifacts": self.artifacts, "freshness": self.freshness}


class StudioSlowLoopServer:
    """Server-facing entrypoint for Slow-loop cache reads."""

    def __init__(
        self,
        config_dir: Path,
        fetcher: Optional[SnapshotFetcher] = None,
        clock: Optional[Callable[[], Any]] = None,
    ) -> None:
        cache_kwargs: MutableMapping[str, Any] = {
            "config_dir": config_dir,
            "fetcher": fetcher,
        }
        if clock is not None:
            cache_kwargs["clock"] = clock
        self._cache = SlowLoopCache(**cache_kwargs)

    def slow_loop_snapshot(self, allow_network: bool = True) -> SlowLoopResponse:
        """Return redacted Slow-loop artifacts with freshness metadata."""

        snapshot = self._cache.fetch_snapshot(allow_network=allow_network)
        return SlowLoopResponse(
            artifacts={name: payload["data"] for name, payload in snapshot.items()},
            freshness={name: payload["freshness"] for name, payload in snapshot.items()},
        )


class StudioSlowLoopClient:
    """Client helper that consumes freshness metadata for rendering decisions."""

    def __init__(self) -> None:
        self.last_rendered: Dict[str, Mapping[str, Any]] = {}

    def consume(self, response: SlowLoopResponse | Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
        """Record how each artifact will be rendered based on freshness."""

        if isinstance(response, SlowLoopResponse):
            artifacts = response.artifacts
            freshness = response.freshness
        else:
            artifacts = response.get("artifacts", {})
            freshness = response.get("freshness", {})

        rendered: Dict[str, Mapping[str, Any]] = {}
        for name, data in artifacts.items():
            freshness_block = freshness.get(name, {})
            render_state = "offline-cache" if freshness_block.get("source") == "cache" else "live-cloud"
            rendered[name] = {
                "data": data,
                "render_state": render_state,
                "stale": bool(freshness_block.get("stale")),
                "expires_at": freshness_block.get("expires_at"),
                "cached_at": freshness_block.get("cached_at"),
            }
        self.last_rendered = rendered
        return rendered
