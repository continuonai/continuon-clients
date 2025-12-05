import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

from continuonbrain.studio_cache import CachedArtifact, SlowLoopCache


def _fixed_now() -> datetime:
    return datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)


def test_offline_snapshot_uses_cached_even_when_stale(tmp_path: Path) -> None:
    cache = SlowLoopCache(config_dir=tmp_path, fetcher=lambda: {}, clock=_fixed_now)
    stale_cached_at = _fixed_now() - timedelta(days=2)

    cache._save_cache(
        {
            "policy_bundle_ids": CachedArtifact(
                payload={"active": ["bundle-a"]},
                cached_at=stale_cached_at,
                ttl=timedelta(hours=24),
                privacy_redactions=(),
            )
        }
    )

    snapshot = cache.fetch_snapshot(allow_network=False)

    policy_payload = snapshot["policy_bundle_ids"]
    assert policy_payload["data"] == {"active": ["bundle-a"]}
    assert policy_payload["freshness"]["source"] == "cache"
    assert policy_payload["freshness"]["stale"] is True


def test_cloud_refresh_replaces_stale_cache(tmp_path: Path) -> None:
    fixed_now = _fixed_now()
    stale_cached_at = fixed_now - timedelta(days=1)

    def fetcher() -> Dict[str, object]:
        return {
            "policy_bundle_ids": {"active": ["bundle-b"]},
            "memory_plane_merge_state": {"merged": ["alpha", "beta"], "spans": ["trim-me"]},
            "drift_alerts": [
                {"id": "alert-1", "severity": "high", "raw_trace": "secret"},
            ],
        }

    cache = SlowLoopCache(config_dir=tmp_path, fetcher=fetcher, clock=lambda: fixed_now)
    cache._save_cache(
        {
            "policy_bundle_ids": CachedArtifact(
                payload={"active": ["bundle-a"]},
                cached_at=stale_cached_at,
                ttl=timedelta(hours=24),
                privacy_redactions=(),
            )
        }
    )

    snapshot = cache.fetch_snapshot(allow_network=True)

    assert snapshot["policy_bundle_ids"]["data"] == {"active": ["bundle-b"]}
    assert snapshot["policy_bundle_ids"]["freshness"]["source"] == "cloud"
    assert snapshot["policy_bundle_ids"]["freshness"]["stale"] is False

    stored = json.loads((tmp_path / "cache" / "slow_loop.json").read_text())
    assert stored["drift_alerts"]["payload"][0] == {"id": "alert-1", "severity": "high"}
    assert stored["memory_plane_merge_state"]["payload"] == {"merged": ["alpha", "beta"]}


def test_partial_cloud_failure_keeps_cached_entries(tmp_path: Path) -> None:
    fixed_now = _fixed_now()
    drift_cached_at = fixed_now - timedelta(hours=2)

    def failing_fetcher() -> Dict[str, object]:
        raise RuntimeError("cloud unavailable")

    cache = SlowLoopCache(config_dir=tmp_path, fetcher=failing_fetcher, clock=lambda: fixed_now)
    cache._save_cache(
        {
            "drift_alerts": CachedArtifact(
                payload=[{"id": "cached-alert", "severity": "low"}],
                cached_at=drift_cached_at,
                ttl=timedelta(hours=1),
                privacy_redactions=(),
            )
        }
    )

    snapshot = cache.fetch_snapshot(allow_network=True)

    drift_payload = snapshot["drift_alerts"]
    assert drift_payload["data"] == [{"id": "cached-alert", "severity": "low"}]
    assert drift_payload["freshness"]["source"] == "cache"
    assert drift_payload["freshness"]["stale"] is True
