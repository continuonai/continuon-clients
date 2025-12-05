from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

from continuonbrain.studio_server import StudioSlowLoopClient, StudioSlowLoopServer


def test_offline_rendering_and_reconnect_reconciliation(tmp_path: Path) -> None:
    now = {"value": datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)}

    cloud_payload: Dict[str, object] = {
        "policy_bundle_ids": {"active": ["bundle-a"]},
        "memory_plane_merge_state": {"merged": ["alpha"], "spans": ["strip-me"]},
        "drift_alerts": [{"id": "alert-a", "severity": "medium", "feature_vector": [0.1]}],
    }

    def clock() -> datetime:
        return now["value"]

    def fetcher() -> Dict[str, object]:
        return cloud_payload

    server = StudioSlowLoopServer(config_dir=tmp_path, fetcher=fetcher, clock=clock)
    client = StudioSlowLoopClient()

    online_response = server.slow_loop_snapshot(allow_network=True)
    rendered = client.consume(online_response)
    assert rendered["policy_bundle_ids"]["render_state"] == "live-cloud"
    assert rendered["drift_alerts"]["data"] == [{"id": "alert-a", "severity": "medium"}]
    assert rendered["drift_alerts"]["stale"] is False

    now["value"] = now["value"] + timedelta(hours=25)

    offline_response = server.slow_loop_snapshot(allow_network=False)
    offline_render = client.consume(offline_response)
    assert offline_render["policy_bundle_ids"]["render_state"] == "offline-cache"
    assert offline_render["policy_bundle_ids"]["stale"] is True
    assert offline_render["policy_bundle_ids"]["data"] == {"active": ["bundle-a"]}

    cloud_payload = {
        "policy_bundle_ids": {"active": ["bundle-b"]},
        "memory_plane_merge_state": {"merged": ["alpha", "beta"]},
        "drift_alerts": [{"id": "alert-b", "severity": "low"}],
    }

    now["value"] = now["value"] + timedelta(minutes=10)

    refreshed_response = server.slow_loop_snapshot(allow_network=True)
    refreshed_render = client.consume(refreshed_response)
    assert refreshed_render["policy_bundle_ids"]["render_state"] == "live-cloud"
    assert refreshed_render["policy_bundle_ids"]["data"] == {"active": ["bundle-b"]}
    assert refreshed_render["drift_alerts"]["stale"] is False
