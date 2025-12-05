from pathlib import Path

from continuonbrain.task_library import TaskLibrary


def test_builds_index_and_lists_categories(tmp_path):
    episodes_dir = Path("continuonbrain/rlds/episodes")
    index_path = tmp_path / "task_index.json"
    library = TaskLibrary(episodes_dir, index_path=index_path)
    index_entries = library.build_index()

    assert len(index_entries) == 3

    categories = library.list_categories()
    assert any(category.xr_mode == "mixed_reality" and "follow" in category.tags for category in categories)
    assert any(category.xr_mode == "virtual_reality" and category.control_role == "copilot" for category in categories)


def test_lookup_latest_and_golden_variants(tmp_path):
    episodes_dir = Path("continuonbrain/rlds/episodes")
    index_path = tmp_path / "task_index.json"
    library = TaskLibrary(episodes_dir, index_path=index_path)
    library.build_index()

    latest = library.latest_variant("follow_user", xr_mode="mixed_reality", control_role="pilot", tag="follow")
    golden = library.golden_variant("follow_user", xr_mode="mixed_reality", control_role="pilot", tag="follow")

    assert latest is not None
    assert latest.episode_id == "follow_user_mr_v2"
    assert golden is not None
    assert golden.is_golden is True
    assert golden.episode_id == "follow_user_mr_v1"


def test_mark_autonomy_persists(tmp_path):
    episodes_dir = Path("continuonbrain/rlds/episodes")
    index_path = tmp_path / "task_index.json"
    library = TaskLibrary(episodes_dir, index_path=index_path)
    library.build_index()

    updated = library.set_autonomy_flag("follow_user_mr_v1", True)
    assert updated is True

    reloaded = TaskLibrary(episodes_dir, index_path=index_path)
    follow_entry = next(entry for entry in reloaded.index if entry.episode_id == "follow_user_mr_v1")
    assert follow_entry.eligible_for_autonomy is True

    filtered = reloaded.list_tasks_by_category(xr_mode="mixed_reality", control_role="pilot", tag="follow")
    eligible_ids = {entry.episode_id for entry in filtered if entry.eligible_for_autonomy}
    assert "follow_user_mr_v1" in eligible_ids
