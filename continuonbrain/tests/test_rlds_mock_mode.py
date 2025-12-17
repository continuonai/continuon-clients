from pathlib import Path

from continuonbrain.rlds.mock_mode import MockGeneratorConfig, generate_mock_mode_episode
from continuonbrain.rlds.validators import load_episode, validate_episode


def test_mock_mode_episode_passes_validator():
    episode = generate_mock_mode_episode(MockGeneratorConfig(step_count=2))
    result = validate_episode(episode)

    assert result.ok, f"Expected no validation errors, found: {result.errors}"

    metadata = episode["metadata"]
    assert metadata["xr_mode"] == "workstation"
    assert metadata["control_role"] == "human_dev_xr"
    assert metadata["software"]["xr_app"] == "studio-mock"
    diagnostics = episode["steps"][0]["observation"]["diagnostics"]
    assert diagnostics["glove_sample_rate_hz"] == 90.0


def test_validator_flags_missing_metadata_field():
    episode = generate_mock_mode_episode(MockGeneratorConfig(step_count=1))
    episode["metadata"].pop("xr_mode")

    result = validate_episode(episode)

    assert not result.ok
    assert any("metadata.xr_mode" in err for err in result.errors)


def test_fixture_aligns_with_proto_schema():
    fixture_path = Path("continuonbrain/rlds/episodes/studio_mock_editor.json")
    episode = load_episode(fixture_path)

    result = validate_episode(episode)

    assert result.ok, f"Fixture drifted from proto: {result.errors}"
