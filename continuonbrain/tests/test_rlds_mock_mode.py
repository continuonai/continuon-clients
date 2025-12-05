from continuonbrain.rlds.mock_mode import MockGeneratorConfig, generate_mock_mode_episode
from continuonbrain.rlds.validators import validate_episode


def test_mock_mode_episode_passes_validator():
    episode = generate_mock_mode_episode(MockGeneratorConfig(step_count=2))
    result = validate_episode(episode)

    assert result.ok, f"Expected no validation errors, found: {result.errors}"

    step = episode["steps"][0]
    diagnostics = step["observation"]["diagnostics"]
    assert diagnostics["mock_mode"] is True
    assert diagnostics["glove_sample_rate_hz"] == 90.0


def test_validator_flags_missing_diagnostics():
    episode = generate_mock_mode_episode(MockGeneratorConfig(step_count=1))
    episode["steps"][0]["observation"]["diagnostics"].pop("mock_mode")

    result = validate_episode(episode)

    assert not result.ok
    assert any("diagnostics.mock_mode" in err for err in result.errors)
