import json
from pathlib import Path

import pytest

from continuonbrain.system_instructions import (
    SafetyProtocol,
    SystemInstructions,
    _BASE_SAFETY_RULES,
    _DEFAULT_SYSTEM_INSTRUCTIONS,
)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_safety_protocol_merges_defaults_with_user_rules(config_dir: Path) -> None:
    config_dir.joinpath("safety").mkdir(parents=True)
    (config_dir / "safety" / "protocol.json").write_text(
        json.dumps(["Maintain line of sight", "Respect local laws and social/legal norms."])
    )

    protocol = SafetyProtocol.load(config_dir)

    assert protocol.rules == _BASE_SAFETY_RULES + ("Maintain line of sight",)


def test_safety_protocol_dict_payload_filters_and_ignores_override(config_dir: Path) -> None:
    config_dir.joinpath("safety").mkdir(parents=True)
    (config_dir / "safety" / "protocol.json").write_text(
        json.dumps(
            {
                "override_defaults": True,
                "rules": ["Stay within boundaries", 42, None],
            }
        )
    )

    protocol = SafetyProtocol.load(config_dir)

    assert protocol.rules == _BASE_SAFETY_RULES + ("Stay within boundaries",)


def test_system_instructions_dict_payload_handles_override_and_non_strings(config_dir: Path) -> None:
    (config_dir / "system_instructions.json").write_text(
        json.dumps(
            {
                "override_defaults": True,
                "instructions": ["Check network connectivity", {"invalid": True}],
            }
        )
    )

    instructions = SystemInstructions.load(config_dir)

    assert instructions.instructions == _DEFAULT_SYSTEM_INSTRUCTIONS + (
        "Check network connectivity",
    )


def test_system_instructions_list_payload_with_safety_list(config_dir: Path) -> None:
    config_dir.joinpath("safety").mkdir(parents=True)
    (config_dir / "safety" / "protocol.json").write_text(json.dumps(["Avoid water"], indent=2))
    (config_dir / "system_instructions.json").write_text(
        json.dumps(["Verify battery level"], indent=2)
    )

    instructions = SystemInstructions.load(config_dir)

    assert instructions.instructions == _DEFAULT_SYSTEM_INSTRUCTIONS + ("Verify battery level",)
    assert instructions.safety_protocol.rules == _BASE_SAFETY_RULES + ("Avoid water",)


def test_load_falls_back_on_malformed_json(config_dir: Path) -> None:
    config_dir.joinpath("safety").mkdir(parents=True)
    (config_dir / "safety" / "protocol.json").write_text("{""invalid")
    (config_dir / "system_instructions.json").write_text("{""invalid")

    instructions = SystemInstructions.load(config_dir)

    assert instructions.instructions == _DEFAULT_SYSTEM_INSTRUCTIONS
    assert instructions.safety_protocol.rules == _BASE_SAFETY_RULES
