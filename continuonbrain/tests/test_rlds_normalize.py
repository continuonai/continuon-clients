from __future__ import annotations

import json
from pathlib import Path

from continuonbrain.rlds.normalize import detect_variant, normalize_to_episode_dir
from continuonbrain.rlds.variant_validators import validate_canonical_episode_dir


def test_detect_variant_episode_dir(tmp_path: Path) -> None:
    ep = tmp_path / "ep1"
    (ep / "steps").mkdir(parents=True)
    (ep / "metadata.json").write_text(json.dumps({"xr_mode": "trainer"}), encoding="utf-8")
    (ep / "steps" / "000000.jsonl").write_text(json.dumps({"observation": {}, "action": {}}) + "\n", encoding="utf-8")
    assert detect_variant(ep).kind == "episode_dir"


def test_normalize_single_json_to_episode_dir(tmp_path: Path) -> None:
    src = tmp_path / "hope_eval_123.json"
    src.write_text(
        json.dumps(
            {
                "steps": [
                    {"obs": {"question": "q1"}, "action": {"answer": "a1"}, "step_metadata": {"tier": "t1"}},
                    {"obs": {"question": "q2"}, "action": {"answer": "a2"}, "step_metadata": {"tier": "t2"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    out_root = tmp_path / "out"
    ep_dir = normalize_to_episode_dir(src, output_root=out_root)
    assert (ep_dir / "metadata.json").exists()
    assert (ep_dir / "steps" / "000000.jsonl").exists()
    result = validate_canonical_episode_dir(ep_dir)
    assert result.ok, result.errors


def test_normalize_single_jsonl_to_episode_dir(tmp_path: Path) -> None:
    src = tmp_path / "episode.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps({"observation": {"x": 1}, "action": {"y": 2}, "reward": 0.0, "is_terminal": False}),
                json.dumps({"observation": {"x": 2}, "action": {"y": 3}, "reward": 1.0, "is_terminal": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "out"
    ep_dir = normalize_to_episode_dir(src, output_root=out_root)
    result = validate_canonical_episode_dir(ep_dir)
    assert result.ok, result.errors


