import json
from pathlib import Path

from continuonbrain.rlds import validators
from continuonbrain.rlds.community_dataset_importer import (
    CommunityDatasetIngestConfig,
    build_episode_payload,
    ingest_grouped_samples,
    write_episode,
)


def _sample(step: int, instruction: str | None = None) -> dict:
    payload = {
        "episode_id": "ep0",
        "timestamp": step * 1_000_000,
        "action": [0.25 * step, -0.1],
    }
    if instruction:
        payload["instruction"] = instruction
    return payload


def test_episode_payload_writes_and_validates(tmp_path: Path):
    cfg = CommunityDatasetIngestConfig(output_dir=tmp_path)
    samples = [_sample(0, instruction="pick up"), _sample(1)]

    payload = build_episode_payload("ep0", samples, cfg)
    report = validators.validate_episode(payload)
    assert report.ok, report.errors

    episode_dir = write_episode(payload, tmp_path, "ep0")
    assert (episode_dir / "metadata.json").exists()

    steps_path = episode_dir / "steps" / "000000.jsonl"
    lines = steps_path.read_text().strip().splitlines()
    assert len(lines) == 2

    first_step = json.loads(lines[0])
    assert first_step["observation"]["video_frame_id"] == first_step["observation"]["depth_frame_id"]
    assert first_step["action"]["command"] == [0.0, -0.1]
    assert first_step["step_metadata"]["source_dataset"] == cfg.dataset_id


def test_ingest_grouped_samples_respects_caps(tmp_path: Path):
    cfg = CommunityDatasetIngestConfig(output_dir=tmp_path, max_episodes=1, max_steps_per_episode=1)
    grouped = {
        "ep0": [_sample(0, instruction="stack boxes"), _sample(1)],
        "ep1": [_sample(0)],
    }

    written = ingest_grouped_samples(grouped, cfg)
    assert len(written) == 1

    metadata = json.loads((written[0] / "metadata.json").read_text())
    assert cfg.origin_tag in metadata["tags"]
    assert metadata["tags"].count(f"hf.split:{cfg.split}") == 1

    steps_path = written[0] / "steps" / "000000.jsonl"
    assert steps_path.exists()
    assert len(steps_path.read_text().strip().splitlines()) == 1
