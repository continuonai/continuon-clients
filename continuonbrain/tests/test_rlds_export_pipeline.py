import json

from continuonbrain.rlds.export_pipeline import (
    PiiAnonymizationConfig,
    anonymize_episode,
    prepare_cloud_export,
)


def _sample_episode():
    return {
        "metadata": {
            "xr_mode": "workstation",
            "control_role": "human_dev_xr",
            "environment_id": "dev-lab",
            "software": {
                "xr_app": "studio-mock",
                "continuonbrain_os": "mock",
                "glove_firmware": "mock",
            },
            "tags": ["user:alice", "continuon.xr_mode:workstation"],
        },
        "steps": [
            {
                "observation": {
                    "headset_pose": {
                        "position": [0.0, 0.0, 0.0],
                        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
                        "valid": True,
                    },
                    "right_hand_pose": {
                        "position": [0.0, 0.0, 0.0],
                        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
                        "valid": True,
                    },
                    "left_hand_pose": {
                        "position": [0.0, 0.0, 0.0],
                        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
                        "valid": True,
                    },
                    "gaze": {
                        "origin": [0.0, 0.0, 0.0],
                        "direction": [0.0, 0.0, 1.0],
                        "confidence": 0.9,
                    },
                    "robot_state": {
                        "timestamp_nanos": 1,
                        "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "end_effector_pose": {
                            "position": [0.0, 0.0, 0.0],
                            "orientation_quat": [0.0, 0.0, 0.0, 1.0],
                            "valid": True,
                        },
                        "gripper_open": True,
                        "frame_id": "frame-0",
                        "joint_velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "joint_efforts": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "end_effector_twist": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "wall_time_millis": 1,
                    },
                    "glove": {
                        "timestamp_nanos": 1,
                        "flex": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "fsr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
                        "accel": [0.0, 0.0, 0.0],
                        "valid": True,
                    },
                    "video_frame_id": "frame-0",
                    "depth_frame_id": "frame-0",
                    "diagnostics": {
                        "latency_ms": 1.0,
                        "glove_drops": 0,
                        "ble_rssi": -10,
                        "glove_sample_rate_hz": 90.0,
                    },
                    "audio": {
                        "uri": "file:///Users/alice/mock.wav",
                        "sample_rate_hz": 16000,
                        "num_channels": 1,
                        "format": "pcm16le",
                        "frame_id": "frame-0",
                    },
                    "ui_context": {
                        "active_panel": "Alice Panel",
                        "layout": {"pane": "left"},
                        "focus_context": {"user": "alice"},
                    },
                },
                "action": {
                    "command": [0.0],
                    "source": "human_teleop_xr",
                    "annotation": {
                        "kind": "note",
                        "fields": {"note": "Operator note", "session_id": "123"},
                    },
                    "ui_action": {
                        "action_type": "panel_interaction",
                        "context": {"notes": "Remember", "user": "alice"},
                    },
                },
                "is_terminal": False,
                "step_metadata": {"note": "Operator tag", "session_id": "abc-123"},
            }
        ],
    }


def test_anonymize_episode_hashes_known_pii():
    cfg = PiiAnonymizationConfig(hash_salt="unit-test")
    episode = _sample_episode()

    anonymized = anonymize_episode(episode, cfg)
    tags = anonymized["metadata"]["tags"]

    assert any(tag.startswith("tag:") for tag in tags)
    assert any(tag.startswith(cfg.anonymization_tag_prefix) for tag in tags)

    step = anonymized["steps"][0]
    assert "session_id" not in step["step_metadata"]
    assert step["step_metadata"]["note"] != "Operator tag"
    assert "session_id" not in step["action"]["annotation"]["fields"]
    assert step["action"]["annotation"]["fields"]["note"] != "Operator note"

    audio_uri = step["observation"]["audio"]["uri"]
    assert audio_uri.startswith("anonymized://")
    assert step["observation"]["ui_context"]["active_panel"] != "Alice Panel"



def test_prepare_cloud_export_bundles_outputs(tmp_path):
    cfg = PiiAnonymizationConfig(hash_salt="unit-test")
    episode_path = tmp_path / "episode.json"
    episode_path.write_text(json.dumps(_sample_episode()), encoding="utf-8")

    output_dir = tmp_path / "bundle"
    manifest = prepare_cloud_export([episode_path], output_dir, config=cfg)

    anonymized_path = output_dir / "episodes" / "episode.json"
    report_path = output_dir / "reports" / "episode.validation.json"

    assert anonymized_path.exists()
    assert report_path.exists()
    assert manifest.episodes[0].anonymized_path == str(anonymized_path)
    assert manifest.episodes[0].validation_report == str(report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
