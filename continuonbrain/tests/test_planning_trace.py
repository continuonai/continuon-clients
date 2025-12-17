from __future__ import annotations

import json
from pathlib import Path


def test_write_planning_episode(tmp_path: Path) -> None:
    from continuonbrain.rlds.planning_trace import write_planning_episode

    ep_dir = write_planning_episode(
        rlds_root=tmp_path,
        environment_id="unit-test",
        control_role="human_supervisor",
        goal={"joint_pos": [0.1, 0, 0, 0, 0, 0]},
        start_state={"joint_pos": [0, 0, 0, 0, 0, 0]},
        plan={
            "ok": True,
            "reason": "ok",
            "diagnostics": {"expanded": 10},
            "steps": [
                {"joint_delta": [0.1, 0, 0, 0, 0, 0], "predicted_joint_pos": [0.1, 0, 0, 0, 0, 0], "score": -0.0, "uncertainty": 0.1},
                {"joint_delta": [0.0, 0, 0, 0, 0, 0], "predicted_joint_pos": [0.1, 0, 0, 0, 0, 0], "score": -0.0, "uncertainty": 0.1},
            ],
        },
        execute={"requested": False, "executed": False, "result": None},
        model_debug={"backend": "stub"},
        tags=["test"],
    )

    assert (ep_dir / "metadata.json").exists()
    assert (ep_dir / "steps" / "000000.jsonl").exists()
    meta = json.loads((ep_dir / "metadata.json").read_text())
    assert meta["environment_id"] == "unit-test"


