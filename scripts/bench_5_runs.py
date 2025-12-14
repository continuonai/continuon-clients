#!/usr/bin/env python3
"""Local benchmark harness for Continuon Brain runtime/training/evals.

Runs N iterations of:
- quick system health
- RLDS JSON -> TFRecord conversion
- JAX local sanity training (CoreModel)
- RLDS export/anonymize + validate
- HOPE eval cycle (facts + hope + followups) using ChatAdapter

Designed to be run under repo .venv via ./cb.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    # scripts/bench_5_runs.py -> repo root
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _detect_chat_fallback(answer: str) -> bool:
    # Mirror ChatAdapter fallback heuristics (best-effort).
    if not answer:
        return True
    if answer.startswith("[model="):
        return True
    if "Status snapshot" in answer or "Robot status:" in answer or "Ready for XR" in answer:
        return True
    # Very short answers are suspicious (often fallback).
    if len(answer.strip()) < 80:
        return True
    return False


def _run_proof_of_learning(run_dir: Path, duration_sec: float = 5.0) -> Dict[str, Any]:
    """
    Run prove_learning_capability.py and parse its proof artifact.
    """
    proof_out = run_dir / "proof_of_learning.json"
    t0 = time.time()
    cmd = [
        sys.executable,
        str(_repo_root() / "prove_learning_capability.py"),
        "--config-dir",
        str(run_dir / "proof_config"),
        "--duration-sec",
        str(duration_sec),
        "--output",
        str(proof_out),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    wall = time.time() - t0

    proof: Dict[str, Any] = {}
    if proof_out.exists():
        with contextlib.suppress(Exception):
            loaded = _read_json(proof_out)
            if isinstance(loaded, dict):
                proof = loaded

    summary = proof.get("summary") if isinstance(proof.get("summary"), dict) else {}
    verdict = str(proof.get("verdict") or "UNKNOWN")
    total_updates = int((summary or {}).get("total_updates") or 0)
    max_delta = float((summary or {}).get("max_param_delta") or 0.0)

    # Save combined stdout for debugging/auditing.
    (run_dir / "proof_of_learning.stdout.txt").write_text(proc.stdout[-20000:], encoding="utf-8")

    return {
        "verdict": verdict,
        "ok": verdict == "SUCCESS",
        "total_updates": total_updates,
        "max_param_delta": max_delta,
        "wall_time_s": wall,
        "exit_code": int(proc.returncode),
        "output_path": str(proof_out),
    }


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extract a JSON object from a response (may include code fences)."""
    if not text:
        return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        return None
    blob = m.group(1)
    try:
        parsed = json.loads(blob)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _score_bench_eval(episode_path: Path) -> Dict[str, Any]:
    """Score a benchmark eval episode (heuristics + fallback gate)."""
    payload = _read_json(episode_path)
    steps = payload.get("steps", [])

    total = len(steps)
    fallback = 0
    errors = 0

    per_q: List[Dict[str, Any]] = []
    raw_score = 0

    for s in steps:
        q = ((s.get("obs") or {}).get("question") or "").strip()
        a = ((s.get("action") or {}).get("answer") or "")
        if not isinstance(a, str):
            a = str(a)
        is_err = a.startswith("[error")
        is_fb = _detect_chat_fallback(a)
        if is_err:
            errors += 1
        if is_fb:
            fallback += 1

        q_score = 0
        q_max = 0
        notes: List[str] = []

        if "API endpoints to trigger training and eval" in q:
            q_max = 40
            needed = ["/api/training/metrics", "/api/training/eval_summary", "/api/training/data_quality"]
            bonus = ["/api/training/hope_eval", "/api/training/wavecore_loops"]
            hit = sum(1 for n in needed if n in a)
            q_score += int((hit / len(needed)) * 30)
            if hit == len(needed):
                q_score += 5
            b = sum(1 for n in bonus if n in a)
            q_score += min(5, b * 3)
            notes.append(f"needed_hit={hit}/{len(needed)} bonus_hit={b}/{len(bonus)}")

        elif "Provide a concise JSON example of a RLDS step" in q:
            q_max = 35
            obj = _extract_json_object(a)
            if obj is None:
                notes.append("no_json_object_parsed")
            else:
                has_action = "action" in obj
                has_obs = ("obs" in obj) or ("observation" in obj)
                if has_action:
                    q_score += 15
                if has_obs:
                    q_score += 10
                if ("reward" in obj) or ("done" in obj) or ("is_terminal" in obj):
                    q_score += 5
                action_obj = obj.get("action") if isinstance(obj.get("action"), dict) else {}
                if isinstance(action_obj, dict):
                    cmd = action_obj.get("command") if isinstance(action_obj.get("command"), dict) else action_obj
                    if isinstance(cmd, dict) and ("steering" in cmd or "steer" in cmd) and ("throttle" in cmd or "speed" in cmd):
                        q_score += 5
                notes.append(f"parsed_keys={sorted(list(obj.keys()))[:8]}")

        elif "minimal Python function to validate an RLDS step dict" in q:
            q_max = 25
            has_def = bool(re.search(r"\bdef\s+\w+\s*\(.*\)\s*:", a))
            returns_tuple = bool(re.search(r"return\s+\(?\s*\w+\s*,\s*\w+\s*\)?", a))
            mentions_keys = all(k in a for k in ["obs", "action", "reward", "done"])
            if has_def:
                q_score += 10
            if returns_tuple:
                q_score += 8
            if mentions_keys:
                q_score += 7
            notes.append(f"has_def={has_def} returns_tuple={returns_tuple} mentions_keys={mentions_keys}")

        per_q.append(
            {
                "question": q,
                "fallback": is_fb,
                "error": is_err,
                "score": q_score,
                "max": q_max,
                "notes": notes,
            }
        )
        raw_score += q_score

    fallback_rate = (fallback / total) if total else None
    valid = (fallback == 0 and errors == 0)
    score = raw_score if valid else 0
    passed = bool(valid and score >= 70)

    return {
        "episode_path": str(episode_path),
        "questions": total,
        "fallback": fallback,
        "errors": errors,
        "fallback_rate": fallback_rate,
        "valid": valid,
        "score": int(score),
        "passed": passed,
        "per_question": per_q,
    }


def _summarize_numbers(xs: List[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "max": None}
    if len(xs) == 1:
        v = float(xs[0])
        return {"n": 1, "mean": v, "stdev": 0.0, "min": v, "max": v}
    return {
        "n": len(xs),
        "mean": float(statistics.mean(xs)),
        "stdev": float(statistics.stdev(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


@dataclass
class RunMetrics:
    run_id: int
    wall_time_s: float

    health_overall: Optional[str]
    cpu_temp_c: Optional[float]
    mem_available_mb: Optional[float]
    disk_available_gb: Optional[float]

    episodes_found: int

    tfrecord_files: int
    tfrecord_records: int
    tfrecord_convert_s: float

    train_steps: int
    train_final_loss: Optional[float]
    train_avg_loss: Optional[float]
    train_wall_time_s: Optional[float]

    proof_verdict: Optional[str]
    proof_total_updates: Optional[int]
    proof_max_param_delta: Optional[float]
    proof_wall_time_s: Optional[float]

    export_episode_count: int

    eval_questions_total: int
    eval_error_answers: int
    eval_fallback_answers: int
    eval_fallback_rate: Optional[float]
    eval_avg_answer_chars: Optional[float]
    eval_wall_time_s: float

    scored_eval_valid: bool
    scored_eval_score: int
    scored_eval_passed: bool


def _run_health(run_dir: Path) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
    from continuonbrain.system_health import SystemHealthChecker

    checker = SystemHealthChecker()
    status, results = checker.run_all_checks(quick_mode=True)

    # Persist report into run_dir (not /tmp) for the benchmark bundle.
    report_path = run_dir / "health_report.json"
    try:
        checker.save_report(str(report_path))
    except Exception:
        # Best-effort; still proceed.
        pass

    overall = getattr(status, "value", None) if status is not None else None

    # Extract a few key numeric signals if present.
    cpu_temp = None
    mem_avail_mb = None
    disk_avail_gb = None
    for r in results or []:
        if getattr(r, "component", "") == "CPU Temperature":
            # r.message like "CPU temp: 60.6°C"
            m = re.search(r"([0-9]+\.?[0-9]*)", str(getattr(r, "message", "")))
            if m:
                cpu_temp = float(m.group(1))
        if getattr(r, "component", "") == "Memory":
            m = re.search(r"([0-9]+\.?[0-9]*)", str(getattr(r, "message", "")))
            if m:
                mem_avail_mb = float(m.group(1))
        if getattr(r, "component", "") == "Disk Space":
            m = re.search(r"([0-9]+\.?[0-9]*)", str(getattr(r, "message", "")))
            if m:
                disk_avail_gb = float(m.group(1))

    return overall, cpu_temp, mem_avail_mb, disk_avail_gb


def _count_episodes(episodes_dir: Path) -> int:
    episode_json = list(episodes_dir.glob("**/episode.json"))
    if episode_json:
        return len(episode_json)
    return len([p for p in episodes_dir.glob("*.json") if p.is_file()])


def _convert_tfrecords(episodes_dir: Path, tfrecord_dir: Path) -> Tuple[int, int, float]:
    from continuonbrain.jax_models.data.tfrecord_converter import convert_directory_to_tfrecord

    start = time.time()
    outputs = convert_directory_to_tfrecord(episodes_dir, output_dir=tfrecord_dir, compress=True)
    elapsed = time.time() - start

    tfrecord_files = len(outputs)

    # Count records.
    record_count = 0
    try:
        import tensorflow as tf

        ds = tf.data.TFRecordDataset([str(p) for p in outputs], compression_type="GZIP")
        record_count = sum(1 for _ in ds)
    except Exception:
        record_count = 0

    return tfrecord_files, record_count, elapsed


def _jax_train(tfrecord_dir: Path, metrics_path: Path) -> Tuple[int, Optional[float], Optional[float], Optional[float]]:
    from continuonbrain.jax_models.train.local_sanity_check import run_sanity_check

    # Keep this small; the goal is repeatable smoke + timing.
    res = run_sanity_check(
        rlds_dir=tfrecord_dir,
        obs_dim=128,
        action_dim=32,
        output_dim=32,
        max_steps=10,
        batch_size=4,
        learning_rate=1e-3,
        use_synthetic_data=False,
        metrics_path=metrics_path,
    )

    return (
        int(res.get("steps") or 0),
        res.get("final_loss"),
        res.get("avg_loss"),
        res.get("wall_time_s"),
    )


def _export_rlds(episodes_dir: Path, export_dir: Path) -> int:
    from continuonbrain.rlds.export_pipeline import prepare_cloud_export

    episode_files = list(episodes_dir.glob("**/episode.json"))
    if not episode_files:
        episode_files = list(episodes_dir.glob("*.json"))

    prepare_cloud_export(sorted(episode_files), export_dir)

    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        return 0
    manifest = _read_json(manifest_path)
    return int(len(manifest.get("episodes", [])))


def _make_small_questions(in_path: Path, out_path: Path, max_total: int) -> int:
    payload = _read_json(in_path)
    tiers = payload.get("tiers", [])

    picked: List[Dict[str, Any]] = []
    remaining = max_total
    for tier in tiers:
        if remaining <= 0:
            break
        name = tier.get("name", "tier")
        qs = list(tier.get("questions", []))
        if not qs:
            continue
        take = min(len(qs), max(1, remaining // max(1, len(tiers))))
        take = min(take, remaining)
        picked.append({"name": name, "questions": qs[:take]})
        remaining -= take

    out = {"tiers": picked}
    _write_json(out_path, out)

    return sum(len(t.get("questions", [])) for t in picked)


async def _eval_cycle(
    config_dir: Path,
    rlds_dir: Path,
    hope_questions: Path,
    facts_questions: Path,
    followup_limit: int,
) -> Dict[str, Any]:
    # Prefer real Gemma if available; fallback otherwise.
    gemma_chat = None
    try:
        from continuonbrain.gemma_chat import build_chat_service

        gemma_chat = build_chat_service()
    except Exception:
        gemma_chat = None

    from continuonbrain.eval.hope_eval_runner import run_hope_eval_and_log
    from continuonbrain.eval.hope_eval_cycle import _generate_followup_questions, _load_steps, _write_questions_file, EvalService

    service = EvalService(config_dir=config_dir, gemma_chat=gemma_chat)

    facts_res = await run_hope_eval_and_log(
        service=service,
        questions_path=facts_questions,
        rlds_dir=rlds_dir,
        use_fallback=True,
        episode_prefix="facts_eval",
        model_label="facts-lite",
    )

    steps = _load_steps(Path(facts_res["episode_path"]))
    followups = _generate_followup_questions(steps, max_questions=followup_limit)
    followup_path = config_dir / "generated_hope_questions.json"
    _write_questions_file(followups, followup_path)

    hope_res = await run_hope_eval_and_log(
        service=service,
        questions_path=hope_questions,
        rlds_dir=rlds_dir,
        use_fallback=True,
        episode_prefix="hope_eval",
        model_label="hope-agent",
    )

    followup_res = await run_hope_eval_and_log(
        service=service,
        questions_path=followup_path,
        rlds_dir=rlds_dir,
        use_fallback=True,
        episode_prefix="hope_eval_followup",
        model_label="hope-agent",
    )

    return {
        "facts_eval": facts_res,
        "hope_eval": hope_res,
        "hope_eval_followup": followup_res,
        "followup_questions_path": str(followup_path),
        "used_real_model": bool(gemma_chat),
    }


def _score_eval_episode(path: Path) -> Tuple[int, int, int, Optional[float]]:
    payload = _read_json(path)
    steps = payload.get("steps", [])
    total = len(steps)
    err = 0
    fb = 0
    lengths: List[int] = []
    for s in steps:
        ans = ((s.get("action") or {}).get("answer") or "")
        if not isinstance(ans, str):
            ans = str(ans)
        lengths.append(len(ans))
        if ans.startswith("[error"):
            err += 1
        if _detect_chat_fallback(ans):
            fb += 1
    avg_len = float(statistics.mean(lengths)) if lengths else None
    return total, err, fb, avg_len


def run_once(run_id: int, base_dir: Path, episodes_dir: Path) -> RunMetrics:
    run_dir = base_dir / f"run_{run_id:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Health
    health_overall, cpu_temp_c, mem_avail_mb, disk_avail_gb = _run_health(run_dir)

    episodes_found = _count_episodes(episodes_dir)

    # TFRecord conversion
    tfrecord_dir = run_dir / "tfrecord"
    tfrecord_files, tfrecord_records, tfrecord_s = _convert_tfrecords(episodes_dir, tfrecord_dir)

    # Train
    metrics_path = run_dir / "jax_sanity.csv"
    train_steps, train_final, train_avg, train_wall = _jax_train(tfrecord_dir, metrics_path)

    # Proof-of-learning (software dynamics + artifact)
    proof = _run_proof_of_learning(run_dir, duration_sec=5.0)

    # Export
    export_dir = run_dir / "export"
    export_episode_count = _export_rlds(episodes_dir, export_dir)

    # Eval cycle (small question sets)
    eval_t0 = time.time()
    eval_config_dir = run_dir / "eval_config"
    eval_rlds_dir = run_dir / "eval_rlds"
    eval_config_dir.mkdir(parents=True, exist_ok=True)
    eval_rlds_dir.mkdir(parents=True, exist_ok=True)

    hope_src = _repo_root() / "continuonbrain" / "eval" / "hope_eval_questions.json"
    facts_src = _repo_root() / "continuonbrain" / "eval" / "facts_eval_questions.json"
    hope_small = eval_config_dir / "hope_questions_small.json"
    facts_small = eval_config_dir / "facts_questions_small.json"

    hope_n = _make_small_questions(hope_src, hope_small, max_total=6)
    facts_n = _make_small_questions(facts_src, facts_small, max_total=3)

    import asyncio

    cycle_res = asyncio.run(
        _eval_cycle(
            config_dir=eval_config_dir,
            rlds_dir=eval_rlds_dir,
            hope_questions=hope_small,
            facts_questions=facts_small,
            followup_limit=6,
        )
    )

    eval_total = 0
    eval_err = 0
    eval_fb = 0
    avg_lens: List[float] = []
    for k in ("facts_eval", "hope_eval", "hope_eval_followup"):
        ep = Path(cycle_res[k]["episode_path"])
        t, e, f, a = _score_eval_episode(ep)
        eval_total += t
        eval_err += e
        eval_fb += f
        if a is not None:
            avg_lens.append(a)

    eval_fb_rate = (eval_fb / eval_total) if eval_total else None
    eval_avg_chars = float(statistics.mean(avg_lens)) if avg_lens else None
    eval_wall = time.time() - eval_t0

    # Persist cycle result for inspection.
    _write_json(run_dir / "eval_cycle_result.json", cycle_res)

    # Scored benchmark eval (machine-checkable + fallback-gated)
    bench_questions = {
        "tiers": [
            {
                "name": "bench_scored",
                "questions": [
                    "List the API endpoints to trigger training and eval from the web UI.",
                    "Provide a concise JSON example of a RLDS step for a drive command with steering=0.2, throttle=0.4, emergency_stop=false.",
                    "Write a minimal Python function to validate an RLDS step dict has obs/action/reward/done keys; return a tuple (ok, errors).",
                ],
            }
        ]
    }
    bench_q_path = eval_config_dir / "bench_scored_questions.json"
    _write_json(bench_q_path, bench_questions)

    from continuonbrain.eval.hope_eval_cycle import EvalService
    from continuonbrain.eval.hope_eval_runner import run_hope_eval_and_log

    bench_episode = asyncio.run(
        run_hope_eval_and_log(
            service=EvalService(config_dir=eval_config_dir),
            questions_path=bench_q_path,
            rlds_dir=eval_rlds_dir,
            use_fallback=True,
            episode_prefix="bench_scored_eval",
            model_label="bench-scored",
        )
    )
    bench_score = _score_bench_eval(Path(bench_episode["episode_path"]))
    _write_json(run_dir / "bench_scored_eval_result.json", bench_score)

    wall = time.time() - t0

    return RunMetrics(
        run_id=run_id,
        wall_time_s=wall,
        health_overall=health_overall,
        cpu_temp_c=cpu_temp_c,
        mem_available_mb=mem_avail_mb,
        disk_available_gb=disk_avail_gb,
        episodes_found=episodes_found,
        tfrecord_files=tfrecord_files,
        tfrecord_records=tfrecord_records,
        tfrecord_convert_s=tfrecord_s,
        train_steps=train_steps,
        train_final_loss=train_final,
        train_avg_loss=train_avg,
        train_wall_time_s=train_wall,
        proof_verdict=str(proof.get("verdict")) if proof else None,
        proof_total_updates=int(proof.get("total_updates") or 0) if proof else None,
        proof_max_param_delta=float(proof.get("max_param_delta") or 0.0) if proof else None,
        proof_wall_time_s=float(proof.get("wall_time_s") or 0.0) if proof else None,
        export_episode_count=export_episode_count,
        eval_questions_total=eval_total,
        eval_error_answers=eval_err,
        eval_fallback_answers=eval_fb,
        eval_fallback_rate=eval_fb_rate,
        eval_avg_answer_chars=eval_avg_chars,
        eval_wall_time_s=eval_wall,
        scored_eval_valid=bool(bench_score.get("valid")),
        scored_eval_score=int(bench_score.get("score") or 0),
        scored_eval_passed=bool(bench_score.get("passed")),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=_repo_root() / "continuonbrain" / "rlds" / "episodes",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp") / f"continuon_bench_{_now_ts()}")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes_dir: Path = args.episodes_dir

    runs: List[RunMetrics] = []
    for i in range(1, args.runs + 1):
        runs.append(run_once(i, out_dir, episodes_dir))

    report = {
        "generated_at": time.time(),
        "repo_root": str(_repo_root()),
        "python_executable": sys.executable,
        "episodes_dir": str(episodes_dir),
        "expectations": {
            "phase1_kpis": {
                "mode_a_valid_rlds_target": 0.95,
                "glove_stream_hz_target": 100,
                "grpc_webrtc_link_success_target": 1.0,
            },
            "pi_training_gates": {
                "min_episodes": 16,
                "control_loop_sync_ms_lte": 5,
                "no_critical_resource_alerts": True,
                "loss_finite_and_decreasing": True,
            },
            "eval_gates": {
                "scored_eval_requires_no_fallback": True,
                "scored_eval_pass_score_gte": 70,
            },
        },
        "runs": [asdict(r) for r in runs],
        "summary": {
            "wall_time_s": _summarize_numbers([r.wall_time_s for r in runs]),
            "tfrecord_convert_s": _summarize_numbers([r.tfrecord_convert_s for r in runs]),
            "train_wall_time_s": _summarize_numbers([float(r.train_wall_time_s or 0.0) for r in runs]),
            "train_avg_loss": _summarize_numbers([float(r.train_avg_loss or 0.0) for r in runs if r.train_avg_loss is not None]),
            "eval_fallback_rate": _summarize_numbers([float(r.eval_fallback_rate or 0.0) for r in runs if r.eval_fallback_rate is not None]),
            "proof_total_updates": _summarize_numbers([float(r.proof_total_updates or 0.0) for r in runs if r.proof_total_updates is not None]),
            "proof_max_param_delta": _summarize_numbers([float(r.proof_max_param_delta or 0.0) for r in runs if r.proof_max_param_delta is not None]),
            "scored_eval_score": _summarize_numbers([float(r.scored_eval_score) for r in runs]),
            "scored_eval_pass_rate": _summarize_numbers([1.0 if r.scored_eval_passed else 0.0 for r in runs]),
            "cpu_temp_c": _summarize_numbers([float(r.cpu_temp_c or 0.0) for r in runs if r.cpu_temp_c is not None]),
            "episodes_found": {
                "min": min(r.episodes_found for r in runs),
                "max": max(r.episodes_found for r in runs),
                "expected_min": 16,
            },
        },
    }

    _write_json(out_dir / "report.json", report)
    print(f"Wrote benchmark report to: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
