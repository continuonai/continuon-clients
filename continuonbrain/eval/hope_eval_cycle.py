from __future__ import annotations

"""
Eval cycle that:
- Runs the facts ("google") eval
- Generates follow-up HOPE questions from the facts eval output
- Runs HOPE eval on both the base and generated questions
- Logs all results as RLDS episodes for export/retraining
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from continuonbrain.eval.hope_eval_runner import run_hope_eval_and_log
from continuonbrain.services.chat_adapter import ChatAdapter


async def _status_stub() -> dict:
    """Minimal status provider for ChatAdapter (no hardware context)."""
    return {
        "status": {
            "mode": "eval",
            "hardware_mode": "mock",
            "allow_motion": False,
        }
    }


class EvalService:
    """Lightweight service that only supplies chat_adapter for eval runners."""

    def __init__(self, config_dir: Path, gemma_chat: Optional[object] = None) -> None:
        self.chat_adapter = ChatAdapter(
            config_dir=str(config_dir),
            status_provider=_status_stub,
            gemma_chat=gemma_chat,
        )


def _load_steps(episode_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(episode_path.read_text())
    return payload.get("steps", [])


def _generate_followup_questions(steps: List[Dict[str, Any]], max_questions: int = 20) -> List[str]:
    """Derive follow-up prompts from facts eval Q/A pairs (deterministic templates)."""
    followups: List[str] = []
    for step in steps:
        if len(followups) >= max_questions:
            break
        obs = step.get("obs", {})
        act = step.get("action", {})
        question = obs.get("question")
        answer = act.get("answer")
        if not question:
            continue

        base = question.strip()
        ans_snippet = (answer or "").strip()
        followups.extend(
            [
                f"Deepen this fact: '{base}'. Provide causal links, prerequisites, and downstream effects.",
                f"Generate edge cases or exceptions for: '{base}'.",
                f"Provide a concrete real-world scenario illustrating: '{base}'.",
            ]
        )
        if ans_snippet:
            followups.append(
                f"Stress-test the answer '{ans_snippet}' to the question '{base}' with counterfactuals."
            )

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for q in followups:
        if q not in seen:
            seen.add(q)
            unique.append(q)
        if len(unique) >= max_questions:
            break
    return unique


def _write_questions_file(questions: List[str], out_path: Path) -> None:
    payload = {"tiers": [{"name": "generated_followups", "questions": questions}]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


async def run_cycle(
    config_dir: Path,
    rlds_dir: Path,
    hope_questions: Path,
    facts_questions: Path,
    fallback_order: Optional[List[str]],
    followup_limit: int,
) -> Dict[str, Any]:
    service = EvalService(config_dir=config_dir)

    # 1) Facts eval (acts as "google" seed)
    facts_res = await run_hope_eval_and_log(
        service=service,
        questions_path=facts_questions,
        rlds_dir=rlds_dir,
        use_fallback=True,
        fallback_order=fallback_order,
        episode_prefix="facts_eval",
        model_label="facts-lite",
    )

    # 2) Generate follow-up HOPE questions from facts episode
    facts_episode = Path(facts_res["episode_path"])
    steps = _load_steps(facts_episode)
    followups = _generate_followup_questions(steps, max_questions=followup_limit)
    followup_path = config_dir / "generated_hope_questions.json"
    _write_questions_file(followups, followup_path)

    # 3) Run base HOPE eval
    hope_res = await run_hope_eval_and_log(
        service=service,
        questions_path=hope_questions,
        rlds_dir=rlds_dir,
        use_fallback=True,
        fallback_order=fallback_order,
        episode_prefix="hope_eval",
        model_label="hope-agent",
    )

    # 4) Run generated follow-up HOPE eval
    followup_res = await run_hope_eval_and_log(
        service=service,
        questions_path=followup_path,
        rlds_dir=rlds_dir,
        use_fallback=True,
        fallback_order=fallback_order,
        episode_prefix="hope_eval_followup",
        model_label="hope-agent",
    )

    return {
        "facts_eval": facts_res,
        "hope_eval": hope_res,
        "hope_eval_followup": followup_res,
        "followup_questions_path": str(followup_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run facts eval → generate follow-ups → HOPE evals")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/tmp/continuonbrain_eval"),
        help="Working directory for chat logs and generated question files",
    )
    parser.add_argument(
        "--rlds-dir",
        type=Path,
        default=Path("/opt/continuonos/brain/rlds/episodes"),
        help="Directory to write RLDS episodes",
    )
    parser.add_argument(
        "--hope-questions",
        type=Path,
        default=Path(__file__).resolve().parent / "hope_eval_questions.json",
        help="Path to base HOPE eval questions",
    )
    parser.add_argument(
        "--facts-questions",
        type=Path,
        default=Path(__file__).resolve().parent / "facts_eval_questions.json",
        help="Path to facts (google-style) eval questions",
    )
    parser.add_argument(
        "--fallback-order",
        nargs="*",
        default=["hailo", "google/gemma-370m", "google/gemma-3n-2b"],
        help="Model fallback order for evals",
    )
    parser.add_argument(
        "--followup-limit",
        type=int,
        default=20,
        help="Maximum generated follow-up questions per cycle",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = asyncio.run(
        run_cycle(
            config_dir=args.config_dir,
            rlds_dir=args.rlds_dir,
            hope_questions=args.hope_questions,
            facts_questions=args.facts_questions,
            fallback_order=args.fallback_order,
            followup_limit=args.followup_limit,
        )
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
