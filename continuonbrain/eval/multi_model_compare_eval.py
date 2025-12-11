from __future__ import annotations

"""
Compare answers across multiple onboard models (HOPE + fallbacks) and log RLDS.

For each question:
- Ask the primary (HOPE via chat_adapter)
- Ask Gemma 3n-2b (model hint)
- Ask Gemini/Gemma 370m (model hint)
- Choose a "winner" by simple heuristic and write all answers into the RLDS episode.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from continuonbrain.eval.hope_eval_runner import load_questions
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


def _score_answers(answers: Dict[str, str]) -> Tuple[str, str]:
    """
    Pick a winner by:
    - prefer first non-error answer
    - break ties by longest content length
    """
    best_model = None
    best_len = -1
    for model, ans in answers.items():
        if not ans:
            continue
        if ans.startswith("[error"):
            continue
        length = len(ans)
        if best_model is None or length > best_len:
            best_model = model
            best_len = length
    if best_model is None:
        # fall back to first available
        best_model = next(iter(answers.keys()), "unknown")
    rationale = "selected non-error with longest content"
    return best_model, rationale


async def run_compare_eval(
    config_dir: Path,
    rlds_dir: Path,
    questions_path: Path,
    model_hints: List[Tuple[str, Optional[str]]],
) -> Dict[str, Any]:
    service = EvalService(config_dir=config_dir)
    questions = load_questions(questions_path)

    steps: List[Dict[str, Any]] = []
    for item in questions:
        q = item["question"]
        tier = item["tier"]
        answers: Dict[str, str] = {}
        for model_label, hint in model_hints:
            try:
                res = await service.chat_adapter.chat(q, [], model_hint=hint)
                ans = res.get("response") or f"[error:{res.get('error')}]"
            except Exception as exc:  # noqa: BLE001
                ans = f"[error:{type(exc).__name__}] {exc}"
            answers[model_label] = ans

        winner, rationale = _score_answers(answers)
        steps.append(
            {
                "obs": {
                    "question": q,
                    "tier": tier,
                },
                "action": {
                    "answers": answers,
                    "winner": winner,
                    "winner_rationale": rationale,
                },
                "step_metadata": {
                    "models": [m for m, _ in model_hints],
                    "timestamp": time.time(),
                },
            }
        )

    rlds_dir.mkdir(parents=True, exist_ok=True)
    out_path = rlds_dir / f"compare_eval_{int(time.time())}.json"
    out_path.write_text(json.dumps({"steps": steps}, indent=2))

    return {
        "episode_path": str(out_path),
        "steps": len(steps),
        "models": [m for m, _ in model_hints],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-model compare eval (HOPE + Gemma/Gemini)")
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
        "--questions",
        type=Path,
        default=Path(__file__).resolve().parent / "hope_eval_questions.json",
        help="Questions file (tiered JSON) to compare on",
    )
    parser.add_argument(
        "--gemma-hint",
        type=str,
        default="google/gemma-3n-2b",
        help="Model hint for Gemma 3n",
    )
    parser.add_argument(
        "--gemini-hint",
        type=str,
        default="google/gemma-370m",  # using available small model label; adjust if Gemini 370m is exposed
        help="Model hint for Gemini/Gemma 370m",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    models = [
        ("hope", None),
        ("gemma_3n", args.gemma_hint),
        ("gemini_370m", args.gemini_hint),
    ]
    results = asyncio.run(
        run_compare_eval(
            config_dir=args.config_dir,
            rlds_dir=args.rlds_dir,
            questions_path=args.questions,
            model_hints=models,
        )
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
