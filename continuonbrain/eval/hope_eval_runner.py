from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_questions(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    tiers = payload.get("tiers", [])
    questions: List[Dict[str, Any]] = []
    for tier in tiers:
        name = tier.get("name", "tier")
        for q in tier.get("questions", []):
            questions.append({"tier": name, "question": q})
    return questions


async def run_hope_eval_and_log(
    service: Any,
    questions_path: Path,
    rlds_dir: Path,
    *,
    use_fallback: bool = True,
    fallback_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Ask HOPE a graded set of questions, fallback to on-device LLM if needed,
    and log as an RLDS JSON episode.
    """
    fallback_order = fallback_order or ["gemma-3.7", "gemma-3n-2b"]
    questions = load_questions(questions_path)
    history: List[Dict[str, str]] = []
    steps: List[Dict[str, Any]] = []

    async def ask_once(prompt: str, model_hint: Optional[str] = None) -> str:
        try:
            # chat_adapter.chat(self, message, history)
            return (await service.chat_adapter.chat(prompt, history))["response"]
        except Exception as exc:  # pragma: no cover
            return f"[error:{type(exc).__name__}] {exc}"

    for item in questions:
        q = item["question"]
        tier = item["tier"]
        answer = await ask_once(q)
        used_fallback = False
        fallback_model = None
        if use_fallback and (not answer or answer.startswith("[error")):
            for fb in fallback_order:
                fb_ans = await ask_once(f"{q}\n\n(model hint: {fb})")
                if fb_ans and not fb_ans.startswith("[error"):
                    answer = fb_ans
                    used_fallback = True
                    fallback_model = fb
                    break
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer or ""})
        steps.append(
            {
                "obs": {
                    "question": q,
                    "tier": tier,
                    "history_len": len(history),
                },
                "action": {
                    "answer": answer,
                    "used_fallback": used_fallback,
                    "fallback_model": fallback_model,
                },
                "step_metadata": {
                    "model": "hope-agent",
                    "fallback_order": fallback_order,
                    "timestamp": time.time(),
                },
            }
        )

    rlds_dir.mkdir(parents=True, exist_ok=True)
    out_path = rlds_dir / f"hope_eval_{int(time.time())}.json"
    out_path.write_text(json.dumps({"steps": steps}, indent=2))

    return {
        "episode_path": str(out_path),
        "steps": len(steps),
        "fallback_order": fallback_order,
    }
