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
    episode_prefix: str = "hope_eval",
    model_label: str = "hope-agent",
    retriever: Optional[callable] = None,
    max_ctx_chars: int = 800,
) -> Dict[str, Any]:
    """
    Ask HOPE a graded set of questions, fallback to on-device LLM if needed,
    and log as an RLDS JSON episode.
    """
    # Preferred order: smallest first, then 3n-2b.
    # Note: Hailo is an inference accelerator (vision/core-model), not an LLM model id.
    fallback_order = fallback_order or ["google/gemma-370m", "google/gemma-3n-2b"]
    questions = load_questions(questions_path)
    history: List[Dict[str, str]] = []
    steps: List[Dict[str, Any]] = []

    async def ask_once(prompt: str, model_hint: Optional[str] = None) -> str:
        try:
            # chat_adapter.chat(self, message, history)
            return (await service.chat_adapter.chat(prompt, history, model_hint=model_hint))["response"]
        except Exception as exc:  # pragma: no cover
            return f"[error:{type(exc).__name__}] {exc}"

    for item in questions:
        q = item["question"]
        tier = item["tier"]

        contexts: List[Dict[str, str]] = []
        if retriever:
            try:
                contexts = retriever(q) or []
            except Exception:
                contexts = []

        context_text = ""
        if contexts:
            parts = []
            for ctx in contexts:
                title = ctx.get("title") or ""
                text = ctx.get("text") or ""
                parts.append(f"Title: {title}\n{text[:max_ctx_chars]}")
            context_text = "\n---\n".join(parts)

        prompt = q if not context_text else f"{q}\n\nContext:\n{context_text}"

        answer = await ask_once(prompt)
        used_fallback = False
        fallback_model = None
        if use_fallback and (not answer or answer.startswith("[error")):
            for fb in fallback_order:
                fb_ans = await ask_once(f"{prompt}\n\n(model hint: {fb})", model_hint=fb)
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
                    "model": model_label,
                    "fallback_order": fallback_order,
                    "timestamp": time.time(),
                    "contexts_used": len(contexts),
                },
            }
        )

    rlds_dir.mkdir(parents=True, exist_ok=True)
    out_path = rlds_dir / f"{episode_prefix}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"steps": steps}, indent=2))

    return {
        "episode_path": str(out_path),
        "steps": len(steps),
        "fallback_order": fallback_order,
        "model_label": model_label,
    }
