from __future__ import annotations

"""
Wiki curiosity boot: run small, bounded HOPE sessions seeded by an offline Wikipedia corpus.

Goals:
- Be offline-first (prefer JSONL corpus; no-op if absent).
- Stay bounded (small number of topics/questions per boot).
- Log RLDS episodes (`wiki_learn_<ts>.json`) for downstream training.

This is intended to be launched as an optional boot sidecar from `startup_manager.py`.
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from continuonbrain.eval.hope_eval_runner import run_hope_eval_and_log
from continuonbrain.eval.hope_eval_cycle import EvalService
from continuonbrain.eval.wiki_retriever import build_wiki_retriever


def _load_titles_from_jsonl(path: Path, *, max_scan: int = 800) -> List[str]:
    titles: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_scan:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                title = obj.get("title")
                if isinstance(title, str) and title.strip():
                    titles.append(title.strip())
    except Exception:
        return []
    # de-dupe preserving order
    seen = set()
    out: List[str] = []
    for t in titles:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _write_questions(config_dir: Path, *, titles: List[str], topics: int) -> Path:
    random.shuffle(titles)
    chosen = titles[: max(1, topics)]
    questions = []
    for title in chosen:
        # Prompts are short; context is supplied via retriever.
        questions.append(f"Wikipedia: {title}. Give a concise summary (5 bullets).")
        questions.append(f"Wikipedia: {title}. List 8 key facts with dates/numbers when possible.")
        questions.append(f"Wikipedia: {title}. Create 5 flashcards (Q/A) to test understanding.")
    payload = {"tiers": [{"name": "wiki_curiosity", "questions": questions}]}
    out_path = config_dir / "generated_wiki_questions.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


async def run_wiki_curiosity(
    *,
    config_dir: Path,
    rlds_dir: Path,
    wiki_jsonl: Path,
    episodes: int,
    topics_per_episode: int,
    max_scan: int,
    top_k: int,
    max_chars: int,
    sleep_s: float,
    fallback_order: Optional[List[str]],
) -> Dict[str, Any]:
    titles = _load_titles_from_jsonl(wiki_jsonl, max_scan=max_scan)
    if not titles:
        return {"status": "noop", "reason": "no_titles", "wiki_jsonl": str(wiki_jsonl)}

    retriever = build_wiki_retriever(
        dataset_path=wiki_jsonl,
        max_scan=max_scan,
        top_k=top_k,
        max_chars=max_chars,
    )
    service = EvalService(config_dir=config_dir)
    rlds_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for _ in range(max(1, episodes)):
        questions_path = _write_questions(config_dir, titles=titles, topics=topics_per_episode)
        res = await run_hope_eval_and_log(
            service=service,
            questions_path=questions_path,
            rlds_dir=rlds_dir,
            use_fallback=True,
            fallback_order=fallback_order,
            episode_prefix="wiki_learn",
            model_label="hope-agent",
            retriever=retriever,
        )
        results.append(res)
        if sleep_s:
            await asyncio.sleep(sleep_s)
    return {"status": "ok", "episodes": results, "wiki_jsonl": str(wiki_jsonl)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded offline Wikipedia curiosity sessions and log RLDS.")
    parser.add_argument("--config-dir", default=os.environ.get("CONFIG_DIR", "/opt/continuonos/brain"))
    parser.add_argument("--rlds-dir", default=None, help="Defaults to <config-dir>/rlds/episodes")
    parser.add_argument("--wiki-jsonl", default=os.environ.get("CONTINUON_WIKI_JSONL", ""))
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("CONTINUON_WIKI_BOOT_EPISODES", "1")))
    parser.add_argument("--topics-per-episode", type=int, default=int(os.environ.get("CONTINUON_WIKI_TOPICS_PER_EPISODE", "2")))
    parser.add_argument("--max-scan", type=int, default=int(os.environ.get("CONTINUON_WIKI_MAX_SCAN", "800")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("CONTINUON_WIKI_TOP_K", "2")))
    parser.add_argument("--max-chars", type=int, default=int(os.environ.get("CONTINUON_WIKI_MAX_CHARS", "800")))
    parser.add_argument("--sleep-s", type=float, default=float(os.environ.get("CONTINUON_WIKI_BOOT_SLEEP_S", "1.0")))
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    rlds_dir = Path(args.rlds_dir) if args.rlds_dir else (config_dir / "rlds" / "episodes")
    wiki_jsonl = Path(args.wiki_jsonl) if args.wiki_jsonl else None

    if not wiki_jsonl or not wiki_jsonl.exists():
        print("wiki_curiosity_boot: no-op (set CONTINUON_WIKI_JSONL to a local JSONL corpus path).")
        return

    fallback_order = os.environ.get("CONTINUON_WIKI_FALLBACK_ORDER")
    fo: Optional[List[str]] = None
    if fallback_order:
        try:
            fo = json.loads(fallback_order)
            if not isinstance(fo, list):
                fo = None
        except Exception:
            fo = None

    out = asyncio.run(
        run_wiki_curiosity(
            config_dir=config_dir,
            rlds_dir=rlds_dir,
            wiki_jsonl=wiki_jsonl,
            episodes=max(1, args.episodes),
            topics_per_episode=max(1, args.topics_per_episode),
            max_scan=max(50, args.max_scan),
            top_k=max(1, args.top_k),
            max_chars=max(200, args.max_chars),
            sleep_s=max(0.0, args.sleep_s),
            fallback_order=fo,
        )
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()


