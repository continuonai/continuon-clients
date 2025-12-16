from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from continuonbrain.jax_models.infer.tool_router_infer import load_tool_router_bundle, predict_topk, ToolRouterBundle


@dataclass
class ToolRouterEvalConfig:
    episodes_root: Path = Path("/opt/continuonos/brain/rlds/episodes")
    include_dirs_prefix: str = "toolchat_hf"
    export_dir: Path = Path("/opt/continuonos/brain/model/adapters/candidate/tool_router_seed")
    max_episodes_scan: int = 30000
    eval_mod: int = 10
    eval_bucket: int = 0  # bucket in [0, eval_mod)
    k: int = 5
    out_path: Path = Path("/opt/continuonos/brain/trainer/logs/tool_router_eval_metrics.json")


def _iter_toolchat_episode_files(root: Path, prefix: str, limit: int) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for p in sorted(root.glob(f"{prefix}*")):
        if not p.is_dir():
            continue
        files.extend(sorted(p.glob("*.json")))
        if len(files) >= limit:
            break
    return files[:limit]


def _bucket_text(text: str, mod: int) -> int:
    h = hashlib.md5((text or "").encode("utf-8"), usedforsecurity=False).digest()  # noqa: S324
    return int.from_bytes(h[:4], "little") % max(2, mod)


def _extract_pairs(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        return []
    user_text: Optional[str] = None
    tools: List[str] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        obs = s.get("observation") or {}
        if isinstance(obs, dict) and obs.get("type") == "chat" and obs.get("role") == "user" and user_text is None:
            c = obs.get("content")
            if isinstance(c, str) and c.strip():
                user_text = c.strip()
        action = s.get("action") or {}
        if isinstance(action, dict) and action.get("type") == "tool_call":
            nm = action.get("name")
            if isinstance(nm, str) and nm.strip():
                tools.append(nm.strip())
    if not user_text or not tools:
        return []
    return [(user_text, t) for t in tools]


def evaluate(cfg: ToolRouterEvalConfig) -> Dict[str, Any]:
    bundle = load_tool_router_bundle(cfg.export_dir)
    files = _iter_toolchat_episode_files(cfg.episodes_root, cfg.include_dirs_prefix, cfg.max_episodes_scan)

    total = 0
    top1 = 0
    top5 = 0
    per_tool: Dict[str, Dict[str, int]] = {}

    for fp in files:
        try:
            payload = json.loads(fp.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        pairs = _extract_pairs(payload)
        for text, true_tool in pairs:
            # deterministic holdout split
            if _bucket_text(text, cfg.eval_mod) != cfg.eval_bucket:
                continue
            total += 1
            preds = predict_topk(bundle, text, k=cfg.k)
            pred_tools = [p.get("tool") for p in preds if isinstance(p, dict)]
            if pred_tools:
                if pred_tools[0] == true_tool:
                    top1 += 1
                if true_tool in pred_tools:
                    top5 += 1
            stats = per_tool.setdefault(true_tool, {"total": 0, "top1": 0, "top5": 0})
            stats["total"] += 1
            stats["top1"] += 1 if (pred_tools and pred_tools[0] == true_tool) else 0
            stats["top5"] += 1 if (true_tool in pred_tools) else 0

    now = time.time()
    res = {
        "status": "ok",
        "timestamp": now,
        "examples": total,
        "top1": (top1 / total) if total else None,
        "top5": (top5 / total) if total else None,
        "k": cfg.k,
        "eval_split": {"mod": cfg.eval_mod, "bucket": cfg.eval_bucket},
        "export_dir": str(cfg.export_dir),
        "episodes_root": str(cfg.episodes_root),
        "include_dirs_prefix": cfg.include_dirs_prefix,
    }

    # Append to metrics series for UI charting
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    series: List[Dict[str, Any]] = []
    if cfg.out_path.exists():
        try:
            prev = json.loads(cfg.out_path.read_text())
            if isinstance(prev, list):
                series = prev
        except Exception:
            series = []
    series.append(
        {
            "step": int(now),
            "top1": res["top1"],
            "top5": res["top5"],
            "examples": total,
            "timestamp": now,
        }
    )
    cfg.out_path.write_text(json.dumps(series[-500:], indent=2, default=str))

    # Provide a small breakdown for debugging
    top_tools = sorted(per_tool.items(), key=lambda kv: kv[1]["total"], reverse=True)[:15]
    res["top_tools"] = [
        {"tool": t, "total": s["total"], "top1": (s["top1"] / s["total"]) if s["total"] else None, "top5": (s["top5"] / s["total"]) if s["total"] else None}
        for t, s in top_tools
    ]
    res["metrics_path"] = str(cfg.out_path)
    return res


