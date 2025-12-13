from __future__ import annotations

"""
Import tool/function-calling datasets into RLDS episodes for HOPE seed training.

Design goals:
- Dependency-light: prefer stdlib; optional pyarrow/pandas for Parquet.
- Offline-first: input is local JSONL or Parquet shards (e.g., from `git lfs` dataset clones).
- Best-effort schema handling: many datasets vary in field names; we detect common shapes.

Output:
- Writes episodes under <episodes-dir>/toolchat_<ts>_<n>.json with RLDS-style steps.
  Steps follow a generic pattern:
    - observation: {type: "chat", role: "user", content: "..."}
    - action:      {type: "chat", role: "assistant", content: "..."} OR
                   {type: "tool_call", name: "...", arguments: {...}}
    - observation: {type: "tool_result", name: "...", content: "..."} (when available)

This is intentionally generic; for high-quality results, add dataset-specific adapters later.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def _safe(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (s or ""))
    return out[:80] or "dataset"


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _iter_json(path: Path) -> Iterator[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        # Some datasets wrap rows under a key (e.g., {"data":[...]})
        for key in ("data", "rows", "items", "examples"):
            v = payload.get(key)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        yield item
                return
        # Fallback: treat entire dict as one row
        yield payload


def _iter_parquet_files(input_path: Path) -> Iterator[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".parquet":
        yield input_path
        return
    if input_path.is_dir():
        for p in sorted(input_path.rglob("*.parquet")):
            if p.is_file():
                yield p


def _iter_rows_pyarrow(parquet_path: Path) -> Iterable[dict]:
    import pyarrow.parquet as pq  # type: ignore

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches():
        table = batch.to_pydict()
        n = len(next(iter(table.values()))) if table else 0
        keys = list(table.keys())
        for i in range(n):
            yield {k: table.get(k, [None] * n)[i] for k in keys}


def _iter_rows_pandas(parquet_path: Path) -> Iterable[dict]:
    import pandas as pd  # type: ignore

    df = pd.read_parquet(parquet_path)
    for _, row in df.iterrows():
        # convert to plain python dict
        d = {}
        for k in df.columns:
            d[k] = row.get(k)
        yield d


def _coerce_json(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def _extract_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract chat messages in a uniform [{role, content}] format when possible.
    Supports common fields:
    - messages: [{role, content}, ...]
    - prompt/completion
    - instruction/output
    """
    for key in ("messages", "conversations"):
        raw = _coerce_json(sample.get(key))
        if isinstance(raw, list):
            out = []
            for m in raw:
                if not isinstance(m, dict):
                    continue
                role = m.get("role") or m.get("from") or m.get("speaker") or ""
                content = m.get("content") or m.get("value") or m.get("text") or ""
                if role and content:
                    out.append({"role": str(role), "content": str(content)})
            if out:
                return out

    # ToolBench-style parquet: {"conversations": {"from": [...], "value": [...]} }
    conv = sample.get("conversations")
    if isinstance(conv, dict) and isinstance(conv.get("from"), list) and isinstance(conv.get("value"), list):
        roles = conv.get("from") or []
        vals = conv.get("value") or []
        out = []
        for r, v in zip(roles, vals):
            if not isinstance(r, str):
                continue
            if not isinstance(v, str):
                v = str(v)
            out.append({"role": r, "content": v})
        if out:
            return out

    # Glaive function calling v2: {"system": "...", "chat": "USER: ... ASSISTANT: ... <|endoftext|>"}
    if isinstance(sample.get("chat"), str):
        chat = sample["chat"]
        out: List[Dict[str, str]] = []
        sys_msg = sample.get("system")
        if isinstance(sys_msg, str) and sys_msg.strip():
            out.append({"role": "system", "content": sys_msg.strip()})

        # Split into segments (dataset often terminates with <|endoftext|>)
        chat = chat.replace("<|endoftext|>", "").strip()
        # Parse simple "ROLE: ..." markers
        buf_role: Optional[str] = None
        buf_lines: List[str] = []
        for line in chat.splitlines():
            raw = line.strip()
            if not raw:
                continue
            m = re.match(r"^(USER|ASSISTANT|SYSTEM|FUNCTION|TOOL)\\s*:\\s*(.*)$", raw, flags=re.IGNORECASE)
            if m:
                if buf_role and buf_lines:
                    out.append({"role": buf_role.lower(), "content": "\n".join(buf_lines).strip()})
                buf_role = m.group(1).lower()
                buf_lines = [m.group(2)]
            else:
                buf_lines.append(raw)
        if buf_role and buf_lines:
            out.append({"role": buf_role.lower(), "content": "\n".join(buf_lines).strip()})
        if out:
            return out

    prompt = sample.get("prompt") or sample.get("instruction") or sample.get("query") or sample.get("question")
    completion = sample.get("completion") or sample.get("output") or sample.get("response") or sample.get("answer")
    if isinstance(prompt, str) and isinstance(completion, str):
        # Some function-calling datasets store tool calls in "answers" while omitting a natural-language completion.
        # If the completion looks like tool calls, log only the user prompt and let tool_call extraction handle the rest.
        if completion.strip().startswith("[{") and ("\"name\"" in completion or "'name'" in completion):
            return [{"role": "user", "content": prompt}]
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
    # Function-calling datasets often omit a natural-language assistant completion and only provide tool calls.
    # If we have a prompt plus any tool-call fields, log at least the user turn.
    if isinstance(prompt, str) and (
        sample.get("answers") is not None
        or sample.get("tool_calls") is not None
        or sample.get("function_calls") is not None
        or sample.get("calls") is not None
        or sample.get("tool_call") is not None
        or sample.get("function_call") is not None
    ):
        return [{"role": "user", "content": prompt}]
    return []


def _extract_tool_calls(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract tool calls in a uniform shape:
      {name: str, arguments: dict|str, id: optional}
    Supports common fields:
    - tool_calls / function_calls
    - calls
    - "tool_call" embedded objects
    """
    for key in ("tool_calls", "function_calls", "calls"):
        raw = _coerce_json(sample.get(key))
        if isinstance(raw, list):
            out = []
            for c in raw:
                if not isinstance(c, dict):
                    continue
                name = c.get("name") or c.get("function") or (c.get("function") or {}).get("name")
                args = c.get("arguments") or (c.get("function") or {}).get("arguments") or c.get("args")
                if name:
                    out.append({"name": str(name), "arguments": _coerce_json(args), "id": c.get("id")})
            if out:
                return out

    # xLAM-style: "answers" is a JSON string of [{"name": "...", "arguments": {...}}, ...]
    raw = _coerce_json(sample.get("answers"))
    if isinstance(raw, list):
        out = []
        for c in raw:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            args = c.get("arguments")
            if name:
                out.append({"name": str(name), "arguments": _coerce_json(args), "id": c.get("id")})
        if out:
            return out

    # Single-call variants
    raw = sample.get("tool_call") or sample.get("function_call")
    raw = _coerce_json(raw)
    if isinstance(raw, dict):
        name = raw.get("name") or (raw.get("function") or {}).get("name")
        args = raw.get("arguments") or (raw.get("function") or {}).get("arguments") or raw.get("args")
        if name:
            return [{"name": str(name), "arguments": _coerce_json(args), "id": raw.get("id")}]
    return []


def _extract_tool_results(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract tool results (best-effort):
      {name: str, content: str, id: optional}
    """
    for key in ("tool_results", "results", "observations"):
        raw = _coerce_json(sample.get(key))
        if isinstance(raw, list):
            out = []
            for r in raw:
                if isinstance(r, dict):
                    name = r.get("name") or r.get("tool_name") or r.get("function") or ""
                    content = r.get("content") or r.get("result") or r.get("output") or r.get("observation") or ""
                    if name or content:
                        out.append({"name": str(name) if name else None, "content": str(content), "id": r.get("id")})
                elif isinstance(r, str):
                    out.append({"name": None, "content": r, "id": None})
            if out:
                return out
    return []


def _build_episode(
    *,
    sample: Dict[str, Any],
    source: str,
    dataset_id: str,
    idx: int,
) -> Dict[str, Any]:
    now = time.time()
    steps: List[Dict[str, Any]] = []
    msgs = _extract_messages(sample)
    tool_calls = _extract_tool_calls(sample)
    tool_results = _extract_tool_results(sample)
    tool_specs = _coerce_json(sample.get("tools"))
    if not isinstance(tool_specs, list):
        tool_specs = []

    # If we have ToolBench-like conversations, preserve order and infer tool calls from assistant text.
    conv = sample.get("conversations")
    if isinstance(conv, dict) and isinstance(conv.get("from"), list) and isinstance(conv.get("value"), list):
        roles = conv.get("from") or []
        vals = conv.get("value") or []
        call_re = re.compile(r"call (?:the )?[\"'](?P<name>[^\"']+)[\"'] function", flags=re.IGNORECASE)
        args_re = re.compile(r"(?:args|arguments)\\s*:\\s*(\\{[\\s\\S]*\\})", flags=re.IGNORECASE)

        for role, content in zip(roles, vals):
            if not isinstance(role, str):
                continue
            if not isinstance(content, str):
                content = str(content)
            role_l = role.lower()

            if role_l in ("system", "user", "assistant"):
                steps.append(
                    {
                        "step_index": len(steps),
                        "timestamp_ns": int(now * 1e9),
                        "observation": {"type": "chat", "role": role_l, "content": content},
                        "action": {},
                        "reward": 0.0,
                        "is_terminal": False,
                        "step_metadata": {"source": source, "dataset_id": dataset_id, "row_index": idx},
                    }
                )
                if role_l == "assistant":
                    m = call_re.search(content)
                    if m:
                        name = m.group("name").strip()
                        args_raw = None
                        am = args_re.search(content)
                        if am:
                            args_raw = am.group(1)
                            args_raw = _coerce_json(args_raw)
                        steps.append(
                            {
                                "step_index": len(steps),
                                "timestamp_ns": int(now * 1e9),
                                "observation": {"type": "tool_call_request"},
                                "action": {"type": "tool_call", "name": name, "arguments": args_raw, "call_id": None},
                                "reward": 0.0,
                                "is_terminal": False,
                                "step_metadata": {"source": source, "dataset_id": dataset_id, "row_index": idx},
                            }
                        )
            elif role_l in ("function", "tool"):
                steps.append(
                    {
                        "step_index": len(steps),
                        "timestamp_ns": int(now * 1e9),
                        "observation": {"type": "tool_result", "name": None, "content": content, "call_id": None},
                        "action": {},
                        "reward": 0.0,
                        "is_terminal": False,
                        "step_metadata": {"source": source, "dataset_id": dataset_id, "row_index": idx},
                    }
                )
    else:
        # 1) Messages (user/assistant/system)
        for m in msgs:
            steps.append(
                {
                    "step_index": len(steps),
                    "timestamp_ns": int(now * 1e9),
                    "observation": {
                        "type": "chat",
                        "role": m.get("role"),
                        "content": m.get("content"),
                    },
                    "action": {},
                    "reward": 0.0,
                    "is_terminal": False,
                    "step_metadata": {"source": source, "dataset_id": dataset_id, "row_index": idx},
                }
            )

        # 2) Tool calls (if any)
        for tc in tool_calls:
            steps.append(
                {
                    "step_index": len(steps),
                    "timestamp_ns": int(now * 1e9),
                    "observation": {"type": "tool_call_request"},
                    "action": {
                        "type": "tool_call",
                        "name": tc.get("name"),
                        "arguments": tc.get("arguments"),
                        "call_id": tc.get("id"),
                    },
                    "reward": 0.0,
                    "is_terminal": False,
                    "step_metadata": {"source": source, "dataset_id": dataset_id, "row_index": idx},
                }
            )

        # 3) Tool results (if any)
        for tr in tool_results:
            steps.append(
                {
                    "step_index": len(steps),
                    "timestamp_ns": int(now * 1e9),
                    "observation": {
                        "type": "tool_result",
                        "name": tr.get("name"),
                        "content": tr.get("content"),
                        "call_id": tr.get("id"),
                    },
                    "action": {},
                    "reward": 0.0,
                    "is_terminal": False,
                    "step_metadata": {"source": source, "dataset_id": dataset_id, "row_index": idx},
                }
            )

    if steps:
        steps[-1]["is_terminal"] = True

    return {
        "metadata": {
            "created_unix_s": int(now),
            "source": source,
            "dataset_id": dataset_id,
            "row_index": idx,
            "has_tool_calls": bool(tool_calls),
            "has_tool_results": bool(tool_results),
            "tool_specs": tool_specs,
        },
        "steps": steps,
    }


def import_dataset(
    *,
    input_path: Path,
    input_format: str,
    episodes_dir: Path,
    dataset_id: str,
    max_episodes: int,
    source: str,
    parquet_reader: str,
) -> Dict[str, Any]:
    episodes_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    wrote = 0
    skipped = 0

    def write_episode(ep: Dict[str, Any], n: int) -> None:
        out = episodes_dir / f"toolchat_{_safe(dataset_id)}_{ts}_{n:06d}.json"
        out.write_text(json.dumps(ep, indent=2, default=str))

    def handle_rows(rows: Iterable[Dict[str, Any]]) -> None:
        nonlocal wrote, skipped
        for sample in rows:
            if wrote >= max_episodes:
                break
            if not isinstance(sample, dict):
                skipped += 1
                continue
            ep = _build_episode(sample=sample, source=source, dataset_id=dataset_id, idx=wrote + skipped)
            if not ep.get("steps"):
                skipped += 1
                continue
            write_episode(ep, wrote)
            wrote += 1

    if input_format == "jsonl":
        handle_rows(_iter_jsonl(input_path))
    elif input_format == "json":
        handle_rows(_iter_json(input_path))
    elif input_format == "parquet":
        files = list(_iter_parquet_files(input_path))
        if not files:
            raise FileNotFoundError(f"No .parquet files found under {input_path}")
        for pf in files:
            if wrote >= max_episodes:
                break
            if parquet_reader == "pandas":
                handle_rows(_iter_rows_pandas(pf))
            else:
                handle_rows(_iter_rows_pyarrow(pf))
    else:
        raise ValueError("input_format must be one of: jsonl, json, parquet")

    return {
        "status": "ok",
        "input": str(input_path),
        "input_format": input_format,
        "episodes_dir": str(episodes_dir),
        "dataset_id": dataset_id,
        "wrote_episodes": wrote,
        "skipped_rows": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import tool/function-calling dataset into RLDS episodes.")
    parser.add_argument("--input", required=True, help="Input JSONL file or Parquet file/dir")
    parser.add_argument("--format", choices=["jsonl", "json", "parquet"], required=True)
    parser.add_argument("--episodes-dir", default="/opt/continuonos/brain/rlds/episodes")
    parser.add_argument("--dataset-id", default="unknown")
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--source", default="hf_tool_dataset")
    parser.add_argument("--parquet-reader", choices=["pyarrow", "pandas"], default="pyarrow")
    args = parser.parse_args()

    res = import_dataset(
        input_path=Path(args.input),
        input_format=args.format,
        episodes_dir=Path(args.episodes_dir),
        dataset_id=args.dataset_id,
        max_episodes=max(1, args.max_episodes),
        source=args.source,
        parquet_reader=args.parquet_reader,
    )
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()


