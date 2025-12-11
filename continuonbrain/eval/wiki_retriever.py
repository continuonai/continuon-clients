from __future__ import annotations

"""
Optional Wikipedia retriever using a local HuggingFace dataset or JSONL corpus.

Design goals:
- Do nothing (return empty) if the corpus is not present or dependencies are missing.
- Keep runtime bounded by scanning a limited number of documents.
- Avoid heavy dependencies; use a simple token-overlap score.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace("\n", " ").split() if t]


def _overlap_score(query_tokens: List[str], doc_tokens: List[str]) -> int:
    qs = set(query_tokens)
    return sum(1 for t in doc_tokens if t in qs)


def build_wiki_retriever(
    *,
    dataset_path: Optional[Path] = None,
    hf_subset: str = "20231101.en",
    max_scan: int = 500,
    top_k: int = 3,
    max_chars: int = 800,
) -> Callable[[str], List[Dict[str, str]]]:
    """
    Create a retriever function that returns up to top_k snippets.
    If no corpus/dependencies are available, the retriever returns [].
    """
    ds = None
    use_jsonl = False
    jsonl_lines: List[str] = []

    # Try HuggingFace datasets (streaming) if available and dataset_path is None
    if dataset_path is None:
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset("wikimedia/wikipedia", hf_subset, split="train", streaming=True)
        except Exception:
            ds = None
    else:
        p = Path(dataset_path)
        if p.exists() and p.is_file():
            try:
                jsonl_lines = p.read_text(encoding="utf-8").splitlines()
                use_jsonl = True
            except Exception:
                jsonl_lines = []
                use_jsonl = False

    def _retrieve(query: str) -> List[Dict[str, str]]:
        if not ds and not use_jsonl:
            return []

        q_tokens = _tokenize(query)
        scored: List[Tuple[int, Dict[str, str]]] = []

        def consider(title: str, text: str):
            tokens = _tokenize(text)
            score = _overlap_score(q_tokens, tokens)
            if score <= 0:
                return
            snippet = text[:max_chars]
            scored.append(
                (
                    score,
                    {
                        "title": title,
                        "text": snippet,
                    },
                )
            )

        if ds:
            # Streaming: scan at most max_scan rows
            for i, row in enumerate(ds):
                if i >= max_scan:
                    break
                title = row.get("title") or ""
                text = row.get("text") or ""
                consider(title, text)
        elif use_jsonl:
            for i, line in enumerate(jsonl_lines):
                if i >= max_scan:
                    break
                try:
                    import json

                    obj = json.loads(line)
                    title = obj.get("title") or ""
                    text = obj.get("text") or ""
                    consider(title, text)
                except Exception:
                    continue

        # Sort by score descending, keep top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    return _retrieve
