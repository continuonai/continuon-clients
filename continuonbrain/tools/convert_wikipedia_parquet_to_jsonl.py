from __future__ import annotations

"""
Convert Hugging Face `wikimedia/wikipedia` Parquet shards into a lightweight JSONL corpus.

Why:
- HF datasets are commonly distributed as Parquet.
- Our offline retriever (`continuonbrain/eval/wiki_retriever.py`) supports a simple JSONL corpus:
    {"title": "...", "text": "..."}

Usage examples:
  python -m continuonbrain.tools.convert_wikipedia_parquet_to_jsonl \
    --input /path/to/wikipedia_parquet_dir \
    --output /opt/continuonos/brain/wikipedia/wikipedia.jsonl \
    --max-rows 20000

Notes:
- This is a one-time pre-processing step. The runtime uses the JSONL file.
- Requires either `pyarrow` (preferred) or `pandas` with parquet support.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional


def _iter_parquet_files(input_path: Path) -> Iterator[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".parquet":
        yield input_path
        return
    if input_path.is_dir():
        for p in sorted(input_path.rglob("*.parquet")):
            if p.is_file():
                yield p


def _iter_rows_pyarrow(parquet_path: Path, columns: list[str]) -> Iterable[dict]:
    import pyarrow.parquet as pq  # type: ignore

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=columns):
        table = batch.to_pydict()
        n = len(next(iter(table.values()))) if table else 0
        for i in range(n):
            yield {k: table.get(k, [None] * n)[i] for k in columns}


def _iter_rows_pandas(parquet_path: Path, columns: list[str]) -> Iterable[dict]:
    import pandas as pd  # type: ignore

    df = pd.read_parquet(parquet_path, columns=columns)
    for _, row in df.iterrows():
        yield {c: row.get(c) for c in columns}


def convert(
    *,
    input_path: Path,
    output_path: Path,
    max_rows: Optional[int] = None,
    prefer: str = "pyarrow",
) -> dict:
    files = list(_iter_parquet_files(input_path))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["title", "text"]
    written = 0
    used = None

    def iter_rows(parquet_file: Path) -> Iterable[dict]:
        nonlocal used
        if prefer == "pandas":
            used = "pandas"
            return _iter_rows_pandas(parquet_file, columns)
        # default: pyarrow
        used = "pyarrow"
        return _iter_rows_pyarrow(parquet_file, columns)

    with output_path.open("w", encoding="utf-8") as out:
        for pf in files:
            try:
                rows = iter_rows(pf)
            except Exception:
                # Try fallback reader if available
                if used == "pyarrow":
                    used = "pandas"
                    rows = _iter_rows_pandas(pf, columns)
                else:
                    used = "pyarrow"
                    rows = _iter_rows_pyarrow(pf, columns)

            for r in rows:
                title = r.get("title")
                text = r.get("text")
                if not isinstance(title, str) or not isinstance(text, str):
                    continue
                title = title.strip()
                text = text.strip()
                if not title or not text:
                    continue
                out.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")
                written += 1
                if max_rows is not None and written >= max_rows:
                    return {
                        "status": "ok",
                        "input": str(input_path),
                        "output": str(output_path),
                        "written_rows": written,
                        "reader": used,
                        "files_scanned": len(files),
                        "truncated": True,
                    }

    return {
        "status": "ok",
        "input": str(input_path),
        "output": str(output_path),
        "written_rows": written,
        "reader": used,
        "files_scanned": len(files),
        "truncated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert wikimedia/wikipedia parquet shards to JSONL.")
    parser.add_argument("--input", required=True, help="Input parquet file or directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-rows", type=int, default=20000, help="Max rows to write (default: 20000)")
    parser.add_argument(
        "--prefer",
        choices=["pyarrow", "pandas"],
        default="pyarrow",
        help="Preferred parquet reader backend",
    )
    args = parser.parse_args()

    res = convert(
        input_path=Path(args.input),
        output_path=Path(args.output),
        max_rows=None if args.max_rows and args.max_rows <= 0 else args.max_rows,
        prefer=args.prefer,
    )
    sys.stdout.write(json.dumps(res, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()


