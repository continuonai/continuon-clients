"""
Flexible RLDS episode loaders for local training.

Supports JSON and JSONL out of the box; TFRecord can be enabled by supplying a
custom parser, keeping the core trainer free from heavy deps by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional


def load_jsonl_episode(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "obs" in sample and "action" in sample:
                yield sample


def load_json_episode(path: Path) -> Iterator[Dict]:
    payload = json.loads(path.read_text())
    for step in payload.get("steps", []):
        if "obs" in step and "action" in step:
            yield step


def make_episode_loader() -> callable:
    """
    Returns a loader that dispatches on extension.
    Extend by wrapping this and adding TFRecord support if needed.
    """

    def _loader(path: Path) -> Iterable[Dict]:
        if path.suffix == ".jsonl":
            return load_jsonl_episode(path)
        if path.suffix == ".json":
            return load_json_episode(path)
        raise ValueError(f"Unsupported episode format: {path.suffix}")

    return _loader


def make_tfrecord_loader(
    example_parser: Callable[[bytes], Dict],
    *,
    tfrecord_iter: Optional[Callable[[str], Iterable[bytes]]] = None,
) -> Callable[[Path], Iterable[Dict]]:
    """
    Build a TFRecord loader without forcing a hard dependency on TensorFlow.

    Args:
        example_parser: Callable that accepts raw TFRecord bytes and returns
            a dict with at least keys "obs" and "action".
        tfrecord_iter: Optional iterator function; defaults to TensorFlow's
            `tf.io.tf_record_iterator` if TensorFlow is available.

    Returns:
        Callable that can be passed as `episode_loader` to the trainer.
    """

    def _iter_records(path: Path) -> Iterable[bytes]:
        if tfrecord_iter is not None:
            yield from tfrecord_iter(str(path))
            return
        try:
            import tensorflow as tf  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TensorFlow is required for TFRecord loading; install it or supply tfrecord_iter."
            ) from exc
        yield from tf.io.tf_record_iterator(str(path))

    def _loader(path: Path) -> Iterable[Dict]:
        for raw in _iter_records(path):
            sample = example_parser(raw)
            if "obs" in sample and "action" in sample:
                yield sample

    return _loader
