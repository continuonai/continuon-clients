from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from continuonbrain.jax_models.train.tool_router_train import featurize


@dataclass
class ToolRouterBundle:
    weights_path: Path
    manifest_path: Path
    labels: List[str]
    W: np.ndarray
    b: np.ndarray
    features_dim: int


def load_tool_router_bundle(export_dir: Path) -> ToolRouterBundle:
    export_dir = Path(export_dir)
    manifest_path = export_dir / "tool_router_manifest.json"
    weights_path = export_dir / "tool_router.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing tool router manifest: {manifest_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing tool router weights: {weights_path}")

    manifest = json.loads(manifest_path.read_text())
    labels = manifest.get("labels") or []
    if not isinstance(labels, list) or not labels:
        raise ValueError("tool_router_manifest.json missing labels")
    cfg = manifest.get("config") or {}
    features_dim = int(cfg.get("features_dim") or 4096)

    data = np.load(weights_path)
    W = data["W"]
    b = data["b"]
    if W.shape[0] != features_dim:
        raise ValueError(f"Feature dim mismatch: manifest={features_dim} weights={W.shape[0]}")
    if W.shape[1] != len(labels) or b.shape[0] != len(labels):
        raise ValueError("Label count mismatch between manifest and weights")

    return ToolRouterBundle(
        weights_path=weights_path,
        manifest_path=manifest_path,
        labels=[str(x) for x in labels],
        W=W.astype(np.float32),
        b=b.astype(np.float32),
        features_dim=features_dim,
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-9)


def predict_topk(bundle: ToolRouterBundle, prompt: str, k: int = 5) -> List[Dict[str, Any]]:
    k = max(1, min(20, int(k)))
    x = featurize(prompt or "", bundle.features_dim)
    logits = x @ bundle.W + bundle.b
    probs = _softmax(logits.astype(np.float64))
    idxs = np.argsort(-probs)[:k]
    out = []
    for i in idxs:
        out.append({"tool": bundle.labels[int(i)], "score": float(probs[int(i)])})
    return out


