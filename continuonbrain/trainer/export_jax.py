"""
Simple exporter to convert adapter checkpoints into JAX-friendly npz artifacts.

This is a placeholder: it loads torch-styled state dicts and writes numpy arrays
so they can be picked up by a JAX/TPU pipeline for further training.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

import torch


def export_adapters_to_npz(adapter_path: Path, out_path: Path) -> Dict[str, Any]:
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    state = torch.load(adapter_path, map_location="cpu")
    np_state = {k: np.array(v) for k, v in state.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **np_state)
    manifest = {
        "source": str(adapter_path),
        "npz": str(out_path),
        "keys": list(np_state.keys()),
    }
    (out_path.with_suffix(".json")).write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export adapter to JAX-friendly npz.")
    parser.add_argument("--adapter", type=Path, required=True, help="Path to torch adapter checkpoint.")
    parser.add_argument("--out", type=Path, required=True, help="Output npz path.")
    args = parser.parse_args()
    manifest = export_adapters_to_npz(args.adapter, args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

