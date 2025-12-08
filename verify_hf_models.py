
import os
from pathlib import Path
import json

CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

def verify_models():
    print(f"Checking HF Cache at: {CACHE_DIR}")
    if not CACHE_DIR.exists():
        print("Cache directory not found!")
        return

    for model_dir in CACHE_DIR.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue
            
        print(f"\nFound Model Directory: {model_dir.name}")
        
        # Check snapshots
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            print("  - No snapshots directory!")
            continue
            
        snapshots = list(snapshots_dir.iterdir())
        print(f"  - Snapshots found: {len(snapshots)}")
        
        for snap in snapshots:
            size_bytes = sum(f.stat().st_size for f in snap.rglob("*") if f.is_file())
            size_mb = size_bytes / (1024 * 1024)
            print(f"    - Snapshot {snap.name}: {size_mb:.2f} MB")
            
            # Check for config.json
            if (snap / "config.json").exists():
                print("      - config.json found (VALID)")
            else:
                print("      - config.json MISSING (INVALID?)")

if __name__ == "__main__":
    verify_models()
