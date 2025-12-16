
import os
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.gemma_chat import _snapshot_download_path

model_id = "google/gemma-3n-E2B-it-litert-lm"
token = os.environ.get("HUGGINGFACE_TOKEN")

print(f"Resolving path for {model_id}...")
try:
    path = _snapshot_download_path(model_id, token)
    print(f"Snapshot path: {path}")
    
    if path.exists():
        print("Files in snapshot:")
        for child in path.iterdir():
            print(f" - {child.name} ({child.stat().st_size} bytes)")
    else:
        print("Snapshot path does not exist!")
        
except Exception as e:
    print(f"Error resolving path: {e}")
