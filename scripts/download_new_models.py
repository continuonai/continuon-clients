from huggingface_hub import snapshot_download
from pathlib import Path
import sys

MODELS = [
    # Function Calling (Small)
    ("google/functiongemma-270m-it", "functiongemma-270m-it"),
    
    # Embeddings (Small)
    ("google/embeddinggemma-300m", "embeddinggemma-300m"),
    
    # Chat (User requested Gemma 3n 2B - Testing valid IDs)
    # Note: If 2B doesn't exist, we try 1B (closest small v3) or 4B.
    # User specifically said "not gemma 2".
    ("google/gemma-3-1b-it", "gemma-3-1b-it"),  # Likely candidate for "small v3"
    ("google/gemma-3-4b-it", "gemma-3-4b-it"),  # Next size up if 1B/2B fail
]

BASE_DIR = Path.home() / "models"
BASE_DIR.mkdir(parents=True, exist_ok=True)

for repo_id, local_name in MODELS:
    print(f"\n⬇️ Downloading {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=BASE_DIR / local_name,
            local_dir_use_symlinks=False,
            # For 3n/2b ambiguity, fail fast if not found so we can try next
        )
        print(f"✅ Downloaded {repo_id}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")

print("\nDownload process complete.")
