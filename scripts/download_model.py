import os
from huggingface_hub import snapshot_download
from pathlib import Path

MODEL_ID = "google/gemma-3n-2b-it"
LOCAL_DIR = Path.home() / "models/gemma-3n-2b-it"

print(f"Downloading {MODEL_ID} to {LOCAL_DIR}...")
try:
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("Download complete.")
except Exception as e:
    print(f"Error downloading model: {e}")
    # specific help for auth errors
    if "401" in str(e) or "403" in str(e):
        print("\nAUTH ERROR: You need to authenticate with Hugging Face.")
        print("Run: huggingface-cli login")
        print("Or set HF_TOKEN environment variable.")
