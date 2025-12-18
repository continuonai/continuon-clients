
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock env vars
os.environ["CONTINUON_ALLOW_MODEL_DOWNLOADS"] = "0"

def _resolve_hf_hub_dir() -> Path:
    return Path("/home/craigm26/.cache/huggingface/hub")

def _snapshot_download_path(model_id: str):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=model_id, local_files_only=True))

def test_load():
    model_name = "google/gemma-3-270m-it"
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
        import torch
        from transformers import AutoModelForImageTextToText
        
        print(f"Loading {model_name}...")
        snapshot_path = _snapshot_download_path(model_name)
        print(f"Snapshot: {snapshot_path}")
        
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(str(snapshot_path), local_files_only=True, trust_remote_code=True)
        print("Processor loaded.")
        
        print("Loading VLM model...")
        model = AutoModelForImageTextToText.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=True,
            device_map=None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False
        )
        print("Model loaded.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
