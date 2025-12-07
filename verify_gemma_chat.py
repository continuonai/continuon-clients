
import sys
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add repo root to path
REPO_ROOT = Path("/home/craigm26/ContinuonXR")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set Mock Token if not present (using one found in download_model.py)
if "HUGGINGFACE_TOKEN" not in os.environ:
    os.environ["HUGGINGFACE_TOKEN"] = "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"
    print("Set fallback HUGGINGFACE_TOKEN for testing")

from continuonbrain.gemma_chat import create_gemma_chat, GemmaChat

def main():
    # Use Gemma 2B directly
    model_id = "google/gemma-2b-it"
    print(f"Initializing GemmaChat with model: {model_id}...")
    
    try:
        # Pass model_name explicitly
        chat = create_gemma_chat(use_mock=False, model_name=model_id)
    except Exception as e:
        print(f"Failed to create chat: {e}")
        return

    info = chat.get_model_info()
    print(f"Model Info: {info}")

    # Verify loading happened (or try to trigger it)
    if not info.get('loaded'):
        print("Model not loaded yet. Triggering first inference/load...")
    
    message = "Hello! Are you ready?"
    print(f"\nSending message: '{message}'")
    
    try:
        response = chat.chat(message)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Chat failed: {e}")

if __name__ == "__main__":
    main()
