import sys
import os
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.gemma_chat import create_gemma_chat

def verify():
    print("Verifying Gemma Chat installation...")
    
    try:
        import transformers
        import torch
        print(f"✅ Transformers version: {transformers.__version__}")
        print(f"✅ Torch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import dependencies: {e}")
        return False

    print("Attempting to create GemmaChat instance...")
    chat = create_gemma_chat(use_mock=False)
    
    model_info = chat.get_model_info()
    print(f"Model Info: {model_info}")
    
    if model_info['model_name'] == 'mock':
        print("❌ GemmaChat is still using mock implementation!")
        return False
    
    print("✅ GemmaChat initialized with real implementation!")
    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
