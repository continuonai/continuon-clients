"""
Debug Gemma Loading.
"""
import sys
import os
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from continuonbrain.gemma_chat import GemmaChat

def debug_load():
    print("Python Executable:", sys.executable)
    print("Sys Path:", sys.path)
    
    # Set HuggingFace token if not already set
    if not os.environ.get('HUGGINGFACE_TOKEN'):
        os.environ['HUGGINGFACE_TOKEN'] = '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'
        print("Set HUGGINGFACE_TOKEN from script")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"Transformers import failed: {e}")
        return

    # Test Gemma 3 270M model (smaller, faster)
    print("\n=== Testing Gemma 3 270M ===")
    chat_270m = GemmaChat(model_name="google/gemma-3-270m", device="cpu")
    print("Attempting to load Gemma 3 270M model...")
    success_270m = chat_270m.load_model()
    
    if success_270m:
        print("✅ Gemma 3 270M load successful")
        # Test a simple chat
        response = chat_270m.chat("Hello, what can you do?")
        print(f"Test response: {response[:100]}...")
    else:
        print("❌ Gemma 3 270M load failed")

    # Test Gemma 3N-E2B model (instruction-tuned)
    print("\n=== Testing Gemma 3N-E2B ===")
    chat_3n = GemmaChat(model_name="google/gemma-3n-E2B-it", device="cpu")
    print("Attempting to load Gemma 3N-E2B model...")
    success_3n = chat_3n.load_model()
    
    if success_3n:
        print("✅ Gemma 3N-E2B load successful")
        # Test a simple chat
        response = chat_3n.chat("Hello, what can you do?")
        print(f"Test response: {response[:100]}...")
    else:
        print("❌ Gemma 3N-E2B load failed")

if __name__ == "__main__":
    debug_load()
