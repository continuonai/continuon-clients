
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure repo root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Inject Token (found in startup_manager.py)
os.environ["HUGGINGFACE_TOKEN"] = "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"

try:
    from continuonbrain.gemma_chat import GemmaChat
except ImportError:
    # Try adjusting path if running from root
    sys.path.append(os.getcwd())
    from continuonbrain.gemma_chat import GemmaChat

def test_inference():
    print("🚀 Initializing GemmaChat (Expect 4B model)...")
    
    # Initialize chat
    # It should use the new DEFAULT_MODEL_ID = "google/gemma-3-4b-it"
    chat = GemmaChat(device="cpu") # Force CPU for test to be safe/simple
    
    print(f"📋 Model ID: {chat.model_name}")
    
    # Test Prompt
    prompt = "Hello! Please introduce yourself in one sentence."
    print(f"\n👤 User: {prompt}")
    
    try:
        if not chat.load_model():
            print("❌ Failed to load model.")
            sys.exit(1)
            
        print("🤖 Model loaded. Generating response...")
        response = chat.chat(prompt)
        print(f"\n🧠 Model: {response}")
        print("\n✅ Inference Test Passed!")
    except Exception as e:
        print(f"\n❌ Inference Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
