
import os
import sys

# Set preference BEFORE importing gemma_chat
os.environ["CONTINUON_PREFER_LITERT"] = "1"
os.environ["CONTINUON_USE_LITERT"] = "1"
# Mock JAX to be present but ignored due to preference
os.environ["CONTINUON_PREFER_JAX"] = "0" 

sys.path.append(os.getcwd())

try:
    from continuonbrain.gemma_chat import build_chat_service
    from continuonbrain.services.chat.litert_chat import LiteRTGemmaChat
    from continuonbrain.server.model_selector import select_model
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def test_selection():
    print("Testing Model Selection...")
    selection = select_model()
    print(f"Selected: {selection.get('selected')}")
    
    print("\nTesting Chat Builder...")
    chat = build_chat_service()
    if chat:
        print(f"Chat Service: {type(chat).__name__}")
        if isinstance(chat, LiteRTGemmaChat):
             print("✅ build_chat_service returned LiteRTGemmaChat")
        else:
             print(f"❌ build_chat_service returned {type(chat)}")
    else:
        print("❌ build_chat_service returned None")

if __name__ == "__main__":
    test_selection()
