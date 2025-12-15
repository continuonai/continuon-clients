
import os
import sys

# Ensure we can import from local source
sys.path.append(os.getcwd())

# Set env to prefer LiteRT
os.environ["CONTINUON_USE_LITERT"] = "1"
os.environ["CONTINUON_HEADLESS"] = "1" # Disable transformers fallback to prove LiteRT

try:
    from continuonbrain.gemma_chat import build_chat_service
    print("Building chat service...")
    chat = build_chat_service()
    
    if chat is None:
        print("FAIL: build_chat_service returned None")
        sys.exit(1)
        
    print(f"Chat service built: {type(chat)}")
    
    if "LiteRT" not in str(type(chat)):
        print(f"FAIL: Expected LiteRTGemmaChat, got {type(chat)}")
        sys.exit(1)
        
    print("LiteRT backend successfully selected.")
    
    # Optional: Try to verify imports inside the class
    # (We won't load the model fully to avoid huge downloads in this quick check, 
    # unless user has it cached. But we can check if the class allows 'load_model' to be called)
    print("LiteRT backend verification successful.")
    
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
