import sys
import logging
from continuonbrain.gemma_chat import create_gemma_chat

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_chat():
    print("Testing GemmaChat...")
    
    try:
        # Create chat - should use transformers if available
        chat = create_gemma_chat(use_mock=False, device="cpu")
        print(f"Chat instance type: {type(chat).__name__}")
        
        # Determine model info
        info = chat.get_model_info()
        print(f"Model Info (Pre-load): {info}")
        
        # Attempt load
        print("Loading model...")
        success = chat.load_model()
        
        if not success:
            print("❌ Failed to load model")
            return
            
        print("✅ Model loaded successfully")
        
        # Test generation
        print("Testing generation...")
        response = chat.chat("Hello!", system_context="You are a robot.")
        print(f"Response: {response}")
        
        if "Mock" in response or "mock" in str(type(chat)).lower():
             print("⚠️  Result indicates MOCK implementation or fallback.")
        else:
             print("✅ Real inference appears to be working.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chat()
