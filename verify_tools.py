
import os
import sys
import logging
import json

# Setup environment
os.environ["HUGGINGFACE_TOKEN"] = "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"
sys.path.append("/home/craigm26/ContinuonXR")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_tools")

from continuonbrain.services.brain_service import BrainService

def verify_tools():
    print("\nInitialzing BrainService (Mock Mode)...")
    service = BrainService(
        config_dir="/tmp/test_tools",
        prefer_real_hardware=False,
        auto_detect=False,
        allow_mock_fallback=True
    )
    
    # Ensure system instructions are loaded for safety check
    service._ensure_system_instructions()
    
    # Enable status updates in agent settings
    service.agent_settings = {"enable_status_updates": True}

    # Test TERMINAL Tool
    print("\n--- Testing TERMINAL Tool ---")
    mock_response = "[TOOL: TERMINAL echo 'Hello from Terminal']"
    
    # We cheat and mock the chat response logic by calling the internal tool parser 
    # effectively by simulating a response that contains the tool command.
    # But since ChatWithGemma executes the chat, we need to mock gemma_chat.chat to return our string.
    
    original_chat = service.gemma_chat.chat
    service.gemma_chat.chat = lambda msg, system_context: mock_response
    
    result = service.ChatWithGemma("Run echo command", [])
    print(f"Response: {result}")
    
    updates = result.get("status_updates", [])
    found_execution = any("Executed 'echo" in update for update in updates)
    
    if found_execution:
        print("✅ PASS: TERMINAL tool executed")
    else:
        print(f"❌ FAIL: TERMINAL tool not executed. Updates: {updates}")
        return False

    # Test BROWSER Tool (Mocked webbrowser)
    print("\n--- Testing BROWSER Tool ---")
    service.gemma_chat.chat = lambda msg, system_context=None: "[TOOL: BROWSER google.com]"
    
    # Mock webbrowser to avoid actually opening a window
    import webbrowser
    original_open = webbrowser.open
    webbrowser.open = lambda url: print(f"  [Mock Browser] Opening {url}")
    
    result = service.ChatWithGemma("Open google", [])
    updates = result.get("status_updates", [])
    found_browser = any("Opened https://google.com" in update for update in updates)
    
    webbrowser.open = original_open # Restore
    
    if found_browser:
        print("✅ PASS: BROWSER tool executed")
    else:
        print(f"❌ FAIL: BROWSER tool not executed. Updates: {updates}")
        return False
        
    return True

if __name__ == "__main__":
    success = verify_tools()
    sys.exit(0 if success else 1)
