
import os
import sys
import time
import logging

# Setup environment
os.environ["HUGGINGFACE_TOKEN"] = "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"
sys.path.append("/home/craigm26/ContinuonXR")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_learning")

from continuonbrain.services.brain_service import BrainService
from continuonbrain.hope_impl.config import HOPEConfig
from continuonbrain.gemma_chat import GemmaChat

def verify():
    # 1. Verify Gemma Model ID
    chat = GemmaChat()
    print(f"Gemma Model ID: {chat.model_name}")
    if chat.model_name != "google/gemma-3-4b-it":
        print("❌ FAIL: Incorrect Gemma Model ID")
        return False
    print("✅ PASS: Gemma Model ID correct")

    # 2. Verify Background Learner Startup in BrainService
    print("\nInitializing BrainService (Mock Mode)...")
    service = BrainService(
        config_dir="/tmp/test_continuonbrain",
        prefer_real_hardware=False,
        auto_detect=False,
        allow_mock_fallback=True
    )
    
    # Force config to enable learning
    # Note: BrainService.initialize() loads config based on memory availability
    # We need to make sure it picks a config with learning enabled, or we patch it.
    
    try:
        import asyncio
        asyncio.run(service.initialize())
        
        if service.background_learner is None:
            print("❌ FAIL: Background learner not initialized")
            return False
            
        if not service.background_learner.running:
            print("❌ FAIL: Background learner not running")
            return False
            
        print("✅ PASS: Background learner initialized and running")
        
        # Clean shutdown
        service.background_learner.stop()
        
    except Exception as e:
        print(f"❌ FAIL: Initialization exception: {e}")
        return False

    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
