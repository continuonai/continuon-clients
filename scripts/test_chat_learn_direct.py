import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Setup paths
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from continuonbrain.services.brain_service import BrainService

async def run_test():
    print("Initializing BrainService (Mock)...")
    
    # Mock dependencies to avoid hardware/model loading
    # We want to test RunChatLearn logic specifically
    
    service = BrainService(
        config_dir=repo_root / "brain_config",
        prefer_real_hardware=False,
        auto_detect=False # Don't run auto-detect
    )
    
    # Manually set mocked components usually set by initialize()
    service.mode_manager = MagicMock()
    service.mode_manager.current_mode.value = "autonomous"
    
    service.hope_brain = MagicMock()
    service.hope_brain.generate_response.return_value = "This is a mock response from HOPE."
    
    # We need ChatWithGemma to return something for odd turns (Agent Manager)
    # so we can test if Gemini is called for even turns (Subagent).
    
    # Monkey-patch ChatWithGemma to simulate local LLM
    def mock_chat(message, history, session_id=None):
        print(f"[MockLocalLLM] responding to: {message[:50]}...")
        return {"response": "What is the capital of France?", "model": "mock-gemma"}
    
    service.ChatWithGemma = mock_chat
    
    # Payload for RunChatLearn
    payload = {
        "turns": 4, # 2 cycles
        "turn_delay": 1,
        "model_hint": "hope-v1",
        "delegate_model_hint": "consult:gemini", # Trigger our logic
        "topic": "Testing Gemini Integration",
        "session_id": "test_session_1"
    }
    
    print("\n--- Starting RunChatLearn ---")
    result = await service.RunChatLearn(payload)
    print("\n--- Result ---")
    print(result)

if __name__ == "__main__":
    asyncio.run(run_test())
