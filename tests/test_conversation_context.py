import pytest
import shutil
from pathlib import Path
from continuonbrain.services.brain_service import BrainService

@pytest.fixture
def temp_storage(tmp_path):
    storage_dir = tmp_path / "continuonbrain_test"
    storage_dir.mkdir()
    return storage_dir

def test_multi_turn_persistence(temp_storage):
    """Verify that conversation context is maintained across service instances."""
    # 1. Initialize first service instance
    service1 = BrainService(config_dir=str(temp_storage), prefer_real_hardware=False)
    session_id = "test_session_123"
    
    # 2. Add a message
    service1.ChatWithGemma("My name is Craig.", history=[], session_id=session_id)
    
    # 3. Simulate restart by creating a new service instance
    service2 = BrainService(config_dir=str(temp_storage), prefer_real_hardware=False)
    
    # 4. Ask follow-up
    # We mock the LLM call to verify history is passed
    import unittest.mock as mock
    with unittest.mock.patch.object(service2.gemma_chat, 'chat') as mock_chat:
        mock_chat.return_value = "Nice to meet you, Craig."
        
        service2.ChatWithGemma("What is my name?", history=[], session_id=session_id)
        
        # 5. Verify history was passed to the LLM
        args, kwargs = mock_chat.call_args
        passed_history = kwargs.get('history')
        assert passed_history is not None
        # Should contain: User: My name is Craig, Assistant: [mock response from s1]
        assert any("My name is Craig" in msg["content"] for msg in passed_history)

def test_session_pruning(temp_storage):
    """Verify that history is capped at 10 turns (5 full rounds)."""
    service = BrainService(config_dir=str(temp_storage), prefer_real_hardware=False)
    session_id = "prune_test"
    
    # Add 15 messages (7.5 turns)
    for i in range(15):
        service.session_store.add_message(session_id, "user", f"msg {i}")
        
    history = service.session_store.get_history(session_id)
    assert len(history) == 10
    assert history[-1]["content"] == "msg 14"
    assert history[0]["content"] == "msg 5"
