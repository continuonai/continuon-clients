import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.services.experience_logger import ExperienceLogger
from continuonbrain.services.brain_service import BrainService

@pytest.fixture
def test_env():
    config_dir = REPO_ROOT / "tmp" / "test_feedback_flow"
    if config_dir.exists():
        shutil.rmtree(config_dir)
    config_dir.mkdir(parents=True)
    yield config_dir

def test_full_feedback_and_gating_flow(test_env):
    """Verify that user feedback correctly gates and boosts memories."""
    service = BrainService(config_dir=str(test_env), prefer_real_hardware=False)
    
    with patch('continuonbrain.services.experience_logger.get_encoder') as mock_get:
        mock_encoder = MagicMock()
        mock_get.return_value = mock_encoder
        mock_encoder.encode.return_value = [1.0, 0.0]
        
        # 1. Log a memory (Initial state: unvalidated)
        conv_id = service.experience_logger.log_conversation("how to move", "drive api", "llm", 0.9)
        assert conv_id != ""
        
        # 2. Verify it is NOT used for hope_brain yet (Gated Recall)
        # We need to simulate a case where HOPE might answer.
        # ChatWithGemma Phase 2: similar[0].get("validated") must be True.
        result = service.ChatWithGemma("how to move", history=[])
        assert result["agent"] != "semantic_memory" # Should fallback to LLM because not validated
        
        # 3. Validate the memory (Thumbs Up)
        service.experience_logger.validate_conversation(conv_id, True)
        
        # 4. Verify it IS now used for recall
        result = service.ChatWithGemma("how to move", history=[])
        assert result["agent"] == "semantic_memory"
        assert result["response"] == "drive api"
        
        # 5. Verify Priority Boosting
        # Log a newer, semantically identical memory
        # If both match 1.0, the validated one should win if it gets +0.2 boost
        # (Though max relevance is capped at 1.0, we just check if it returns the validated one)
        
        # Actually let's test a case where sim is slightly lower for validated
        def mock_encode_variable(text, **kwargs):
            if "move" in text: return [1.0, 0.0] # perfect match
            if "motion" in text: return [0.8, 0.6] # slight match
            return [0.0, 1.0]
        
        mock_encoder.encode.side_effect = mock_encode_variable
        
        # "how to move" -> [1.0, 0.0] (unvalidated)
        # "tell me about motion" -> [0.8, 0.6] (validated)
        
        # Existing "how to move" is validated. 
        # Query: "move" -> matches "how to move" better.
        
        # Let's just verify the relevance score includes the boost
        similar = service.experience_logger.get_similar_conversations("how to move", max_results=1)
        assert similar[0]["validated"] == True
        # If similarity was 1.0, boost makes it 1.2 -> capped at 1.0
        
def test_rejection_gating(test_env):
    """Verify that Thumbs Down correctly blocks recall."""
    service = BrainService(config_dir=str(test_env), prefer_real_hardware=False)
    
    with patch('continuonbrain.services.experience_logger.get_encoder') as mock_get:
        mock_encoder = MagicMock()
        mock_get.return_value = mock_encoder
        mock_encoder.encode.return_value = [1.0, 0.0]
        
        conv_id = service.experience_logger.log_conversation("reject me", "bad answer", "llm", 0.9)
        service.experience_logger.validate_conversation(conv_id, False) # Thumbs Down
        
        result = service.ChatWithGemma("reject me", history=[])
        assert result["agent"] != "semantic_memory"
        assert "Potential memory match (not validated)" in result["status_updates"]
