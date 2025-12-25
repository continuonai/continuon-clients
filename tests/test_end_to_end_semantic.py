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
    config_dir = REPO_ROOT / "tmp" / "test_e2e_semantic"
    if config_dir.exists():
        shutil.rmtree(config_dir)
    config_dir.mkdir(parents=True)
    yield config_dir

def test_end_to_end_semantic_flow(test_env):
    """Verify Q1 -> Learn -> Q2 (similar) -> Recall flow."""
    service = BrainService(config_dir=str(test_env), prefer_real_hardware=False)
    
    with patch('continuonbrain.services.experience_logger.get_encoder') as mock_get:
        mock_encoder = MagicMock()
        mock_get.return_value = mock_encoder
        
        # Mock encoding: predictable vectors
        def mock_encode(text, **kwargs):
            if "move" in text.lower() or "motion" in text.lower():
                return [1.0, 0.0]
            return [0.0, 1.0]
        
        mock_encoder.encode.side_effect = mock_encode
        
        # 1. Ask Q1 and ensure it's learned
        # Manually log since ChatWithGemma usually handles the LLM call
        service.experience_logger.log_conversation("How do I move?", "Drive using the API.", "llm", 0.9)
        
        # 2. Ask Q2 (semantically similar)
        # Note: ChatWithGemma is async internally or calls sync Chat
        # We'll call it directly
        result = service.ChatWithGemma("Tell me about motion", history=[])
        
        # 3. Verify recall
        assert result["agent"] == "semantic_memory"
        assert result["response"] == "Drive using the API."
        assert result["semantic_confidence"] > 0.9
