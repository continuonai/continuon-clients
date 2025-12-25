import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.services.experience_logger import ExperienceLogger

@pytest.fixture
def temp_storage():
    storage_dir = REPO_ROOT / "tmp" / "test_retrieval"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True)
    yield storage_dir

def test_semantic_neighborhood_retrieval(temp_storage):
    """Verify that semantically similar questions are retrieved."""
    logger = ExperienceLogger(temp_storage)
    
    with patch('continuonbrain.services.experience_logger.get_encoder') as mock_get:
        mock_encoder = MagicMock()
        mock_get.return_value = mock_encoder
        
        # Mock encoding: pre-defined vectors for specific terms
        def mock_encode(text, **kwargs):
            if "move" in text.lower():
                return [1.0, 0.0]
            if "camera" in text.lower():
                return [0.0, 1.0]
            return [0.5, 0.5]
        
        mock_encoder.encode.side_effect = mock_encode
        
        # 1. Seed memories
        logger.log_conversation("How do I move?", "Drive forward.", "hope", 0.9)
        logger.log_conversation("Check the camera", "Vision active.", "hope", 0.9)
        
        # 2. Query for move-related
        results = logger.get_similar_conversations("Tell me about moving")
        assert len(results) >= 1
        assert "move" in results[0]["question"].lower()
        assert results[0]["relevance"] > 0.9
