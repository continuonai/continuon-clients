import pytest
import json
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
    storage_dir = REPO_ROOT / "tmp" / "test_dedup"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True)
    yield storage_dir
    # cleanup
    # shutil.rmtree(storage_dir)

def test_active_gating_deduplication(temp_storage):
    """Verify that similar questions are not duplicated."""
    logger = ExperienceLogger(temp_storage)
    
    # Mock the encoder to return predictable results
    with patch('continuonbrain.services.experience_logger.get_encoder') as mock_get:
        mock_encoder = MagicMock()
        mock_get.return_value = mock_encoder
        
        # Simple mock encoding: same string = same vector
        def mock_encode(text, **kwargs):
            if "move" in text.lower():
                return [1.0, 0.0]
            return [0.0, 1.0]
        
        mock_encoder.encode.side_effect = mock_encode
        
        # 1. Log first conversation
        logger.log_conversation("How do I move?", "Use the drive API.", "hope", 0.9)
        
        # 2. Log very similar conversation
        logger.log_conversation("Tell me how to move", "Use the drive API.", "hope", 0.9)
        
        # 3. Verify only 1 line in jsonl
        lines = logger.conversations_file.read_text().splitlines()
        assert len(lines) == 1
        
        # 4. Log distinct conversation
        logger.log_conversation("What is your name?", "Continuon.", "hope", 0.9)
        lines = logger.conversations_file.read_text().splitlines()
        assert len(lines) == 2
