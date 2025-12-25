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
    storage_dir = REPO_ROOT / "tmp" / "test_merging"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True)
    yield storage_dir

def test_cluster_similar_memories(temp_storage):
    """Verify that similar memories are correctly grouped into clusters."""
    logger = ExperienceLogger(temp_storage)
    
    with patch('continuonbrain.services.experience_logger.get_encoder') as mock_get:
        mock_encoder = MagicMock()
        mock_get.return_value = mock_encoder
        
        # Mock encoding: predictable vectors
        # Cluster 1: move related
        # Cluster 2: camera related
        def mock_encode(text, **kwargs):
            if "move" in text.lower():
                return [1.0, 0.0]
            if "camera" in text.lower():
                return [0.0, 1.0]
            return [0.5, 0.5]
        
        mock_encoder.encode.side_effect = mock_encode
        
        # 1. Seed memories
        logger.log_conversation("How do I move?", "Drive forward.", "llm", 0.9)
        logger.log_conversation("Tell me about moving", "Use the drive API.", "llm", 0.8) # Similar to 1
        logger.log_conversation("Check the camera", "Vision active.", "llm", 0.9)
        logger.log_conversation("What can you see?", "Camera is on.", "llm", 0.7) # Similar to 3
        logger.log_conversation("Hello", "Hi!", "llm", 0.9) # Distinct
        
        # 2. Run clustering
        clusters = logger.cluster_similar_memories(threshold=0.9)
        
        # 3. Verify clusters
        # Expected: [[conv1, conv2], [conv3, conv4]]
        # "Hello" is distinct so it shouldn't be in a multi-item cluster (depending on implementation)
        assert len(clusters) >= 2
        
        # Check that move cluster contains 2 items
        move_cluster = [c for c in clusters if "move" in c[0]["question"].lower()][0]
        assert len(move_cluster) == 2
        
        # Check that camera cluster contains 2 items
        camera_cluster = [c for c in clusters if "camera" in c[0]["question"].lower() or "see" in c[0]["question"].lower()][0]
        assert len(camera_cluster) == 2
