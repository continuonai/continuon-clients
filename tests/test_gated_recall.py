import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.services.experience_logger import ExperienceLogger
from continuonbrain.services.agent_hope import HOPEAgent

@pytest.fixture
def temp_storage():
    storage_dir = REPO_ROOT / "tmp" / "test_gated_recall"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True)
    yield storage_dir

def test_gated_recall_validation_required(temp_storage):
    """Verify that unvalidated memories cannot be used by hope_brain directly."""
    logger = ExperienceLogger(temp_storage)
    
    # 1. Log a memory but don't validate it
    logger.log_conversation("test q", "test a", "llm", 0.9)
    
    # 2. Setup mock brain
    mock_brain = MagicMock()
    mock_col = MagicMock()
    mock_col._state = MagicMock()
    mock_col.last_confidence = 0.9 # High confidence
    mock_brain.columns = [mock_col]
    mock_brain.active_column_idx = 0
    
    agent = HOPEAgent(mock_brain)
    
    # 3. Check if can answer (should be false because it's not in validated store)
    # Actually, can_answer currently doesn't check retrieval.
    # The spec says: "Restrict direct HOPE responses (agent='hope_brain') to memories that have a validated: true status."
    # This logic usually happens in BrainService.ChatWithGemma.
    
    # Wait, spec says: "Update HOPEAgent.can_answer to check the validation status of retrieved memories."
    # Let's see how HOPEAgent.can_answer works.
