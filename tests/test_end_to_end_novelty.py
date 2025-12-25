import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.services.brain_service import BrainService

@pytest.fixture
def test_env():
    config_dir = REPO_ROOT / "tmp" / "test_e2e_novelty"
    if config_dir.exists():
        shutil.rmtree(config_dir)
    config_dir.mkdir(parents=True)
    yield config_dir

def test_novelty_fallback_prefix(test_env):
    """Verify that low HOPE confidence triggers the transparent surprise prefix."""
    service = BrainService(config_dir=str(test_env), prefer_real_hardware=False)
    
    # 1. Setup mock brain with high novelty (low confidence)
    mock_brain = MagicMock()
    mock_col = MagicMock()
    mock_col._state = MagicMock()
    # hope_confidence = 0.4 triggers prefix (threshold 0.6, untrained floor 0.1)
    mock_col.last_confidence = 0.4 
    mock_brain.columns = [mock_col]
    mock_brain.active_column_idx = 0
    service.hope_brain = mock_brain
    
    # 2. Mock the LLM backend
    mock_chat = MagicMock()
    mock_chat.chat.return_value = "This is my reasoning."
    service.gemma_chat = mock_chat
    
    # 3. Trigger chat
    result = service.ChatWithGemma("gibberish novel query", history=[])
    
    # 4. Verify prefix
    assert "[Surprise: 0.60]" in result["response"]
    assert "novel situation" in result["response"]
    assert "This is my reasoning." in result["response"]
    assert result["agent"] == "llm_only" # Since it fell back to LLM
