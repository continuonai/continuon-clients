import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.services.experience_logger import get_encoder

def test_lazy_loading():
    """Verify that the encoder is only loaded when requested."""
    # Note: This test might be tricky if other tests already loaded the encoder
    # We use a mock to verify the import and initialization
    
    with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
        # Reset the global encoder state for testing
        import continuonbrain.services.experience_logger as logger_mod
        logger_mod._encoder = None
        
        # 1. Access encoder
        encoder = get_encoder()
        
        # 2. Verify mock was called
        mock_transformer.assert_called_once_with('all-MiniLM-L6-v2')
        assert encoder is not None

def test_encoding_fallback():
    """Verify that get_encoder returns None if sentence-transformers is missing."""
    with patch('continuonbrain.services.experience_logger._encoder', None):
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            encoder = get_encoder()
            assert encoder is None
