import pytest
import os
import sqlite3
from pathlib import Path
from continuonbrain.core.feedback_store import SQLiteFeedbackStore

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_feedback.db"
    return str(db_path)

def test_initialize_db(temp_db):
    store = SQLiteFeedbackStore(temp_db)
    store.initialize_db()
    
    # Verify table exists
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
    assert cursor.fetchone() is not None
    conn.close()

def test_add_get_feedback(temp_db):
    store = SQLiteFeedbackStore(temp_db)
    store.initialize_db()
    
    conv_id = "test_conv_123"
    store.add_feedback(conv_id, True, "Good job")
    
    fb = store.get_feedback(conv_id)
    assert fb is not None
    assert fb["conversation_id"] == conv_id
    assert fb["is_validated"] is True
    assert fb["correction"] == "Good job"

def test_update_feedback(temp_db):
    store = SQLiteFeedbackStore(temp_db)
    store.initialize_db()
    
    conv_id = "test_conv_123"
    store.add_feedback(conv_id, True)
    store.add_feedback(conv_id, False, "Incorrect")
    
    fb = store.get_feedback(conv_id)
    assert fb["is_validated"] is False
    assert fb["correction"] == "Incorrect"

def test_get_summary(temp_db):
    store = SQLiteFeedbackStore(temp_db)
    store.initialize_db()
    
    store.add_feedback("c1", True)
    store.add_feedback("c2", True)
    store.add_feedback("c3", False)
    
    summary = store.get_summary()
    assert summary["total_feedback_count"] == 3
    assert summary["validated_count"] == 2
    assert summary["rejection_count"] == 1
    assert summary["validation_rate"] == 2/3
