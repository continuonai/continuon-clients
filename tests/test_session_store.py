import pytest
import os
import sqlite3
from continuonbrain.core.session_store import SQLiteSessionStore

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_sessions.db"
    return str(db_path)

def test_initialize_db(temp_db):
    store = SQLiteSessionStore(temp_db)
    store.initialize_db()
    
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
    assert cursor.fetchone() is not None
    conn.close()

def test_add_get_history(temp_db):
    store = SQLiteSessionStore(temp_db)
    store.initialize_db()
    
    session_id = "test_s1"
    store.add_message(session_id, "user", "hello")
    store.add_message(session_id, "assistant", "hi there")
    
    history = store.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "hi there"

def test_pruning(temp_db):
    max_history = 3
    store = SQLiteSessionStore(temp_db, max_history=max_history)
    store.initialize_db()
    
    session_id = "test_s1"
    for i in range(5):
        store.add_message(session_id, "user", f"msg {i}")
        
    history = store.get_history(session_id)
    assert len(history) == max_history
    # Should be the last 3 messages: msg 2, msg 3, msg 4
    assert history[0]["content"] == "msg 2"
    assert history[2]["content"] == "msg 4"

def test_clear_session(temp_db):
    store = SQLiteSessionStore(temp_db)
    store.initialize_db()
    
    session_id = "test_s1"
    store.add_message(session_id, "user", "hello")
    store.clear_session(session_id)
    
    history = store.get_history(session_id)
    assert len(history) == 0
