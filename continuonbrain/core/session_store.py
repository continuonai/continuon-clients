import sqlite3
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SQLiteSessionStore:
    """
    Persistence layer for conversation sessions.
    Maintains a rolling history of messages per session.
    """
    def __init__(self, db_path: str, max_history: int = 10):
        self.db_path = db_path
        self.max_history = max_history
        self.conn = None

    def _get_conn(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def initialize_db(self):
        """Create the messages table if it doesn't exist."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Index for fast retrieval by session
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        
        conn.commit()
        logger.info(f"Session database initialized at {self.db_path}")

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the session history and prune old messages."""
        conn = self._get_conn()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, timestamp))
        
        # Prune older messages for this session to maintain max_history
        # We keep the last max_history messages
        cursor.execute("""
            DELETE FROM messages 
            WHERE id IN (
                SELECT id FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT -1 OFFSET ?
            )
        """, (session_id, self.max_history))
        
        conn.commit()

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieve message history for a session."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        query_limit = limit or self.max_history
        
        cursor.execute("""
            SELECT role, content FROM messages 
            WHERE session_id = ? 
            ORDER BY timestamp ASC 
            LIMIT ?
        """, (session_id, query_limit))
        
        return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]

    def clear_session(self, session_id: str):
        """Clear all messages for a session."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        logger.info(f"Cleared session history for: {session_id}")
