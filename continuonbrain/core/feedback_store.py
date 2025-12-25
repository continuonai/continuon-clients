import sqlite3
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class SQLiteFeedbackStore:
    """
    Persistence layer for user validation feedback on robot responses.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def _get_conn(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def initialize_db(self):
        """Create the feedback table if it doesn't exist."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                conversation_id TEXT PRIMARY KEY,
                is_validated BOOLEAN NOT NULL,
                correction TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        conn.commit()
        logger.info(f"Feedback database initialized at {self.db_path}")

    def add_feedback(self, conversation_id: str, is_validated: bool, correction: Optional[str] = None):
        """Store or update user feedback for a conversation."""
        conn = self._get_conn()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO feedback (conversation_id, is_validated, correction, timestamp)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, is_validated, correction, timestamp))
        
        conn.commit()
        logger.info(f"Feedback added for {conversation_id}: validated={is_validated}")

    def get_feedback(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve feedback for a specific conversation."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback WHERE conversation_id = ?", (conversation_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            "conversation_id": row["conversation_id"],
            "is_validated": bool(row["is_validated"]),
            "correction": row["correction"],
            "timestamp": row["timestamp"]
        }

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """Retrieve all feedback records."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "conversation_id": row["conversation_id"],
                "is_validated": bool(row["is_validated"]),
                "correction": row["correction"],
                "timestamp": row["timestamp"]
            })
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate validation statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM feedback")
        total = cursor.fetchone()["total"]
        
        cursor.execute("SELECT COUNT(*) as validated FROM feedback WHERE is_validated = 1")
        validated = cursor.fetchone()["validated"]
        
        return {
            "total_feedback_count": total,
            "validated_count": validated,
            "rejection_count": total - validated,
            "validation_rate": validated / total if total > 0 else 0.0
        }
