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

        # Create table with extended schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                conversation_id TEXT PRIMARY KEY,
                is_validated BOOLEAN NOT NULL,
                correction TEXT,
                rating INTEGER,
                tags TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        # Add new columns if upgrading from old schema
        try:
            cursor.execute("ALTER TABLE feedback ADD COLUMN rating INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE feedback ADD COLUMN tags TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        conn.commit()
        logger.info(f"Feedback database initialized at {self.db_path}")

    def add_feedback(
        self,
        conversation_id: str,
        is_validated: bool,
        correction: Optional[str] = None,
        rating: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Store or update user feedback for a conversation.

        Args:
            conversation_id: Unique identifier for the conversation
            is_validated: Whether the response was validated as correct
            correction: Optional correction text if response was wrong
            rating: Optional 1-5 rating scale
            tags: Optional list of tags (e.g., ["incorrect", "safety", "helpful"])
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()

        # Validate rating if provided
        if rating is not None and (rating < 1 or rating > 5):
            raise ValueError("Rating must be between 1 and 5")

        # Serialize tags to JSON
        tags_json = json.dumps(tags) if tags else None

        cursor.execute("""
            INSERT OR REPLACE INTO feedback
            (conversation_id, is_validated, correction, rating, tags, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (conversation_id, is_validated, correction, rating, tags_json, timestamp))

        conn.commit()
        logger.info(f"Feedback added for {conversation_id}: validated={is_validated}, rating={rating}")

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a database row to a feedback dictionary."""
        tags = None
        try:
            if row["tags"]:
                tags = json.loads(row["tags"])
        except (json.JSONDecodeError, TypeError):
            tags = None

        return {
            "conversation_id": row["conversation_id"],
            "is_validated": bool(row["is_validated"]),
            "correction": row["correction"],
            "rating": row["rating"],
            "tags": tags,
            "timestamp": row["timestamp"],
        }

    def get_feedback(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve feedback for a specific conversation."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback WHERE conversation_id = ?", (conversation_id,))
        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_dict(row)

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """Retrieve all feedback records."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback ORDER BY timestamp DESC")

        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def list_recent(self, limit: int = 50, validated_only: bool = False) -> List[Dict[str, Any]]:
        """
        List recent feedback entries.

        Args:
            limit: Maximum number of entries to return
            validated_only: If True, only return validated feedback

        Returns:
            List of feedback dictionaries ordered by timestamp (newest first)
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        if validated_only:
            cursor.execute(
                "SELECT * FROM feedback WHERE is_validated = 1 ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        else:
            cursor.execute(
                "SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )

        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate validation statistics including rating distribution."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM feedback")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as validated FROM feedback WHERE is_validated = 1")
        validated = cursor.fetchone()["validated"]

        cursor.execute("SELECT AVG(rating) as avg_rating FROM feedback WHERE rating IS NOT NULL")
        avg_rating_row = cursor.fetchone()
        avg_rating = avg_rating_row["avg_rating"] if avg_rating_row["avg_rating"] else None

        cursor.execute("SELECT COUNT(*) as rated FROM feedback WHERE rating IS NOT NULL")
        rated_count = cursor.fetchone()["rated"]

        # Rating distribution
        rating_dist = {}
        for i in range(1, 6):
            cursor.execute("SELECT COUNT(*) as cnt FROM feedback WHERE rating = ?", (i,))
            rating_dist[str(i)] = cursor.fetchone()["cnt"]

        return {
            "total_feedback_count": total,
            "validated_count": validated,
            "rejection_count": total - validated,
            "validation_rate": validated / total if total > 0 else 0.0,
            "rated_count": rated_count,
            "average_rating": round(avg_rating, 2) if avg_rating else None,
            "rating_distribution": rating_dist,
        }
