"""
Benchmark Tracker - Tracks model performance over time.

Stores benchmark results in SQLite for historical analysis and
automatic "beat previous best" progression tracking.
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """A single benchmark measurement."""
    benchmark_id: str
    category: str  # reasoning, tool_use, facts, system
    metric_name: str
    value: float
    model_name: str
    model_version: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metadata"] = json.dumps(d.get("metadata") or {})
        return d


@dataclass
class BenchmarkComparison:
    """Comparison of current vs best benchmark."""
    metric_name: str
    current_value: float
    best_value: float
    best_timestamp: float
    improvement: float  # positive = better
    is_new_best: bool
    higher_is_better: bool


class BenchmarkTracker:
    """
    Tracks benchmark results over time with SQLite storage.

    Features:
    - Store benchmark results with metadata
    - Track "best" score per metric
    - Analyze trends (improving/degrading)
    - Compare current vs best
    """

    # Metrics where higher is better
    HIGHER_IS_BETTER = {
        "hope_eval_accuracy",
        "facts_eval_accuracy",
        "tool_accuracy",
        "response_coherence",
        "curiosity_novelty",
        "memory_retention",
    }

    # Metrics where lower is better
    LOWER_IS_BETTER = {
        "inference_latency_ms",
        "tool_latency_ms",
        "memory_usage_mb",
        "loss",
    }

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("/opt/continuonos/brain/benchmarks.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        logger.info(f"BenchmarkTracker initialized: {self.db_path}")

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS benchmarks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        benchmark_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        model_name TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metric_name
                    ON benchmarks(metric_name)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON benchmarks(timestamp)
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS best_scores (
                        metric_name TEXT PRIMARY KEY,
                        value REAL NOT NULL,
                        benchmark_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    def record(self, result: BenchmarkResult) -> BenchmarkComparison:
        """
        Record a benchmark result and compare to best.

        Returns comparison showing if this is a new best.
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                # Insert result
                conn.execute("""
                    INSERT INTO benchmarks
                    (benchmark_id, category, metric_name, value, model_name,
                     model_version, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.benchmark_id,
                    result.category,
                    result.metric_name,
                    result.value,
                    result.model_name,
                    result.model_version,
                    result.timestamp,
                    json.dumps(result.metadata or {}),
                ))

                # Get current best
                cursor = conn.execute("""
                    SELECT value, timestamp FROM best_scores
                    WHERE metric_name = ?
                """, (result.metric_name,))
                row = cursor.fetchone()

                higher_is_better = result.metric_name in self.HIGHER_IS_BETTER

                if row is None:
                    # First result - it's the best by default
                    best_value = result.value
                    best_timestamp = result.timestamp
                    is_new_best = True
                    improvement = 0.0
                else:
                    best_value, best_timestamp = row

                    if higher_is_better:
                        is_new_best = result.value > best_value
                        improvement = result.value - best_value
                    else:
                        is_new_best = result.value < best_value
                        improvement = best_value - result.value

                # Update best if new best
                if is_new_best:
                    conn.execute("""
                        INSERT OR REPLACE INTO best_scores
                        (metric_name, value, benchmark_id, model_name,
                         model_version, timestamp, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        result.metric_name,
                        result.value,
                        result.benchmark_id,
                        result.model_name,
                        result.model_version,
                        result.timestamp,
                    ))
                    logger.info(
                        f"New best for {result.metric_name}: {result.value:.4f} "
                        f"(prev: {best_value:.4f}, improvement: {improvement:+.4f})"
                    )

                conn.commit()

                return BenchmarkComparison(
                    metric_name=result.metric_name,
                    current_value=result.value,
                    best_value=best_value if not is_new_best else result.value,
                    best_timestamp=best_timestamp if not is_new_best else result.timestamp,
                    improvement=improvement,
                    is_new_best=is_new_best,
                    higher_is_better=higher_is_better,
                )

            finally:
                conn.close()

    def get_best(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get best score for a metric."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute("""
                    SELECT value, benchmark_id, model_name, model_version, timestamp
                    FROM best_scores WHERE metric_name = ?
                """, (metric_name,))
                row = cursor.fetchone()
                if row:
                    return {
                        "metric_name": metric_name,
                        "value": row[0],
                        "benchmark_id": row[1],
                        "model_name": row[2],
                        "model_version": row[3],
                        "timestamp": row[4],
                    }
                return None
            finally:
                conn.close()

    def get_all_bests(self) -> List[Dict[str, Any]]:
        """Get all best scores."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute("""
                    SELECT metric_name, value, benchmark_id, model_name,
                           model_version, timestamp
                    FROM best_scores ORDER BY metric_name
                """)
                return [
                    {
                        "metric_name": row[0],
                        "value": row[1],
                        "benchmark_id": row[2],
                        "model_name": row[3],
                        "model_version": row[4],
                        "timestamp": row[5],
                    }
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    def get_history(
        self,
        metric_name: str,
        limit: int = 100,
        since_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Get historical values for a metric."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                if since_timestamp:
                    cursor = conn.execute("""
                        SELECT value, timestamp, model_name, model_version, metadata
                        FROM benchmarks
                        WHERE metric_name = ? AND timestamp >= ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (metric_name, since_timestamp, limit))
                else:
                    cursor = conn.execute("""
                        SELECT value, timestamp, model_name, model_version, metadata
                        FROM benchmarks
                        WHERE metric_name = ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (metric_name, limit))

                return [
                    {
                        "value": row[0],
                        "timestamp": row[1],
                        "model_name": row[2],
                        "model_version": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {},
                    }
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    def get_trend(self, metric_name: str, window: int = 10) -> Dict[str, Any]:
        """
        Analyze trend for a metric over recent samples.

        Returns trend direction and statistics.
        """
        history = self.get_history(metric_name, limit=window)
        if len(history) < 2:
            return {
                "metric_name": metric_name,
                "trend": "insufficient_data",
                "samples": len(history),
            }

        values = [h["value"] for h in history]

        # Simple trend: compare first half to second half
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(values[mid:]) / (len(values) - mid) if len(values) > mid else 0

        higher_is_better = metric_name in self.HIGHER_IS_BETTER

        if higher_is_better:
            if second_half_avg > first_half_avg * 1.01:
                trend = "improving"
            elif second_half_avg < first_half_avg * 0.99:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            if second_half_avg < first_half_avg * 0.99:
                trend = "improving"
            elif second_half_avg > first_half_avg * 1.01:
                trend = "degrading"
            else:
                trend = "stable"

        return {
            "metric_name": metric_name,
            "trend": trend,
            "samples": len(values),
            "latest_value": values[0],
            "avg_value": sum(values) / len(values),
            "min_value": min(values),
            "max_value": max(values),
            "first_half_avg": first_half_avg,
            "second_half_avg": second_half_avg,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get overall benchmark summary."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                # Count total benchmarks
                cursor = conn.execute("SELECT COUNT(*) FROM benchmarks")
                total_benchmarks = cursor.fetchone()[0]

                # Count unique metrics
                cursor = conn.execute("SELECT COUNT(DISTINCT metric_name) FROM benchmarks")
                unique_metrics = cursor.fetchone()[0]

                # Get latest benchmark time
                cursor = conn.execute("SELECT MAX(timestamp) FROM benchmarks")
                latest_timestamp = cursor.fetchone()[0]

                # Get category breakdown
                cursor = conn.execute("""
                    SELECT category, COUNT(*) FROM benchmarks
                    GROUP BY category
                """)
                categories = {row[0]: row[1] for row in cursor.fetchall()}

                # Get all bests
                bests = self.get_all_bests()

                return {
                    "total_benchmarks": total_benchmarks,
                    "unique_metrics": unique_metrics,
                    "latest_timestamp": latest_timestamp,
                    "latest_datetime": datetime.fromtimestamp(latest_timestamp).isoformat() if latest_timestamp else None,
                    "categories": categories,
                    "best_scores": bests,
                }
            finally:
                conn.close()

    def clear_history(self, before_timestamp: Optional[float] = None) -> int:
        """Clear old benchmark history (keeps best scores)."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                if before_timestamp:
                    cursor = conn.execute(
                        "DELETE FROM benchmarks WHERE timestamp < ?",
                        (before_timestamp,)
                    )
                else:
                    cursor = conn.execute("DELETE FROM benchmarks")
                count = cursor.rowcount
                conn.commit()
                logger.info(f"Cleared {count} benchmark records")
                return count
            finally:
                conn.close()
