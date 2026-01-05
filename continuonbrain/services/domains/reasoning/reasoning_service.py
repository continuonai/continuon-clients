"""
Reasoning Service

Domain service for reasoning/planning functionality.
Wraps context graph and decision trace logging.
"""
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from continuonbrain.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class ReasoningService:
    """
    Reasoning domain service implementing IReasoningService.

    Handles:
    - Symbolic search and planning
    - Context graph operations
    - Decision trace logging
    - Explainability
    """

    def __init__(
        self,
        config_dir: str = "/opt/continuonos/brain",
        container: Optional["ServiceContainer"] = None,
        db_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize reasoning service.

        Args:
            config_dir: Configuration directory
            container: Service container for dependencies
            db_path: Path to context graph database
        """
        self.config_dir = Path(config_dir)
        self._container = container
        self.db_path = db_path or str(self.config_dir / "context_graph.db")

        self._context_store = None
        self._graph_ingestor = None
        self._decision_logger = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of context graph."""
        if self._initialized:
            return

        self._initialized = True

        try:
            from continuonbrain.core.context_graph_store import SQLiteContextStore
            from continuonbrain.core.graph_ingestor import GraphIngestor
            from continuonbrain.core.decision_trace_logger import DecisionTraceLogger

            self._context_store = SQLiteContextStore(self.db_path)
            self._graph_ingestor = GraphIngestor(self._context_store)
            self._decision_logger = DecisionTraceLogger(self._context_store)

            logger.info("Reasoning service initialized with context graph")

        except ImportError as e:
            logger.warning(f"Context graph modules not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize reasoning: {e}")

    async def symbolic_search(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run symbolic/tree search for planning."""
        # Placeholder for symbolic search implementation
        return {
            "success": False,
            "plan": [],
            "confidence": 0.0,
            "search_stats": {},
            "error": "Symbolic search not yet implemented",
        }

    def get_context_subgraph(
        self,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """Retrieve context subgraph for reasoning."""
        self._ensure_initialized()

        if not self._context_store:
            return {"nodes": [], "edges": [], "summary": "Context store not available"}

        try:
            # Query context store
            nodes = self._context_store.get_nodes(
                session_id=session_id,
                tags=tags,
                limit=100,
            )

            edges = []
            for node in nodes:
                node_edges = self._context_store.get_edges(
                    source_id=node.id,
                    depth=depth,
                )
                edges.extend(node_edges)

            return {
                "nodes": [n.to_dict() for n in nodes],
                "edges": [e.to_dict() for e in edges],
                "summary": f"Retrieved {len(nodes)} nodes and {len(edges)} edges",
            }

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return {"nodes": [], "edges": [], "summary": f"Error: {e}"}

    def get_decision_trace(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get decision trace for a session."""
        self._ensure_initialized()

        if not self._decision_logger:
            return []

        try:
            traces = self._decision_logger.get_traces(
                session_id=session_id,
                limit=limit,
            )
            return [t.to_dict() for t in traces]

        except Exception as e:
            logger.error(f"Failed to get decision trace: {e}")
            return []

    def record_decision(
        self,
        session_id: str,
        action: str,
        outcome: str,
        reason: str,
        confidence: float = 0.5,
    ) -> Optional[str]:
        """Log a decision for traceability."""
        self._ensure_initialized()

        if not self._decision_logger:
            return None

        try:
            decision_id = str(uuid.uuid4())

            self._decision_logger.log_decision(
                decision_id=decision_id,
                session_id=session_id,
                action=action,
                outcome=outcome,
                reason=reason,
                confidence=confidence,
                timestamp=time.time(),
            )

            return decision_id

        except Exception as e:
            logger.error(f"Failed to record decision: {e}")
            return None

    def ingest_episode(
        self,
        episode_data: Dict[str, Any],
    ) -> bool:
        """Ingest episode data into context graph."""
        self._ensure_initialized()

        if not self._graph_ingestor:
            return False

        try:
            self._graph_ingestor.ingest_episode(episode_data)
            return True

        except Exception as e:
            logger.error(f"Episode ingestion failed: {e}")
            return False

    def query_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query context graph with semantic search."""
        self._ensure_initialized()

        if not self._context_store:
            return []

        try:
            from continuonbrain.core.context_retriever import ContextRetriever

            retriever = ContextRetriever(self._context_store)
            results = retriever.query(query, top_k=top_k)

            return [r.to_dict() for r in results]

        except ImportError:
            logger.debug("ContextRetriever not available")
            return []
        except Exception as e:
            logger.error(f"Context query failed: {e}")
            return []

    def is_available(self) -> bool:
        """Check if reasoning service is available."""
        return True  # Always available with at least basic functionality

    def shutdown(self) -> None:
        """Shutdown reasoning service."""
        if self._context_store:
            try:
                self._context_store.close()
            except Exception:
                pass

        self._context_store = None
        self._graph_ingestor = None
        self._decision_logger = None
        self._initialized = False
