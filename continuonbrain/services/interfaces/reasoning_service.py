"""
Reasoning Service Interface

Protocol definition for reasoning/planning services.
"""
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class IReasoningService(Protocol):
    """
    Protocol for reasoning/planning services.

    Implementations handle:
    - Symbolic search and planning
    - Context graph operations
    - Decision trace logging
    - Explainability
    """

    async def symbolic_search(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run symbolic/tree search for planning.

        Args:
            config: Search configuration dictionary

        Returns:
            Dictionary containing:
                - success: bool
                - plan: list - Sequence of actions
                - confidence: float - Plan confidence
                - search_stats: dict - Search statistics
        """
        ...

    def get_context_subgraph(
        self,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Retrieve context subgraph for reasoning.

        Args:
            session_id: Optional session ID for filtering
            tags: Optional list of tags to filter by
            depth: Graph traversal depth

        Returns:
            Dictionary containing:
                - nodes: list - Context nodes
                - edges: list - Context edges
                - summary: str - Graph summary
        """
        ...

    def get_decision_trace(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get decision trace for a session.

        Args:
            session_id: Session ID
            limit: Maximum number of decisions to return

        Returns:
            List of decision dictionaries:
                - timestamp: float
                - action: str
                - outcome: str
                - reason: str
                - confidence: float
        """
        ...

    def record_decision(
        self,
        session_id: str,
        action: str,
        outcome: str,
        reason: str,
        confidence: float = 0.5,
    ) -> Optional[str]:
        """
        Log a decision for traceability.

        Args:
            session_id: Session ID
            action: Action taken
            outcome: Outcome description
            reason: Reasoning explanation
            confidence: Confidence score

        Returns:
            Decision ID if recorded, None if failed
        """
        ...

    def ingest_episode(
        self,
        episode_data: Dict[str, Any],
    ) -> bool:
        """
        Ingest episode data into context graph.

        Args:
            episode_data: Episode data dictionary

        Returns:
            True if ingestion was successful
        """
        ...

    def query_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query context graph with semantic search.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of relevant context items
        """
        ...

    def is_available(self) -> bool:
        """Check if reasoning service is available."""
        ...
