import datetime
import uuid
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from continuonbrain.core.context_graph_models import Edge, Node
from continuonbrain.core.context_graph_store import SQLiteContextStore


class DecisionTraceLogger:
    """Lightweight helper to project decision traces into the context graph."""

    def __init__(
        self,
        store: SQLiteContextStore,
        *,
        session_node_provider: Optional[Callable[[str], str]] = None,
        default_session: str = "system",
    ) -> None:
        self.store = store
        self.session_node_provider = session_node_provider
        self.default_session = default_session

    def _now(self) -> str:
        return datetime.datetime.utcnow().isoformat()

    def _ensure_session_node(self, session_id: Optional[str]) -> str:
        sid = session_id or self.default_session
        if callable(self.session_node_provider):
            return self.session_node_provider(sid)

        node_id = f"context_session/{sid}"
        existing = self.store.get_node(node_id)
        if not existing:
            self.store.add_node(
                Node(
                    id=node_id,
                    type="context_session",
                    name=f"Session {sid}",
                    attributes={"session_id": sid, "tags": ["chat_session"]},
                )
            )
        return node_id

    def _edge_salience(self) -> Dict[str, Any]:
        return {"score": 1.0, "last_updated": self._now()}

    def _add_tool_edges(
        self,
        source: str,
        tools: Iterable[str],
        provenance: Dict[str, Any],
    ) -> None:
        for tool in tools:
            tool_id = f"tool/{tool}"
            self.store.add_node(
                Node(
                    id=tool_id,
                    type="tool",
                    name=tool,
                    attributes={"tags": ["tool"]},
                )
            )
            edge = Edge(
                id=f"edge/{uuid.uuid4()}",
                source=source,
                target=tool_id,
                type="tool_use",
                provenance=dict(provenance),
                salience=self._edge_salience(),
            )
            self.store.add_edge(edge)

    def log_action_plan(
        self,
        *,
        session_id: Optional[str],
        plan_text: str,
        actor: str,
        tools: Optional[Sequence[str]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record an action plan proposed by the HOPE/LLM orchestrator.

        Returns:
            The node id representing the plan/decision.
        """
        decision_id = f"decision/{uuid.uuid4()}"
        plan_node = Node(
            id=decision_id,
            type="decision",
            name=plan_text[:120] or "Action plan",
            attributes={
                "kind": "action_plan",
                "actor": actor,
                "tags": ["decision_trace"],
                "summary": plan_text,
            },
        )
        self.store.add_node(plan_node)

        provenance_payload = {"source": "llm_action_plan", "actor": actor, **(provenance or {})}
        session_node_id = self._ensure_session_node(session_id)
        edge = Edge(
            id=f"edge/{uuid.uuid4()}",
            source=session_node_id,
            target=decision_id,
            type="policy",
            provenance=provenance_payload,
            salience=self._edge_salience(),
            scope={"timestamp": self._now()},
        )
        self.store.add_edge(edge)

        if tools:
            self._add_tool_edges(
                source=decision_id,
                tools=[t for t in tools if t],
                provenance={**provenance_payload, "decision_id": decision_id},
            )
        return decision_id

    def log_policy_decision(
        self,
        *,
        session_id: Optional[str],
        action_ref: str,
        outcome: str,
        reason: str,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a policy or safety gate decision.

        Returns:
            The target node id used for the policy edge.
        """
        target_id = action_ref or f"decision/{uuid.uuid4()}"
        target_node = self.store.get_node(target_id)
        if not target_node:
            target_node = Node(
                id=target_id,
                type="decision",
                name=action_ref or "policy_decision",
                attributes={"kind": "policy_gate", "tags": ["decision_trace"]},
            )
            self.store.add_node(target_node)

        session_node_id = self._ensure_session_node(session_id)
        payload = {
            "source": "safety_kernel",
            "outcome": outcome,
            "reason": reason,
            **(provenance or {}),
        }
        edge = Edge(
            id=f"edge/{uuid.uuid4()}",
            source=session_node_id,
            target=target_id,
            type="policy",
            provenance=payload,
            salience=self._edge_salience(),
            scope={"timestamp": self._now()},
        )
        self.store.add_edge(edge)
        return target_id

    def log_human_feedback(
        self,
        *,
        session_id: Optional[str],
        action_ref: str,
        approved: bool,
        user_id: str,
        notes: str = "",
    ) -> str:
        """
        Record a human validation/rejection event for an action or plan.

        Returns:
            The action reference node id that the human decision is attached to.
        """
        target_id = action_ref or f"decision/{uuid.uuid4()}"
        target_node = self.store.get_node(target_id)
        if not target_node:
            target_node = Node(
                id=target_id,
                type="decision",
                name=action_ref or "action",
                attributes={"kind": "action_ref", "tags": ["decision_trace"]},
            )
            self.store.add_node(target_node)

        provenance = {
            "source": "human",
            "approved": approved,
            "user_id": user_id,
            "notes": notes,
        }
        session_node_id = self._ensure_session_node(session_id)
        edge = Edge(
            id=f"edge/{uuid.uuid4()}",
            source=session_node_id,
            target=target_id,
            type="policy",
            provenance=provenance,
            salience=self._edge_salience(),
            scope={"timestamp": self._now()},
        )
        self.store.add_edge(edge)
        return target_id
