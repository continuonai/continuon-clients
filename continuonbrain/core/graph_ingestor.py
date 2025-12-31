import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from .context_graph_store import SQLiteContextStore
from .context_graph_models import Node, Edge

logger = logging.getLogger(__name__)


class GraphIngestor:
    def __init__(self, store: SQLiteContextStore, gemma_chat=None):
        self.store = store
        self.gemma_chat = gemma_chat

    def _generate_id(self, prefix: str, content: str) -> str:
        h = hashlib.md5(content.encode("utf-8")).hexdigest()
        return f"{prefix}/{h}"

    def _edge_salience(self, timestamp: Optional[str] = None, score: float = 1.0) -> Dict[str, Any]:
        ts = timestamp or datetime.utcnow().isoformat()
        return {"score": score, "decay_fn": "exp", "half_life_s": 3600, "last_updated": ts}

    def _merge_list(self, current: List[Any], incoming: List[Any]) -> List[Any]:
        merged = list(current or [])
        for item in incoming:
            if item not in merged:
                merged.append(item)
        return merged

    def _upsert_node(self, node: Node, episode_id: Optional[str] = None):
        existing = self.store.get_node(node.id)
        if existing:
            merged_attrs = dict(existing.attributes)
            for key, value in node.attributes.items():
                if isinstance(value, list) and isinstance(merged_attrs.get(key), list):
                    merged_attrs[key] = self._merge_list(merged_attrs.get(key, []), value)
                else:
                    merged_attrs[key] = value
            if episode_id:
                merged_attrs["rlds_episode_ids"] = self._merge_list(
                    merged_attrs.get("rlds_episode_ids", []),
                    [episode_id],
                )
            node.attributes = merged_attrs
            node.embedding = node.embedding or existing.embedding
            node.belief = node.belief or existing.belief
        elif episode_id:
            node.attributes["rlds_episode_ids"] = self._merge_list(
                node.attributes.get("rlds_episode_ids", []),
                [episode_id],
            )
        self.store.add_node(node)

    def _add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        provenance: Dict[str, Any],
        confidence: float = 1.0,
        scope: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Edge:
        edge_id = self._generate_id("edge", f"{source}->{target}:{edge_type}:{json.dumps(provenance, sort_keys=True)}")
        edge = Edge(
            id=edge_id,
            source=source,
            target=target,
            type=edge_type,
            scope=scope or {},
            provenance=provenance,
            confidence=confidence,
            salience=self._edge_salience(timestamp),
        )
        self.store.add_edge(edge)
        return edge

    def _extract_tool_calls(self, action: Dict[str, Any]) -> List[str]:
        tools: List[str] = []
        tool_calls = action.get("tool_calls", []) if isinstance(action, dict) else []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else None
            if name:
                tools.append(name)
        # Legacy single-tool field
        if isinstance(action, dict):
            legacy_name = action.get("tool_name") or action.get("tool")
            if legacy_name:
                tools.append(legacy_name)
        return tools

    def _collect_spans(
        self,
        step_metadata: Dict[str, Any],
        cms_spans: Optional[Dict[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        spans = step_metadata.get("cms_spans", []) if isinstance(step_metadata, dict) else []
        collected: List[Dict[str, Any]] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            span_id = span.get("id") or span.get("span_id")
            if cms_spans and span_id and span_id in cms_spans:
                merged = dict(cms_spans[span_id])
                merged.update(span)
                collected.append(merged)
            else:
                collected.append(span)
        return collected

    def ingest_episode(self, rlds_path: str, cms_spans: Optional[List[Dict[str, Any]]] = None):
        """
        Ingest a saved RLDS episode into the graph.
        """
        path = Path(rlds_path)
        if not path.exists():
            logger.error(f"RLDS path not found: {rlds_path}")
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            episode_id = data.get("episode_id", path.stem)
            agent_id = data.get("agent_id", "unknown")
            metadata = data.get("metadata", {}) or {}
            timestamp = data.get("timestamp") or metadata.get("timestamp") or datetime.utcnow().isoformat()
            tags = metadata.get("tags", [])
            
            # Create Episode Node
            ep_node = Node(
                id=f"episode/{episode_id}",
                type="episode",
                name=f"Episode {timestamp}",
                attributes={
                    "path": str(path),
                    "agent": agent_id,
                    "timestamp": timestamp,
                    "tags": tags,
                }
            )
            
            steps = data.get("steps", [])
            summary_text = f"Episode by {agent_id} at {timestamp}. "
            if steps:
                first_instruction = steps[0].get("observation", {}).get("instruction", "")
                summary_text += f"Goal: {first_instruction}"
            
            if self.gemma_chat:
                ep_node.embedding = self.gemma_chat.embed(summary_text)
                
            self._upsert_node(ep_node, ep_node.id)
            cms_span_lookup = {s.get("id") or s.get("span_id"): s for s in cms_spans or [] if isinstance(s, dict)}
            
            # Process Steps -> Intents, Tools, CMS spans
            for idx, step in enumerate(steps):
                observation = step.get("observation", {}) if isinstance(step, dict) else {}
                action = step.get("action", {}) if isinstance(step, dict) else {}
                step_metadata = step.get("step_metadata", {}) if isinstance(step, dict) else {}

                instruction = (
                    observation.get("instruction")
                    or step_metadata.get("language_instruction")
                    or step_metadata.get("intent")
                    or ""
                )
                intent_node_id = None
                if instruction:
                    intent_node_id = self._generate_id("intent", instruction)
                    intent_node = Node(
                        id=intent_node_id,
                        type="intent",
                        name=instruction[:120],
                        attributes={"tags": step_metadata.get("tags", [])},
                        embedding=self.gemma_chat.embed(instruction) if self.gemma_chat else None,
                        belief={"score": 0.5, "updated_at": timestamp},
                    )
                    self._upsert_node(intent_node, ep_node.id)
                    self._add_edge(
                        ep_node.id,
                        intent_node_id,
                        "membership",
                        provenance={"episode_id": ep_node.id, "step_index": idx},
                        timestamp=timestamp,
                    )

                # Tools -> tool nodes + tool_use edges
                for tool_name in self._extract_tool_calls(action):
                    tool_node = Node(
                        id=f"tool/{tool_name}",
                        type="tool",
                        name=tool_name,
                        attributes={"tags": ["tool"]},
                    )
                    self._upsert_node(tool_node, ep_node.id)
                    self._add_edge(
                        intent_node_id or ep_node.id,
                        tool_node.id,
                        "tool_use",
                        provenance={"episode_id": ep_node.id, "step_index": idx},
                        timestamp=timestamp,
                        confidence=0.8,
                    )

                # CMS spans / attention spans
                span_candidates = self._collect_spans(step_metadata, cms_span_lookup)
                for span in span_candidates:
                    span_id_val = span.get("id") or span.get("span_id") or f"span_{idx}"
                    span_node_id = f"span/{span_id_val}"
                    span_name = span.get("text") or span.get("summary") or span_id_val
                    span_node = Node(
                        id=span_node_id,
                        type="span",
                        name=span_name,
                        attributes={
                            "cms_span_ids": [span_id_val],
                            "tags": span.get("tags", []),
                        },
                        embedding=self.gemma_chat.embed(span_name) if self.gemma_chat else None,
                        belief={"score": span.get("salience", 0.6), "updated_at": timestamp},
                    )
                    self._upsert_node(span_node, ep_node.id)
                    self._add_edge(
                        ep_node.id,
                        span_node_id,
                        "membership",
                        provenance={"episode_id": ep_node.id, "step_index": idx, "cms_span_id": span_id_val},
                        timestamp=timestamp,
                        confidence=span.get("confidence", 0.8),
                    )
                    if intent_node_id:
                        self._add_edge(
                            intent_node_id,
                            span_node_id,
                            "assertion",
                            provenance={"episode_id": ep_node.id, "step_index": idx, "cms_span_id": span_id_val},
                            scope={"audience": span.get("audience")},
                            timestamp=timestamp,
                            confidence=span.get("confidence", 0.8),
                        )
            
            logger.info(f"Ingested episode {episode_id} into Context Graph")
            
        except Exception as e:
            logger.error(f"Failed to ingest episode {rlds_path}: {e}")
