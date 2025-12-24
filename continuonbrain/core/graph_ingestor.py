import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
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

    def ingest_episode(self, rlds_path: str):
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
            timestamp = data.get("timestamp")
            
            # Create Episode Node
            ep_node = Node(
                id=f"episode/{episode_id}",
                type="episode",
                name=f"Episode {timestamp}",
                attributes={
                    "path": str(path),
                    "agent": agent_id,
                    "timestamp": timestamp
                }
            )
            
            steps = data.get("steps", [])
            summary_text = f"Episode by {agent_id} at {timestamp}. "
            if steps:
                first_instruction = steps[0].get("observation", {}).get("instruction", "")
                summary_text += f"Goal: {first_instruction}"
            
            if self.gemma_chat:
                ep_node.embedding = self.gemma_chat.embed(summary_text)
                
            self.store.add_node(ep_node)
            
            # Process Goal
            if steps:
                instruction = steps[0].get("observation", {}).get("instruction", "")
                if instruction:
                    goal_id = self._generate_id("goal", instruction)
                    # Check if goal exists to avoid overwriting if we want to merge? 
                    # Store merge logic is "INSERT OR REPLACE", so attributes update.
                    # We might want to keep existing attributes or increment a counter.
                    # For now, simple add.
                    goal_node = Node(
                        id=goal_id,
                        type="goal",
                        name=instruction,
                        embedding=self.gemma_chat.embed(instruction) if self.gemma_chat else None
                    )
                    self.store.add_node(goal_node)
                    
                    # Link Episode to Goal
                    edge_id = self._generate_id("edge", f"{ep_node.id}->{goal_id}")
                    edge = Edge(
                        id=edge_id,
                        source=ep_node.id,
                        target=goal_node.id,
                        type="membership",
                        confidence=1.0,
                        provenance={"source": "rlds_ingest"}
                    )
                    self.store.add_edge(edge)
            
            logger.info(f"Ingested episode {episode_id} into Context Graph")
            
        except Exception as e:
            logger.error(f"Failed to ingest episode {rlds_path}: {e}")
