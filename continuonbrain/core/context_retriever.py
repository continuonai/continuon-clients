from typing import List, Set, Optional, Dict
from .context_store_interface import ContextStore
from .context_graph_models import Node, Edge

class ContextRetriever:
    def __init__(self, store: ContextStore):
        self.store = store

    def expand_neighborhood(self, seed_node_ids: List[str], depth: int = 1, min_confidence: float = 0.0) -> List[Node]:
        """
        Expand from seed nodes to find the relevant subgraph.
        """
        visited_ids = set(seed_node_ids)
        frontier = list(seed_node_ids)
        result_nodes: Dict[str, Node] = {}
        
        # Load seeds first
        for nid in seed_node_ids:
            node = self.store.get_node(nid)
            if node:
                result_nodes[nid] = node

        current_depth = 0
        while current_depth < depth and frontier:
            next_frontier = []
            for current_id in frontier:
                edges = self.store.get_outbound_edges(current_id)
                for edge in edges:
                    if edge.confidence < min_confidence:
                        continue
                    
                    target_id = edge.target
                    if target_id not in visited_ids:
                        target_node = self.store.get_node(target_id)
                        if target_node:
                            result_nodes[target_id] = target_node
                            visited_ids.add(target_id)
                            next_frontier.append(target_id)
                            
            current_depth += 1
            frontier = next_frontier
            
        return list(result_nodes.values())
