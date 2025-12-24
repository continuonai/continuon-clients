import networkx as nx
from typing import List, Optional
from .context_graph_models import Node, Edge
from .context_store_interface import ContextStore

class NetworkXContextStore(ContextStore):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.edges_map = {} # Store edge objects by ID

    def add_node(self, node: Node):
        self.graph.add_node(node.id, data=node)

    def get_node(self, node_id: str) -> Optional[Node]:
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]["data"]
        return None

    def add_edge(self, edge: Edge):
        self.graph.add_edge(edge.source, edge.target, id=edge.id)
        self.edges_map[edge.id] = edge

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        return self.edges_map.get(edge_id)

    def get_neighbors(self, node_id: str) -> List[Node]:
        if not self.graph.has_node(node_id):
            return []
        
        # Outgoing neighbors
        neighbors = []
        for neighbor_id in self.graph.successors(node_id):
            neighbors.append(self.graph.nodes[neighbor_id]["data"])
        return neighbors

    def get_nearest_nodes(self, query_embedding: List[float], limit: int = 5) -> List[Node]:
        import numpy as np
        q_vec = np.array(query_embedding)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        
        candidates = []
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]["data"]
            if node.embedding:
                vec = np.array(node.embedding)
                norm = np.linalg.norm(vec)
                if norm == 0:
                    continue
                sim = np.dot(q_vec, vec) / (q_norm * norm)
                candidates.append((node, sim))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:limit]]

    def get_outbound_edges(self, node_id: str) -> List[Edge]:
        if not self.graph.has_node(node_id):
            return []
        
        edges = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            edge_id = data.get("id")
            if edge_id and edge_id in self.edges_map:
                edges.append(self.edges_map[edge_id])
        return edges


