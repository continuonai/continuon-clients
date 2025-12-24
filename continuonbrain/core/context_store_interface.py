from abc import ABC, abstractmethod
from typing import List, Optional
from .context_graph_models import Node, Edge

class ContextStore(ABC):
    @abstractmethod
    def add_node(self, node: Node):
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        pass

    @abstractmethod
    def add_edge(self, edge: Edge):
        pass

    @abstractmethod
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str) -> List[Node]:
        pass

    @abstractmethod
    def get_nearest_nodes(self, query_embedding: List[float], limit: int = 5) -> List[Node]:
        pass

    @abstractmethod
    def get_outbound_edges(self, node_id: str) -> List[Edge]:
        pass
