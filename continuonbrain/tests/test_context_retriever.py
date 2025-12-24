import pytest
from continuonbrain.core.context_graph_models import Node, Edge
from continuonbrain.core.context_graph_networkx import NetworkXContextStore
from continuonbrain.core.context_retriever import ContextRetriever

class TestContextRetriever:
    def setup_method(self):
        self.store = NetworkXContextStore()
        # Create a simple graph
        # A -> B -> C
        # A -> D
        self.store.add_node(Node(id="A", type="concept", name="A"))
        self.store.add_node(Node(id="B", type="concept", name="B"))
        self.store.add_node(Node(id="C", type="concept", name="C"))
        self.store.add_node(Node(id="D", type="concept", name="D"))
        
        self.store.add_edge(Edge(id="e1", source="A", target="B", type="causal", confidence=0.9))
        self.store.add_edge(Edge(id="e2", source="B", target="C", type="temporal", confidence=0.8))
        self.store.add_edge(Edge(id="e3", source="A", target="D", type="weak", confidence=0.2))
        
        self.retriever = ContextRetriever(self.store)

    def test_expand_neighborhood(self):
        subgraph = self.retriever.expand_neighborhood(["A"], depth=1)
        # Should contain A, B, D
        ids = [n.id for n in subgraph]
        assert "A" in ids
        assert "B" in ids
        assert "D" in ids
        assert "C" not in ids

    def test_confidence_filter(self):
        subgraph = self.retriever.expand_neighborhood(["A"], depth=1, min_confidence=0.5)
        ids = [n.id for n in subgraph]
        assert "B" in ids
        assert "D" not in ids # Low confidence
