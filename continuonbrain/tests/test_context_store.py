import pytest
import sqlite3
import json
from datetime import datetime
from continuonbrain.core.context_graph_models import Node, Edge
from continuonbrain.core.context_graph_store import SQLiteContextStore
from continuonbrain.core.graph_ingestor import GraphIngestor
from continuonbrain.core.context_retriever import ContextRetriever

class TestSQLiteContextStore:
    def setup_method(self):
        # Use in-memory SQLite for testing
        self.store = SQLiteContextStore(":memory:")
        self.store.initialize_db()

    def test_add_and_get_node(self):
        node = Node(
            id="node_1",
            type="entity",
            name="test_entity",
            attributes={"tag": "value"},
            embedding=[0.1, 0.2]
        )
        self.store.add_node(node)
        
        retrieved = self.store.get_node("node_1")
        assert retrieved is not None
        assert retrieved.id == "node_1"
        assert retrieved.name == "test_entity"
        assert retrieved.attributes == {"tag": "value"}
        assert retrieved.embedding == [0.1, 0.2]

    def test_add_and_get_edge(self):
        node1 = Node(id="n1", type="t", name="n1")
        node2 = Node(id="n2", type="t", name="n2")
        self.store.add_node(node1)
        self.store.add_node(node2)

        edge = Edge(
            id="e1",
            source="n1",
            target="n2",
            type="causal",
            confidence=0.8,
            scope={"time": "now"}
        )
        self.store.add_edge(edge)

        retrieved = self.store.get_edge("e1")
        assert retrieved is not None
        assert retrieved.source == "n1"
        assert retrieved.target == "n2"
        assert retrieved.confidence == 0.8
        assert retrieved.scope == {"time": "now"}

    def test_get_neighbors(self):
        n1 = Node(id="n1", type="t", name="n1")
        n2 = Node(id="n2", type="t", name="n2")
        self.store.add_node(n1)
        self.store.add_node(n2)
        
        e1 = Edge(id="e1", source="n1", target="n2", type="link")
        self.store.add_edge(e1)

        neighbors = self.store.get_neighbors("n1")
        assert len(neighbors) == 1
        assert neighbors[0].id == "n2"

from continuonbrain.core.context_graph_networkx import NetworkXContextStore

class TestNetworkXContextStore:
    def setup_method(self):
        self.store = NetworkXContextStore()

    def test_add_and_get_node(self):
        node = Node(id="nx_1", type="t", name="nx_1")
        self.store.add_node(node)
        assert self.store.get_node("nx_1").name == "nx_1"

    def test_add_and_get_edge(self):
        n1 = Node(id="n1", type="t", name="n1")
        n2 = Node(id="n2", type="t", name="n2")
        self.store.add_node(n1)
        self.store.add_node(n2)
        
        edge = Edge(id="e1", source="n1", target="n2", type="link")
        self.store.add_edge(edge)
        
        assert self.store.get_edge("e1").source == "n1"

    def test_get_neighbors(self):
        n1 = Node(id="n1", type="t", name="n1")
        n2 = Node(id="n2", type="t", name="n2")
        self.store.add_node(n1)
        self.store.add_node(n2)
        
        edge = Edge(id="e1", source="n1", target="n2", type="link")
        self.store.add_edge(edge)
        
        neighbors = self.store.get_neighbors("n1")
        assert len(neighbors) == 1
        assert neighbors[0].id == "n2"


def test_graph_ingestor_builds_typed_nodes_and_edges(tmp_path):
    store = SQLiteContextStore(":memory:")
    store.initialize_db()
    ingestor = GraphIngestor(store, gemma_chat=None)

    episode_payload = {
        "episode_id": "ep_ingest",
        "agent_id": "tester",
        "timestamp": "2024-01-01T00:00:00Z",
        "steps": [
            {
                "observation": {"instruction": "Pick up the block"},
                "action": {"tool_calls": [{"name": "gripper"}]},
                "step_metadata": {"cms_spans": [{"id": "span-1", "text": "block grasp", "confidence": 0.9}]},
            }
        ],
    }
    episode_path = tmp_path / "episode.json"
    episode_path.write_text(json.dumps(episode_payload))

    ingestor.ingest_episode(str(episode_path), cms_spans=[{"id": "span-1", "tags": ["cms"]}])

    nodes = store.list_nodes(limit=20)
    node_types = {node.type for node in nodes}
    assert "episode" in node_types
    assert "intent" in node_types
    assert "tool" in node_types
    assert any(node.type == "span" for node in nodes)

    edges = store.list_edges()
    assert any(edge.type == "tool_use" for edge in edges)
    assert any(edge.provenance.get("cms_span_id") == "span-1" for edge in edges)

    retriever = ContextRetriever(store)
    episode_ids = [node.id for node in nodes if node.type == "episode"]
    subgraph = retriever.build_subgraph(episode_ids, depth=2)
    sub_nodes = {node.type for node in subgraph["nodes"]}
    assert "tool" in sub_nodes
    assert "intent" in sub_nodes
