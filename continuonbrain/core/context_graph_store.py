import sqlite3
import json
from typing import List, Optional, Any
from .context_graph_models import Node, Edge
from .context_store_interface import ContextStore

class SQLiteContextStore(ContextStore):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def _get_conn(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def initialize_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT,
                attributes TEXT,
                embedding TEXT,
                belief TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                type TEXT NOT NULL,
                scope TEXT,
                provenance TEXT,
                assertion TEXT,
                confidence REAL,
                embedding TEXT,
                salience TEXT,
                policy TEXT,
                FOREIGN KEY(source) REFERENCES nodes(id),
                FOREIGN KEY(target) REFERENCES nodes(id)
            )
        """)
        
        # Index for efficient neighbor lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
        
        conn.commit()

    def add_node(self, node: Node):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO nodes (id, type, name, attributes, embedding, belief)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            node.id,
            node.type,
            node.name,
            json.dumps(node.attributes),
            json.dumps(node.embedding) if node.embedding else None,
            json.dumps(node.belief)
        ))
        conn.commit()

    def get_node(self, node_id: str) -> Optional[Node]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        return Node(
            id=row["id"],
            type=row["type"],
            name=row["name"],
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            belief=json.loads(row["belief"]) if row["belief"] else {}
        )

    def add_edge(self, edge: Edge):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO edges (id, source, target, type, scope, provenance, assertion, confidence, embedding, salience, policy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge.id,
            edge.source,
            edge.target,
            edge.type,
            json.dumps(edge.scope),
            json.dumps(edge.provenance),
            json.dumps(edge.assertion) if edge.assertion else None,
            edge.confidence,
            json.dumps(edge.embedding) if edge.embedding else None,
            json.dumps(edge.salience),
            json.dumps(edge.policy) if edge.policy else None
        ))
        conn.commit()

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM edges WHERE id = ?", (edge_id,))
        row = cursor.fetchone()
        if not row:
            return None
            
        return Edge(
            id=row["id"],
            source=row["source"],
            target=row["target"],
            type=row["type"],
            scope=json.loads(row["scope"]) if row["scope"] else {},
            provenance=json.loads(row["provenance"]) if row["provenance"] else {},
            assertion=json.loads(row["assertion"]) if row["assertion"] else None,
            confidence=row["confidence"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            salience=json.loads(row["salience"]) if row["salience"] else {},
            policy=json.loads(row["policy"]) if row["policy"] else None
        )

    def get_neighbors(self, node_id: str) -> List[Node]:
        conn = self._get_conn()
        cursor = conn.cursor()
        # Find edges where source is node_id, then get target nodes
        cursor.execute("""
            SELECT n.* FROM nodes n
            JOIN edges e ON n.id = e.target
            WHERE e.source = ?
        """, (node_id,))
        
        neighbors = []
        for row in cursor.fetchall():
            neighbors.append(Node(
                id=row["id"],
                type=row["type"],
                name=row["name"],
                attributes=json.loads(row["attributes"]) if row["attributes"] else {},
                embedding=json.loads(row["embedding"]) if row["embedding"] else None,
                belief=json.loads(row["belief"]) if row["belief"] else {}
            ))
        return neighbors

    def get_nearest_nodes(self, query_embedding: List[float], limit: int = 5) -> List[Node]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL")
        
        candidates = []
        import numpy as np
        
        q_vec = np.array(query_embedding)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
            
        for row in cursor.fetchall():
            try:
                vec_data = json.loads(row["embedding"])
                vec = np.array(vec_data)
                norm = np.linalg.norm(vec)
                if norm == 0:
                    continue
                    
                # Cosine similarity
                sim = np.dot(q_vec, vec) / (q_norm * norm)
                candidates.append((row["id"], sim))
            except Exception:
                continue
            
        # Sort by similarity desc
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = candidates[:limit]
        
        results = []
        for node_id, score in top_k:
            node = self.get_node(node_id)
            if node:
                # Inject score into belief for downstream use? 
                # Or just return node.
                # Let's add transient score if needed, but for now just return node.
                results.append(node)
                
        return results

    def get_outbound_edges(self, node_id: str) -> List[Edge]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM edges WHERE source = ?", (node_id,))
        edges = []
        for row in cursor.fetchall():
            edges.append(Edge(
                id=row["id"],
                source=row["source"],
                target=row["target"],
                type=row["type"],
                scope=json.loads(row["scope"]) if row["scope"] else {},
                provenance=json.loads(row["provenance"]) if row["provenance"] else {},
                assertion=json.loads(row["assertion"]) if row["assertion"] else None,
                confidence=row["confidence"],
                embedding=json.loads(row["embedding"]) if row["embedding"] else None,
                salience=json.loads(row["salience"]) if row["salience"] else {},
                policy=json.loads(row["policy"]) if row["policy"] else None
            ))
        return edges


