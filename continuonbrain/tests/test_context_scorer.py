import pytest
import time
from datetime import datetime, timedelta
from continuonbrain.core.context_graph_models import Edge
from continuonbrain.core.context_scorer import ContextScorer

class TestContextScorer:
    def test_exp_decay(self):
        scorer = ContextScorer()
        now = datetime.now()
        past = now - timedelta(seconds=3600) # 1 hour ago
        
        edge = Edge(
            id="e1", source="a", target="b", type="t",
            salience={
                "score": 1.0,
                "decay_fn": "exp",
                "half_life_s": 3600,
                "last_updated": past.isoformat()
            }
        )
        
        score = scorer.compute_salience(edge, current_time_ts=now.timestamp())
        assert abs(score - 0.5) < 0.01

    def test_no_decay(self):
        scorer = ContextScorer()
        edge = Edge(id="e1", source="a", target="b", type="t")
        assert scorer.compute_salience(edge) == 1.0
