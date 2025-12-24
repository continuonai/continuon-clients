import math
import time
from datetime import datetime
from typing import List, Dict
from .context_graph_models import Edge

class ContextScorer:
    def __init__(self):
        pass

    def compute_salience(self, edge: Edge, current_time_ts: float = None) -> float:
        """
        Compute current salience score based on decay.
        """
        if not edge.salience:
            return 1.0
            
        initial_score = edge.salience.get("score", 1.0)
        decay_fn = edge.salience.get("decay_fn", "exp")
        half_life_s = edge.salience.get("half_life_s", 3600)
        last_updated_str = edge.salience.get("last_updated")
        
        if not last_updated_str:
            return initial_score
            
        try:
            # Parse ISO format
            last_updated = datetime.fromisoformat(last_updated_str)
            if not current_time_ts:
                current_time_ts = time.time()
                
            elapsed_s = current_time_ts - last_updated.timestamp()
            if elapsed_s < 0:
                return initial_score
                
            if decay_fn == "exp":
                # N(t) = N0 * (1/2)^(t / half_life)
                if half_life_s <= 0:
                    return 0.0
                score = initial_score * (0.5 ** (elapsed_s / half_life_s))
            elif decay_fn == "linear":
                # Linear decay
                if half_life_s <= 0:
                     return 0.0
                # Assume half_life is time to reach 0.5? Or total life?
                # Spec says "half_life_s". Let's stick to exp as primary.
                score = max(0.0, initial_score - (elapsed_s / (half_life_s * 2)))
            else:
                score = initial_score
                
            return score
        except Exception:
            return initial_score
