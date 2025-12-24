from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

@dataclass
class Node:
    id: str
    type: str  # "entity|tool|goal|intent|concept|policy|context_session"
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict) # tags, cms_span_ids, rlds_episode_ids
    embedding: Optional[List[float]] = None
    belief: Dict[str, Any] = field(default_factory=dict) # score, updated_at

@dataclass
class Edge:
    id: str
    source: str # Node ID
    target: str # Node ID
    type: str # "causal|temporal|membership|policy|tool_use|assertion"
    scope: Dict[str, Any] = field(default_factory=dict) # time_range, location, audience
    provenance: Dict[str, Any] = field(default_factory=dict) # episode_id, step_index, cms_span_id
    assertion: Optional[Dict[str, Any]] = None # predicate, value
    confidence: float = 1.0
    embedding: Optional[List[float]] = None
    salience: Dict[str, Any] = field(default_factory=dict) # score, decay_fn, half_life_s, last_updated
    policy: Optional[Dict[str, Any]] = None # allow_roles, deny_roles

@dataclass
class Episode:
    id: str
    start_time: datetime
    end_time: datetime
    location: str
    nodes: List[str] = field(default_factory=list) # Node IDs
    edges: List[str] = field(default_factory=list) # Edge IDs
    embedding: Optional[List[float]] = None

@dataclass
class ContextSession:
    id: str
    goals: List[str] = field(default_factory=list) # Node IDs
    tools: List[str] = field(default_factory=list) # Node IDs
    active_entities: List[str] = field(default_factory=list) # Node IDs
    intents: List[str] = field(default_factory=list) # Node IDs
    context_state_vector: Optional[List[float]] = None
    constraints: Dict[str, Any] = field(default_factory=dict) # permissions, time_window
