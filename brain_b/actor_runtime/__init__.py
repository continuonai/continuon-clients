"""
Brain B Actor Runtime

Event-sourced actor model for teachable robots.
"""

from .agent import Agent, ResourceBudget
from .event_log import Event, EventLog
from .runtime import ActorRuntime
from .teaching import Behavior, TeachingSystem

__all__ = [
    "Agent",
    "ResourceBudget",
    "Event",
    "EventLog",
    "ActorRuntime",
    "Behavior",
    "TeachingSystem",
]
