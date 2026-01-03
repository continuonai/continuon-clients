"""
ContinuonBrain Swarm Intelligence Module

Enables robots to:
1. Build new robots from available parts
2. Clone seed images to new hardware
3. Coordinate with sibling robots
4. Share learned experiences (with consent)

Safety Principles:
- Owner must authorize all robot construction
- Parts must be legally owned by the owner
- New robots inherit safety kernel (Ring 0)
- New robots are paired to same owner by default
"""

from .builder import RobotBuilder, BuildPlan, PartInventory, Part, PartCategory, RobotArchetype
from .replicator import SeedReplicator, CloneJob
from .coordination import SwarmCoordinator, SwarmMessage, SwarmRobot
from .authorized_builder import AuthorizedRobotBuilder, create_authorized_builder

__all__ = [
    # Builder
    'RobotBuilder',
    'BuildPlan', 
    'PartInventory',
    'Part',
    'PartCategory',
    'RobotArchetype',
    # Replicator
    'SeedReplicator',
    'CloneJob',
    # Coordination
    'SwarmCoordinator',
    'SwarmMessage',
    'SwarmRobot',
    # Authorized Builder (integrates with safety)
    'AuthorizedRobotBuilder',
    'create_authorized_builder',
]

