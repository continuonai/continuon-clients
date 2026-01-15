"""
Filesystem Memory System for Brain B.

Implements a three-level memory hierarchy:
- Procedural (L2): Stable skills, agent definitions, capabilities
- Semantic (L1): Accumulated knowledge, learned behaviors, patterns
- Episodic (L0): Session events, conversations, RLDS episodes

Maps to CMS (Continuous Memory System) with filesystem persistence.
"""

from memory.manager import FilesystemMemory, MemoryLevel
from memory.consolidation import MemoryConsolidator

__all__ = ["FilesystemMemory", "MemoryLevel", "MemoryConsolidator"]
