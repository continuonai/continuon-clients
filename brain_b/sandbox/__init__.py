"""
Brain B Sandbox - Isolated execution environment for agents.

Based on Anthropic's Claude Cowork sandboxing pattern.
Provides hardware and network gating with allow/deny lists.
"""

from .manager import Sandbox, SandboxConfig, SandboxManager, SandboxViolation
from .hardware_gate import HardwareGate
from .network_gate import NetworkGate

__all__ = [
    "Sandbox",
    "SandboxConfig",
    "SandboxManager",
    "SandboxViolation",
    "HardwareGate",
    "NetworkGate",
]
