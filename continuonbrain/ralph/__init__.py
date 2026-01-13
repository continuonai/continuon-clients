"""
Ralph Layer for ContinuonBrain
==============================

Implements Geoffrey Huntley's Ralph Wiggum technique for autonomous AI development
with deliberate context rotation, applied to the ContinuonBrain's three-loop architecture.

Architecture:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    META RALPH AGENT                                  │
    │                 (Primary User Interaction)                          │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
    │   │ FastLoop    │  │ MidLoop     │  │ SlowLoop    │  │ Safety    │  │
    │   │ Ralph       │  │ Ralph       │  │ Ralph       │  │ Ralph     │  │
    │   │ (10ms)      │  │ (100ms)     │  │ (Cloud)     │  │ (Ring 0)  │  │
    │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  │
    │          │                │                │               │        │
    │          ▼                ▼                ▼               ▼        │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │              HOPE Brain (Three Loops)                       │  │
    │   │   Fast Loop ◄──► Mid Loop ◄──► Slow Loop                   │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                              │                                      │
    │                              ▼                                      │
    │   ┌─────────────────────────────────────────────────────────────┐  │
    │   │              Safety Kernel (Ring 0)                         │  │
    │   └─────────────────────────────────────────────────────────────┘  │
    │                                                                       │
    ├─────────────────────────────────────────────────────────────────────┤
    │   Claude Code CLI (Swappable - Opus 4.5 / Gemini / Ollama)        │
    └─────────────────────────────────────────────────────────────────────┘

Each Ralph layer provides:
- Fresh context each iteration (deliberate context rotation)
- Guardrails (signs) that persist lessons learned
- Git-based state persistence
- Loop-specific meta-layer for introspection

Usage:
    from ralph import MetaRalphAgent

    agent = MetaRalphAgent(brain_service)
    response = await agent.process_user_message("Hello, robot!")
"""

from .base import RalphLayer, RalphConfig, RalphState
from .fast_loop_ralph import FastLoopRalph
from .mid_loop_ralph import MidLoopRalph
from .slow_loop_ralph import SlowLoopRalph
from .safety_ralph import SafetyRalph
from .meta_ralph import MetaRalphAgent
from .claude_code_cli import ClaudeCodeCLI, CLIProvider
from .chat_integration import RalphChatAdapter, RalphAPIRoutes, create_ralph_chat_adapter
from .config import RalphLayerConfig, get_ralph_config, reload_ralph_config

__all__ = [
    # Core Ralph layers
    "RalphLayer",
    "RalphConfig",
    "RalphState",
    "FastLoopRalph",
    "MidLoopRalph",
    "SlowLoopRalph",
    "SafetyRalph",

    # Meta agent and CLI
    "MetaRalphAgent",
    "ClaudeCodeCLI",
    "CLIProvider",

    # Chat integration
    "RalphChatAdapter",
    "RalphAPIRoutes",
    "create_ralph_chat_adapter",

    # Configuration
    "RalphLayerConfig",
    "get_ralph_config",
    "reload_ralph_config",
]

__version__ = "1.0.0"
