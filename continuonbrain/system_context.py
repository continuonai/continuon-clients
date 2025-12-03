"""Shared runtime context for ContinuonBrain services.

This module registers the merged :class:`SystemInstructions` so that different
components (mode manager, API server) can consult the same rules without
reloading disparate copies. It also persists the merged payload for child
processes that need to hydrate their own copy during startup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from continuonbrain.system_instructions import SystemInstructions


class SystemContext:
    """Global registry for system instructions within a process."""

    _system_instructions: Optional[SystemInstructions] = None
    _persist_path: Optional[Path] = None

    @classmethod
    def register_instructions(
        cls,
        instructions: SystemInstructions,
        *,
        persist_path: Optional[Path] = None,
    ) -> None:
        """Register merged instructions and optionally persist them to disk."""

        if instructions is None:
            raise ValueError("instructions must be provided")

        cls._system_instructions = instructions
        cls._persist_path = persist_path

        if persist_path:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            persist_path.write_text(json.dumps(instructions.as_dict(), indent=2))

    @classmethod
    def get_instructions(cls) -> Optional[SystemInstructions]:
        """Return the registered instructions if available."""

        return cls._system_instructions

    @classmethod
    def require_instructions(cls) -> SystemInstructions:
        """Return registered instructions or raise if missing."""

        if cls._system_instructions is None:
            raise RuntimeError("System instructions have not been registered")

        return cls._system_instructions

    @classmethod
    def get_persist_path(cls) -> Optional[Path]:
        """Return the path used to persist the registered instructions."""

        return cls._persist_path

    @classmethod
    def load_and_register(cls, source_path: Path) -> SystemInstructions:
        """Load serialized instructions from disk and register them."""

        instructions = SystemInstructions.load_serialized(source_path)
        cls.register_instructions(instructions, persist_path=source_path)
        return instructions

