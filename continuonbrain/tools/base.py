"""Base classes for Brain Tools."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseBrainTool(ABC):
    """Standard interface for all tools the Brain can use."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with provided arguments."""
        pass

    def get_spec(self) -> Dict[str, Any]:
        """Return the tool's specification for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_params_spec()
        }

    @abstractmethod
    def _get_params_spec(self) -> Dict[str, Any]:
        """Return the JSON schema for tool parameters."""
        pass


class ToolRegistry:
    """Registry for dynamic loading and lookup of Brain capabilities."""

    def __init__(self):
        self._tools: Dict[str, BaseBrainTool] = {}

    def register(self, tool: BaseBrainTool) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseBrainTool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return specifications for all registered tools."""
        return [t.get_spec() for t in self._tools.values()]

    async def call(self, name: str, **kwargs: Any) -> Any:
        """Invoke a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry.")
        return await tool.execute(**kwargs)
