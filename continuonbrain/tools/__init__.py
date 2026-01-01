"""Brain Tools Package."""
from .base import ToolRegistry, BaseBrainTool
from .calculator import CalculatorTool
from .filesystem import FileSystemTool
from .wikipedia import WikipediaTool
from .physics import PhysicsTool

def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FileSystemTool())
    registry.register(WikipediaTool())
    registry.register(PhysicsTool())
    return registry
