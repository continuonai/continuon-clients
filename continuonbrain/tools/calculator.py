"""Calculator tool for deterministic math."""
from __future__ import annotations
import math
from typing import Any, Dict
from .base import BaseBrainTool

try:
    from simpleeval import SimpleEval
except ImportError:
    SimpleEval = None


class CalculatorTool(BaseBrainTool):
    """Executes safe mathematical expressions."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate a mathematical expression (e.g., '2 + 2', 'math.sqrt(16)'). Use for precise calculations."
        )
        if SimpleEval:
            self._evaluator = SimpleEval()
            # Register math functions
            self._evaluator.functions = {
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "exp": math.exp,
                "pow": pow,
                "abs": abs,
            }
            self._evaluator.names = {"math": math, "pi": math.pi, "e": math.e}
        else:
            self._evaluator = None

    async def execute(self, expression: str) -> Any:
        """Execute the calculation."""
        if not self._evaluator:
            return {"error": "simpleeval package not installed. Cannot execute math."}
        
        try:
            result = self._evaluator.eval(expression)
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": f"Evaluation error: {str(e)}", "expression": expression}

    def _get_params_spec(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate."
                }
            },
            "required": ["expression"]
        }
