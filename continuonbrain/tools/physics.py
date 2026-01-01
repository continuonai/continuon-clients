"""Physics tool for symbolic reasoning and deterministic physics calculations."""
from __future__ import annotations
import math
from typing import Any, Dict
from .base import BaseBrainTool

class PhysicsTool(BaseBrainTool):
    """Executes physics-specific calculations and symbolic reasoning."""

    def __init__(self):
        super().__init__(
            name="physics_solver",
            description="Solve physics problems related to kinematics, dynamics, and energy. (e.g., 'kinematics: v=u+at', 'force: f=ma')."
        )
        # Physics constants
        self.constants = {
            "g": 9.80665,  # Standard gravity m/s²
            "G": 6.67430e-11,  # Gravitational constant m³/kg·s²
            "c": 299792458,  # Speed of light m/s
            "k_B": 1.380649e-23,  # Boltzmann constant J/K
        }

    async def execute(self, problem_type: str, variables: Dict[str, float]) -> Any:
        """Solve a physics problem."""
        try:
            if problem_type == "kinematics":
                # v = u + at
                if "u" in variables and "a" in variables and "t" in variables:
                    v = variables["u"] + variables["a"] * variables["t"]
                    return {"result": v, "formula": "v = u + at", "unit": "m/s"}
                # s = ut + 0.5at²
                if "u" in variables and "a" in variables and "t" in variables:
                    s = variables["u"] * variables["t"] + 0.5 * variables["a"] * (variables["t"]**2)
                    return {"result": s, "formula": "s = ut + 0.5at²", "unit": "m"}
                
            elif problem_type == "force":
                # F = ma
                if "m" in variables and "a" in variables:
                    f = variables["m"] * variables["a"]
                    return {"result": f, "formula": "F = ma", "unit": "N"}
                # Weight: W = mg
                if "m" in variables:
                    w = variables["m"] * self.constants["g"]
                    return {"result": w, "formula": "W = mg", "unit": "N"}

            elif problem_type == "energy":
                # Kinetic Energy: Ek = 0.5mv²
                if "m" in variables and "v" in variables:
                    ek = 0.5 * variables["m"] * (variables["v"]**2)
                    return {"result": ek, "formula": "Ek = 0.5mv²", "unit": "J"}
                # Potential Energy: Ep = mgh
                if "m" in variables and "h" in variables:
                    ep = variables["m"] * self.constants["g"] * variables["h"]
                    return {"result": ep, "formula": "Ep = mgh", "unit": "J"}

            return {"error": f"Incomplete variables for problem type: {problem_type}", "provided": variables}
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}", "problem_type": problem_type}

    def _get_params_spec(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "problem_type": {
                    "type": "string",
                    "description": "Type of physics problem (kinematics, force, energy)."
                },
                "variables": {
                    "type": "object",
                    "description": "Dictionary of numeric variables (e.g., {'m': 5, 'a': 2})."
                }
            },
            "required": ["problem_type", "variables"]
        }
