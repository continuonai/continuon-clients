"""
Tool Predictor Service for Brain B.

Loads trained model and provides tool predictions
based on context from Claude Code sessions.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Prediction:
    """A tool prediction with confidence."""
    tool: str
    confidence: float
    alternatives: List[Tuple[str, float]]  # (tool, probability)


class ToolPredictorService:
    """
    Service that provides tool predictions using trained model.

    Integrates with Brain B to suggest which tool to use next
    based on learned patterns from RLDS episodes.
    """

    # Default tool vocabulary
    TOOLS = [
        "Bash", "Read", "Write", "Edit", "Grep", "Glob",
        "Task", "WebFetch", "WebSearch", "TodoWrite",
        "AskUserQuestion", "Other"
    ]

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor service.

        Args:
            model_path: Path to trained model JSON file
        """
        self.model_path = model_path
        self.weights: Optional[List[List[float]]] = None
        self.input_dim: int = 32
        self.num_tools: int = len(self.TOOLS)
        self.tool_vocab: List[str] = self.TOOLS.copy()
        self._loaded = False

        if model_path:
            self.load(model_path)

    def load(self, path: str) -> bool:
        """Load model from file."""
        try:
            with open(path) as f:
                data = json.load(f)

            self.input_dim = data.get("input_dim", 32)
            self.num_tools = data.get("num_tools", len(self.TOOLS))
            self.weights = data.get("weights", [])
            self.tool_vocab = data.get("tool_vocab", self.TOOLS)
            self._loaded = True

            print(f"[Predictor] Loaded model from {path}")
            print(f"[Predictor] Input dim: {self.input_dim}, Tools: {self.num_tools}")
            return True

        except Exception as e:
            print(f"[Predictor] Failed to load model: {e}")
            self._loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._loaded and self.weights is not None

    def predict(self, context: Dict) -> Optional[Prediction]:
        """
        Predict next tool given context.

        Args:
            context: Dictionary with context information:
                - current_tool: Currently active tool (if any)
                - prev_tool: Previously used tool
                - prev_success: Whether previous tool succeeded
                - step_idx: Current step in session

        Returns:
            Prediction with tool, confidence, and alternatives
        """
        if not self.is_ready:
            return None

        # Build context vector
        vector = self._context_to_vector(context)

        # Forward pass
        probs = self._forward(vector)

        # Get top prediction and alternatives
        indexed_probs = [(i, p) for i, p in enumerate(probs)]
        indexed_probs.sort(key=lambda x: -x[1])

        best_idx, best_prob = indexed_probs[0]
        alternatives = [
            (self.tool_vocab[idx], prob)
            for idx, prob in indexed_probs[1:4]  # Top 3 alternatives
        ]

        return Prediction(
            tool=self.tool_vocab[best_idx],
            confidence=best_prob,
            alternatives=alternatives,
        )

    def predict_for_task(self, task_description: str, history: List[str] = None) -> Optional[Prediction]:
        """
        Predict tool based on task description and history.

        Args:
            task_description: Natural language description of task
            history: List of previously used tools

        Returns:
            Prediction with tool and confidence
        """
        # Build context from task
        context = {
            "current_tool": "",
            "prev_tool": history[-1] if history else "",
            "prev_success": True,
            "step_idx": len(history) if history else 0,
        }

        # Enhance context with task keywords
        task_lower = task_description.lower()

        # Heuristic boosting based on keywords
        if any(kw in task_lower for kw in ["run", "execute", "command", "git", "npm"]):
            context["hint_bash"] = True
        elif any(kw in task_lower for kw in ["read", "look at", "check", "view"]):
            context["hint_read"] = True
        elif any(kw in task_lower for kw in ["write", "create", "new file"]):
            context["hint_write"] = True
        elif any(kw in task_lower for kw in ["edit", "modify", "change", "update"]):
            context["hint_edit"] = True
        elif any(kw in task_lower for kw in ["find", "search", "grep"]):
            context["hint_grep"] = True

        return self.predict(context)

    def _context_to_vector(self, context: Dict) -> List[float]:
        """Convert context dictionary to feature vector."""
        vector = []

        # Current tool one-hot (12 features)
        current = context.get("current_tool", "")
        for tool in self.tool_vocab:
            vector.append(1.0 if tool == current else 0.0)

        # Step position normalized
        step_idx = context.get("step_idx", 0)
        vector.append(min(step_idx / 100.0, 1.0))

        # Previous tool one-hot (12 features)
        prev = context.get("prev_tool", "")
        for tool in self.tool_vocab:
            vector.append(1.0 if tool == prev else 0.0)

        # Previous success
        vector.append(1.0 if context.get("prev_success", True) else 0.0)

        # Hint features (boost certain tools)
        vector.append(1.0 if context.get("hint_bash") else 0.0)
        vector.append(1.0 if context.get("hint_read") else 0.0)
        vector.append(1.0 if context.get("hint_write") else 0.0)
        vector.append(1.0 if context.get("hint_edit") else 0.0)
        vector.append(1.0 if context.get("hint_grep") else 0.0)

        # Pad to input_dim
        while len(vector) < self.input_dim:
            vector.append(0.0)

        return vector[:self.input_dim]

    def _forward(self, vector: List[float]) -> List[float]:
        """Forward pass through linear model."""
        import math

        # Linear transformation
        logits = []
        for tool_weights in self.weights:
            logit = sum(w * x for w, x in zip(tool_weights, vector))
            logits.append(logit)

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        return probs

    def get_tool_stats(self) -> Dict[str, float]:
        """Get learned tool biases (useful for debugging)."""
        if not self.is_ready:
            return {}

        # Sum of absolute weights per tool (rough importance)
        stats = {}
        for i, tool in enumerate(self.tool_vocab):
            if i < len(self.weights):
                stats[tool] = sum(abs(w) for w in self.weights[i])

        return stats


# Singleton instance for easy import
_predictor_instance: Optional[ToolPredictorService] = None


def get_predictor(model_path: Optional[str] = None) -> ToolPredictorService:
    """Get or create the predictor singleton."""
    global _predictor_instance

    if _predictor_instance is None:
        # Try default path
        if model_path is None:
            default_path = Path(__file__).parent.parent / "brain_b_data" / "models" / "tool_predictor_model.json"
            if default_path.exists():
                model_path = str(default_path)

        _predictor_instance = ToolPredictorService(model_path)

    return _predictor_instance


def predict_tool(context: Dict) -> Optional[Prediction]:
    """Convenience function for quick predictions."""
    return get_predictor().predict(context)


if __name__ == "__main__":
    # Test the predictor
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    predictor = ToolPredictorService(model_path)

    if not predictor.is_ready:
        print("Model not loaded. Trying default path...")
        default = Path(__file__).parent.parent / "brain_b_data" / "models" / "tool_predictor_model.json"
        if default.exists():
            predictor.load(str(default))

    if predictor.is_ready:
        print("\n=== Tool Prediction Test ===\n")

        # Test predictions
        test_cases = [
            {"current_tool": "", "prev_tool": "", "step_idx": 0},
            {"current_tool": "", "prev_tool": "Bash", "step_idx": 1, "prev_success": True},
            {"current_tool": "", "prev_tool": "Read", "step_idx": 2, "prev_success": True},
            {"current_tool": "", "prev_tool": "Write", "step_idx": 3, "prev_success": True},
        ]

        for ctx in test_cases:
            pred = predictor.predict(ctx)
            if pred:
                print(f"Context: prev={ctx.get('prev_tool', 'None'):10} -> Predicted: {pred.tool:10} ({pred.confidence:.2%})")
                print(f"  Alternatives: {', '.join(f'{t}({p:.1%})' for t, p in pred.alternatives)}")

        print("\n=== Task-Based Predictions ===\n")

        tasks = [
            "Run the tests",
            "Read the config file",
            "Create a new module",
            "Fix the bug in handler.py",
            "Search for error handling code",
        ]

        for task in tasks:
            pred = predictor.predict_for_task(task)
            if pred:
                print(f"Task: '{task}'")
                print(f"  -> {pred.tool} ({pred.confidence:.2%})")

        print("\n=== Tool Weight Stats ===\n")
        stats = predictor.get_tool_stats()
        for tool, weight in sorted(stats.items(), key=lambda x: -x[1])[:5]:
            print(f"  {tool}: {weight:.2f}")
    else:
        print("No model available for testing")
