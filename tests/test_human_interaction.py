#!/usr/bin/env python3
"""
Human Interaction Test Suite

Tests robot interaction quality at different training stages:
1. Raw mode (no training) - pattern matching only
2. Trained mode - with learned navigation model
3. Conversation mode - with trained conversation classifier
4. AI-assisted mode - with LLM backend (Claude/Gemini CLI)
5. Full inference mode - autonomous agent behavior
"""

import sys
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "brain_b"))

from conversation.intents import IntentClassifier, Intent, ParsedIntent
from simulator.simulator_training import (
    SimulatorActionPredictor,
    get_simulator_predictor,
    IDX_TO_ACTION,
)

# Conversation trainer (new)
try:
    from trainer.conversation_trainer import (
        ConversationTrainer,
        get_llm_backend,
        CONVERSATION_INTENTS,
    )
    HAS_CONVERSATION_TRAINER = True
except ImportError:
    HAS_CONVERSATION_TRAINER = False


@dataclass
class InteractionTest:
    """A test case for human interaction."""
    input_text: str
    expected_intent: Optional[Intent]
    expected_action: Optional[str]
    description: str
    mode: str  # "raw", "trained", "ai", "inference"


@dataclass
class InteractionResult:
    """Result of an interaction test."""
    test: InteractionTest
    actual_intent: Intent
    actual_action: Optional[str]
    confidence: float
    passed: bool
    notes: str


class HumanInteractionTester:
    """Tests human-robot interaction quality."""

    # Test cases for different interaction scenarios
    TEST_CASES = [
        # Basic commands (should work in all modes)
        InteractionTest("forward", Intent.MOVE_FORWARD, "forward", "Basic forward command", "raw"),
        InteractionTest("stop", Intent.STOP, "stop", "Basic stop command", "raw"),
        InteractionTest("teach dance", Intent.START_TEACHING, None, "Start teaching", "raw"),
        InteractionTest("help", Intent.HELP, None, "Help request", "raw"),

        # Natural language (requires AI mode)
        InteractionTest("hello", Intent.UNKNOWN, None, "Greeting", "ai"),
        InteractionTest("what can you do", Intent.UNKNOWN, None, "Capability question", "ai"),
        InteractionTest("go to the kitchen", Intent.UNKNOWN, "forward", "Navigation request", "ai"),
        InteractionTest("turn around", Intent.UNKNOWN, "left", "Compound action", "ai"),

        # Shorthand commands
        InteractionTest("f", Intent.MOVE_FORWARD, "forward", "Shorthand forward", "raw"),
        InteractionTest("l", Intent.TURN_LEFT, "left", "Shorthand left", "raw"),
        InteractionTest("s", Intent.STOP, "stop", "Shorthand stop", "raw"),

        # Edge cases
        InteractionTest("", Intent.UNKNOWN, None, "Empty input", "raw"),
        InteractionTest("asdfghjkl", Intent.UNKNOWN, None, "Gibberish", "raw"),
        InteractionTest("FORWARD", Intent.MOVE_FORWARD, "forward", "Uppercase command", "raw"),
        InteractionTest("  forward  ", Intent.MOVE_FORWARD, "forward", "Whitespace padded", "raw"),

        # Teaching workflow
        InteractionTest("teach patrol", Intent.START_TEACHING, None, "Start teaching patrol", "raw"),
        InteractionTest("done", Intent.STOP_TEACHING, None, "Finish teaching", "raw"),
        InteractionTest("cancel", Intent.CANCEL_TEACHING, None, "Cancel teaching", "raw"),

        # Memory commands
        InteractionTest("list behaviors", Intent.LIST_BEHAVIORS, None, "List learned behaviors", "raw"),
        InteractionTest("forget patrol", Intent.FORGET_BEHAVIOR, None, "Forget behavior", "raw"),

        # Speed control
        InteractionTest("faster", Intent.SPEED_UP, None, "Speed up", "raw"),
        InteractionTest("slower", Intent.SLOW_DOWN, None, "Slow down", "raw"),
        InteractionTest("speed 75", Intent.SET_SPEED, None, "Set specific speed", "raw"),
    ]

    def __init__(self):
        self.classifier = IntentClassifier()
        self.predictor = get_simulator_predictor()
        self.results: List[InteractionResult] = []

        # Load trained model if available
        self._load_best_model()

    def _load_best_model(self):
        """Load the best available trained model."""
        checkpoint_dir = Path("brain_b_data/simulator_checkpoints")
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("sim_model_*.json"))
            if checkpoints:
                latest = checkpoints[-1]
                try:
                    self.predictor.load(str(latest))
                    print(f"✅ Loaded model: {latest.name}")
                except Exception as e:
                    print(f"⚠️ Failed to load model: {e}")

    def test_raw_mode(self) -> List[InteractionResult]:
        """Test raw pattern-matching mode (no AI)."""
        print("\n" + "="*60)
        print("📋 RAW MODE TEST (Pattern Matching Only)")
        print("="*60)

        results = []
        raw_tests = [t for t in self.TEST_CASES if t.mode == "raw"]

        for test in raw_tests:
            parsed = self.classifier.classify(test.input_text)
            passed = parsed.intent == test.expected_intent

            result = InteractionResult(
                test=test,
                actual_intent=parsed.intent,
                actual_action=None,
                confidence=parsed.confidence,
                passed=passed,
                notes=f"Got {parsed.intent.name}" if not passed else ""
            )
            results.append(result)

            status = "✅" if passed else "❌"
            print(f"  {status} '{test.input_text}' → {parsed.intent.name} ({test.description})")

        return results

    def test_inference_mode(self) -> List[InteractionResult]:
        """Test trained model inference."""
        print("\n" + "="*60)
        print("🧠 INFERENCE MODE TEST (Trained Navigation Model)")
        print("="*60)

        if not self.predictor.is_ready:
            print("  ⚠️ No trained model available")
            return []

        # Simulate different scenarios
        scenarios = [
            ("Clear path ahead", [0.9, 0.1, 0.1, 0.0, 0.0, 0.0] + [0.0]*42),
            ("Obstacle in front", [0.1, 0.3, 0.7, 0.0, 0.0, 0.0] + [0.0]*42),
            ("Target on left", [0.2, 0.1, 0.1, 0.8, 0.0, 0.0] + [0.0]*42),
            ("Dead end", [0.1, 0.1, 0.1, 0.1, 0.0, 0.0] + [0.0]*42),
        ]

        for name, state in scenarios:
            probs = self.predictor.predict(state)
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            best_action = IDX_TO_ACTION.get(best_idx, f"action_{best_idx}")
            confidence = probs[best_idx]

            print(f"  📍 {name}")
            print(f"     → {best_action} ({confidence:.1%} confidence)")

            # Show top 3 actions
            sorted_actions = sorted(
                [(IDX_TO_ACTION.get(i, f"action_{i}"), p) for i, p in enumerate(probs)],
                key=lambda x: -x[1]
            )[:3]
            for action, prob in sorted_actions:
                bar = "█" * int(prob * 10)
                print(f"       {action:15s} [{bar:10s}] {prob:.1%}")

        return []

    def test_conversation_mode(self) -> List[InteractionResult]:
        """Test trained conversation classifier."""
        print("\n" + "="*60)
        print("💬 CONVERSATION MODE TEST (Trained NLU Classifier)")
        print("="*60)

        if not HAS_CONVERSATION_TRAINER:
            print("  ⚠️ Conversation trainer not available")
            return []

        # Load conversation trainer
        trainer = ConversationTrainer("brain_b_data")
        if not trainer.load_model():
            print("  ⚠️ No trained conversation model found")
            print("  Run: python brain_b/trainer/conversation_trainer.py --train")
            return []

        # Check LLM backend
        llm = get_llm_backend()
        print(f"  LLM Backend: {llm.backend}")
        print(f"  Claude CLI: {'✅' if llm.claude_available else '❌'}")
        print(f"  Gemini: {'✅' if llm.gemini_available else '❌'}")
        print()

        # Test natural language inputs
        test_inputs = [
            ("hello", "greeting"),
            ("hi there", "greeting"),
            ("what can you do", "ask_capabilities"),
            ("how are you", "ask_status"),
            ("go forward", "nav_forward"),
            ("move ahead", "nav_forward"),
            ("stop", "nav_stop"),
            ("turn left", "nav_left"),
            ("go to the kitchen", "nav_location"),
            ("yes", "confirm"),
            ("good job", "praise"),
        ]

        passed = 0
        for text, expected in test_inputs:
            result = trainer.predict(text, use_llm=False)  # Test local model
            actual = result["intent"]
            confidence = result["confidence"]
            is_pass = actual == expected

            status = "✅" if is_pass else "❌"
            print(f"  {status} '{text}' → {actual} ({confidence:.0%}) [expected: {expected}]")

            if is_pass:
                passed += 1

        print(f"\n  Result: {passed}/{len(test_inputs)} tests passed")
        return []

    def test_ai_mode(self) -> List[InteractionResult]:
        """Test AI-assisted mode (requires API keys or Claude CLI)."""
        print("\n" + "="*60)
        print("🤖 AI MODE TEST (LLM-Assisted Conversation)")
        print("="*60)

        # Check for API keys
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        if not gemini_key and not anthropic_key:
            print("  ⚠️ No API keys found (GEMINI_API_KEY or ANTHROPIC_API_KEY)")
            print("  AI conversation not available")
            print()
            print("  Without AI, these inputs are not understood:")
            ai_tests = [t for t in self.TEST_CASES if t.mode == "ai"]
            for test in ai_tests:
                parsed = self.classifier.classify(test.input_text)
                print(f"    '{test.input_text}' → {parsed.intent.name}")
            return []

        # Test with AI backend
        print("  ✅ API key found, testing AI conversation...")
        # TODO: Add actual AI backend testing
        return []

    def run_all_tests(self):
        """Run complete interaction test suite."""
        print("\n" + "="*60)
        print("🔬 HUMAN INTERACTION TEST SUITE")
        print("="*60)

        # Run all test modes
        raw_results = self.test_raw_mode()
        inference_results = self.test_inference_mode()
        conversation_results = self.test_conversation_mode()
        ai_results = self.test_ai_mode()

        # Summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)

        raw_passed = sum(1 for r in raw_results if r.passed)
        raw_total = len(raw_results)

        # Check conversation trainer status
        conv_status = "Not available"
        if HAS_CONVERSATION_TRAINER:
            trainer = ConversationTrainer("brain_b_data")
            if trainer.load_model():
                conv_status = "✅ Trained"
            else:
                conv_status = "⚠️ Not trained"

        # Check LLM backend
        llm_status = "Not available"
        if HAS_CONVERSATION_TRAINER:
            llm = get_llm_backend()
            if llm.is_available:
                llm_status = f"✅ {llm.backend.capitalize()} CLI"
            else:
                llm_status = "❌ No CLI available"

        print(f"  Raw Mode:          {raw_passed}/{raw_total} tests passed")
        print(f"  Inference Mode:    Model {'ready' if self.predictor.is_ready else 'not available'}")
        print(f"  Conversation Mode: {conv_status}")
        print(f"  LLM Backend:       {llm_status}")

        # Recommendations
        print("\n📝 RECOMMENDATIONS:")
        if raw_passed < raw_total:
            print("  - Fix failing pattern matches in intents.py")
        if not self.predictor.is_ready:
            print("  - Run more simulator training to improve inference")
        if conv_status != "✅ Trained":
            print("  - Train conversation model: python brain_b/trainer/conversation_trainer.py --train")
        if "Not available" in llm_status or "No CLI" in llm_status:
            print("  - Install Claude CLI for enhanced understanding: npm i -g @anthropic/claude-cli")

        return raw_results, inference_results, ai_results


def main():
    """Run the human interaction tests."""
    tester = HumanInteractionTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
