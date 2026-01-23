#!/usr/bin/env python3
"""
Conversation Trainer - Train natural language understanding for Brain B

This module trains the robot to understand:
1. Natural language commands ("go to the kitchen" → forward)
2. Conversational intents ("hello", "what can you do")
3. Context-aware responses (remembering conversation state)

Training Data Sources:
- Recorded conversations from Brain B sessions
- Synthetic conversation examples
- RLDS episodes with language annotations
- LLM-generated training data (Claude/Gemini when available)

Output:
- Intent classifier model
- Response templates
- Conversation embeddings

LLM Backends:
- Claude Code CLI (claude --print -p "...")
- Gemini CLI (gemini "...")
- Falls back to local model when unavailable
"""

import json
import math
import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any


# =============================================================================
# LLM Backend for Enhanced Understanding
# =============================================================================

class LLMConversationBackend:
    """
    Uses Claude Code CLI or Gemini CLI for advanced conversation understanding.
    Falls back to local model when LLMs are unavailable.
    """

    ROBOT_SYSTEM_PROMPT = """You are a robot assistant brain. Parse user input and respond with JSON:
{
    "intent": "<one of: greeting, farewell, ask_capabilities, ask_status, nav_forward, nav_backward, nav_left, nav_right, nav_stop, nav_location, start_teaching, confirm, deny, praise, correction, clarify, unknown>",
    "action": {"type": "<forward|backward|left|right|stop|null>"} or null,
    "response": "<brief friendly response to user>",
    "confidence": <0.0-1.0>
}

Available robot actions: forward, backward, left, right, stop
Teaching: user can say "teach <name>" to record a behavior sequence

Be concise. Parse the intent accurately."""

    def __init__(self):
        self.claude_available = self._check_claude()
        self.gemini_available = self._check_gemini()
        self.backend = self._select_backend()

    def _check_claude(self) -> bool:
        """Check if Claude Code CLI is available."""
        return shutil.which("claude") is not None

    def _check_gemini(self) -> bool:
        """Check if Gemini CLI is available."""
        # Check for gemini CLI or API key
        if shutil.which("gemini"):
            return True
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI"))

    def _select_backend(self) -> str:
        """Select the best available backend."""
        if self.claude_available:
            return "claude"
        if self.gemini_available:
            return "gemini"
        return "local"

    @property
    def is_available(self) -> bool:
        """Check if any LLM backend is available."""
        return self.claude_available or self.gemini_available

    def process(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """
        Process user input through LLM backend.

        Args:
            user_input: The user's message
            context: Optional conversation context

        Returns:
            Dict with intent, action, response, confidence
        """
        if self.backend == "claude":
            return self._process_claude(user_input, context)
        elif self.backend == "gemini":
            return self._process_gemini(user_input, context)
        else:
            return self._process_local(user_input, context)

    def _process_claude(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Process using Claude Code CLI."""
        try:
            prompt = f"""{self.ROBOT_SYSTEM_PROMPT}

User says: "{user_input}"

Respond with only valid JSON, no explanation."""

            result = subprocess.run(
                ["claude", "--print", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Parse JSON from response
                response_text = result.stdout.strip()
                # Find JSON in response
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())

        except subprocess.TimeoutExpired:
            print("[LLM] Claude timeout")
        except json.JSONDecodeError as e:
            print(f"[LLM] JSON parse error: {e}")
        except Exception as e:
            print(f"[LLM] Claude error: {e}")

        return self._process_local(user_input, context)

    def _process_gemini(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Process using Gemini (API or CLI)."""
        try:
            # Try Gemini Python API first
            import google.generativeai as genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI")
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")

                prompt = f"""{self.ROBOT_SYSTEM_PROMPT}

User says: "{user_input}"

Respond with only valid JSON."""

                response = model.generate_content(prompt)
                response_text = response.text.strip()

                # Find JSON in response
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())

        except ImportError:
            # Try CLI fallback
            if shutil.which("gemini"):
                try:
                    result = subprocess.run(
                        ["gemini", f'{self.ROBOT_SYSTEM_PROMPT}\n\nUser: {user_input}'],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        json_match = re.search(r'\{[^{}]*\}', result.stdout, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group())
                except Exception:
                    pass
        except Exception as e:
            print(f"[LLM] Gemini error: {e}")

        return self._process_local(user_input, context)

    def _process_local(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Fallback local processing (pattern matching)."""
        return {
            "intent": "unknown",
            "action": None,
            "response": None,  # Let caller handle
            "confidence": 0.0,
            "backend": "local",
        }

    def generate_training_samples(self, base_phrases: List[str], count: int = 100) -> List[Dict]:
        """
        Use LLM to generate diverse training samples from base phrases.

        Args:
            base_phrases: List of example phrases
            count: Number of samples to generate

        Returns:
            List of training sample dicts
        """
        if not self.is_available:
            return []

        samples = []
        batch_size = 10

        prompt_template = """Generate {n} natural variations of robot commands. For each, provide:
- user_input: what the user says
- intent: the intent category
- action: robot action if applicable

Base examples: {examples}

Categories: greeting, farewell, nav_forward, nav_backward, nav_left, nav_right, nav_stop, ask_capabilities, ask_status

Return as JSON array. Be creative with phrasing variations."""

        for i in range(0, count, batch_size):
            try:
                examples = random.sample(base_phrases, min(5, len(base_phrases)))
                prompt = prompt_template.format(n=batch_size, examples=examples)

                if self.backend == "claude":
                    result = subprocess.run(
                        ["claude", "--print", "-p", prompt],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode == 0:
                        # Find JSON array
                        json_match = re.search(r'\[[\s\S]*\]', result.stdout)
                        if json_match:
                            batch = json.loads(json_match.group())
                            samples.extend(batch)

            except Exception as e:
                print(f"[LLM] Generation error: {e}")
                continue

        return samples[:count]


# Global LLM backend instance
_llm_backend: Optional[LLMConversationBackend] = None


def get_llm_backend() -> LLMConversationBackend:
    """Get or create LLM backend singleton."""
    global _llm_backend
    if _llm_backend is None:
        _llm_backend = LLMConversationBackend()
    return _llm_backend


# =============================================================================
# Conversation Intents & Templates
# =============================================================================

# Conversation intents (extended from intents.py)
CONVERSATION_INTENTS = {
    # Greetings
    "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "howdy", "greetings"],
    "farewell": ["bye", "goodbye", "see you", "later", "take care", "good night"],

    # Questions about capabilities
    "ask_capabilities": [
        "what can you do", "what are your abilities", "help me", "what do you know",
        "show me what you can do", "what are you capable of", "your skills"
    ],
    "ask_status": [
        "how are you", "what's your status", "are you ok", "how do you feel",
        "what are you doing", "status report"
    ],

    # Navigation requests (natural language)
    "nav_forward": [
        "go forward", "move ahead", "go straight", "keep going", "continue",
        "go to the front", "move forward please"
    ],
    "nav_backward": [
        "go back", "move backward", "reverse", "back up", "go backwards"
    ],
    "nav_left": [
        "turn left", "go left", "rotate left", "left turn", "bear left"
    ],
    "nav_right": [
        "turn right", "go right", "rotate right", "right turn", "bear right"
    ],
    "nav_stop": [
        "stop", "halt", "freeze", "wait", "hold on", "pause", "don't move"
    ],
    "nav_location": [
        "go to the kitchen", "go to the bedroom", "go to the living room",
        "find the door", "go to the table", "come here", "follow me"
    ],

    # Teaching
    "start_teaching": [
        "let me teach you", "learn this", "watch what I do", "remember this",
        "I'll show you how", "let me show you"
    ],
    "confirm": ["yes", "yeah", "yep", "correct", "right", "ok", "okay", "sure", "affirmative"],
    "deny": ["no", "nope", "wrong", "incorrect", "cancel", "nevermind", "stop that"],

    # Feedback
    "praise": ["good job", "well done", "nice", "perfect", "excellent", "great work"],
    "correction": ["no not that", "wrong way", "other direction", "try again"],

    # Clarification
    "clarify": [
        "what do you mean", "I don't understand", "explain", "say again",
        "what", "huh", "pardon"
    ],
}

# Response templates for each intent
RESPONSE_TEMPLATES = {
    "greeting": [
        "Hello! I'm ready to help. Say 'help' to see what I can do.",
        "Hi there! How can I assist you today?",
        "Hey! I'm your robot assistant. What would you like me to do?",
    ],
    "farewell": [
        "Goodbye! Call me if you need anything.",
        "See you later! I'll be here when you need me.",
        "Take care!",
    ],
    "ask_capabilities": [
        "I can move around (forward, back, left, right), learn new behaviors when you teach me, and follow your commands. Try 'teach dance' to create a new behavior!",
        "I understand movement commands, can learn sequences you show me, and remember behaviors. Say 'list' to see what I've learned.",
    ],
    "ask_status": [
        "I'm doing well and ready for commands!",
        "All systems operational. Speed is at {speed}%.",
        "I'm here and listening. {behaviors} behaviors learned so far.",
    ],
    "nav_forward": ["Moving forward.", "Going ahead.", "On my way forward."],
    "nav_backward": ["Backing up.", "Moving backward.", "Reversing."],
    "nav_left": ["Turning left.", "Rotating left.", "Going left."],
    "nav_right": ["Turning right.", "Rotating right.", "Going right."],
    "nav_stop": ["Stopped.", "Holding position.", "Waiting for your next command."],
    "nav_location": [
        "I'll try to navigate there. Moving forward to start.",
        "Heading in that direction.",
    ],
    "start_teaching": [
        "I'm ready to learn! Show me the steps and say 'done' when finished.",
        "Teaching mode activated. What should I call this behavior?",
    ],
    "confirm": ["Got it!", "Understood.", "Okay!"],
    "deny": ["Alright, I'll stop.", "Cancelled.", "Okay, nevermind."],
    "praise": ["Thank you! I'll remember that worked well.", "Glad I could help!"],
    "correction": ["Sorry about that. Let me try a different approach.", "Adjusting..."],
    "clarify": [
        "I can move (forward, back, left, right), learn behaviors (teach <name>), and run learned sequences. What would you like?",
    ],
    "unknown": [
        "I'm not sure what you mean. Try 'help' for available commands.",
        "Could you rephrase that? I understand movement and teaching commands.",
    ],
}

# Intent to action mapping
INTENT_TO_ACTION = {
    "nav_forward": {"type": "forward"},
    "nav_backward": {"type": "backward"},
    "nav_left": {"type": "left"},
    "nav_right": {"type": "right"},
    "nav_stop": {"type": "stop"},
    "nav_location": {"type": "forward"},  # Default to forward for location requests
}


@dataclass
class ConversationSample:
    """A single conversation training sample."""
    user_input: str
    intent: str
    response: str
    action: Optional[Dict] = None
    context: Dict = field(default_factory=dict)
    source: str = "synthetic"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationTrainingMetrics:
    """Metrics from conversation training."""
    samples_trained: int = 0
    intents_learned: int = 0
    accuracy: float = 0.0
    loss: float = 0.0
    epochs: int = 0


class ConversationDataset:
    """Dataset for conversation training."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.samples: List[ConversationSample] = []
        self.intent_counts: Dict[str, int] = {}

    def load(self) -> int:
        """Load conversation data from files."""
        conv_dir = self.data_dir / "conversations"
        if not conv_dir.exists():
            return 0

        for conv_file in conv_dir.glob("*.json"):
            try:
                data = json.loads(conv_file.read_text())
                for sample_data in data.get("samples", []):
                    sample = ConversationSample(
                        user_input=sample_data["user_input"],
                        intent=sample_data["intent"],
                        response=sample_data.get("response", ""),
                        action=sample_data.get("action"),
                        context=sample_data.get("context", {}),
                        source=sample_data.get("source", "file"),
                    )
                    self.samples.append(sample)
                    self.intent_counts[sample.intent] = self.intent_counts.get(sample.intent, 0) + 1
            except Exception as e:
                print(f"Error loading {conv_file}: {e}")

        return len(self.samples)

    def generate_synthetic(self, count: int = 1000) -> int:
        """Generate synthetic conversation samples."""
        generated = 0

        for intent, phrases in CONVERSATION_INTENTS.items():
            # Generate variations for each phrase
            samples_per_intent = count // len(CONVERSATION_INTENTS)

            for _ in range(samples_per_intent):
                # Pick a random phrase
                phrase = random.choice(phrases)

                # Apply augmentations
                augmented = self._augment_phrase(phrase)

                # Get response template
                templates = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["unknown"])
                response = random.choice(templates)

                # Get action if applicable
                action = INTENT_TO_ACTION.get(intent)

                sample = ConversationSample(
                    user_input=augmented,
                    intent=intent,
                    response=response,
                    action=action,
                    source="synthetic",
                )
                self.samples.append(sample)
                self.intent_counts[intent] = self.intent_counts.get(intent, 0) + 1
                generated += 1

        return generated

    def _augment_phrase(self, phrase: str) -> str:
        """Apply random augmentations to a phrase."""
        augmentations = [
            lambda p: p,  # Original
            lambda p: p.upper(),  # Uppercase
            lambda p: p.lower(),  # Lowercase
            lambda p: p.capitalize(),  # Capitalized
            lambda p: f"  {p}  ",  # Whitespace padding
            lambda p: f"please {p}",  # Polite prefix
            lambda p: f"{p} please",  # Polite suffix
            lambda p: f"can you {p}",  # Question form
            lambda p: f"{p}!",  # Exclamation
            lambda p: f"{p}?",  # Question mark
        ]

        augmentation = random.choice(augmentations)
        return augmentation(phrase)

    def save(self, filename: str = "conversation_data.json"):
        """Save dataset to file."""
        conv_dir = self.data_dir / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "samples": [
                {
                    "user_input": s.user_input,
                    "intent": s.intent,
                    "response": s.response,
                    "action": s.action,
                    "context": s.context,
                    "source": s.source,
                    "timestamp": s.timestamp,
                }
                for s in self.samples
            ],
            "intent_counts": self.intent_counts,
            "generated_at": datetime.now().isoformat(),
        }

        filepath = conv_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def get_train_test_split(self, test_ratio: float = 0.2) -> Tuple[List[ConversationSample], List[ConversationSample]]:
        """Split data into train and test sets."""
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * (1 - test_ratio))
        return self.samples[:split_idx], self.samples[split_idx:]


class ConversationIntentClassifier:
    """Neural-style intent classifier using bag-of-words + simple network."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.intent_to_idx: Dict[str, int] = {}
        self.idx_to_intent: Dict[int, str] = {}

        # Simple weight matrices (will be trained)
        self.weights_hidden: List[List[float]] = []
        self.weights_output: List[List[float]] = []
        self.bias_hidden: List[float] = []
        self.bias_output: List[float] = []

        self.hidden_size = 64
        self.is_trained = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def _build_vocab(self, samples: List[ConversationSample]):
        """Build vocabulary from samples."""
        self.vocab = {"<UNK>": 0}
        self.intent_to_idx = {}

        for sample in samples:
            # Add words to vocab
            for word in self._tokenize(sample.user_input):
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

            # Add intent
            if sample.intent not in self.intent_to_idx:
                idx = len(self.intent_to_idx)
                self.intent_to_idx[sample.intent] = idx
                self.idx_to_intent[idx] = sample.intent

    def _text_to_bow(self, text: str) -> List[float]:
        """Convert text to bag-of-words vector."""
        bow = [0.0] * len(self.vocab)
        for word in self._tokenize(text):
            idx = self.vocab.get(word, 0)
            bow[idx] = 1.0
        return bow

    def _intent_to_onehot(self, intent: str) -> List[float]:
        """Convert intent to one-hot vector."""
        onehot = [0.0] * len(self.intent_to_idx)
        idx = self.intent_to_idx.get(intent, 0)
        onehot[idx] = 1.0
        return onehot

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation."""
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax activation."""
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]

    def _forward(self, bow: List[float]) -> Tuple[List[float], List[float]]:
        """Forward pass through network."""
        # Hidden layer
        hidden = []
        for i in range(self.hidden_size):
            activation = self.bias_hidden[i]
            for j, bj in enumerate(bow):
                activation += bj * self.weights_hidden[j][i]
            hidden.append(self._sigmoid(activation))

        # Output layer
        output = []
        for i in range(len(self.intent_to_idx)):
            activation = self.bias_output[i]
            for j, hj in enumerate(hidden):
                activation += hj * self.weights_output[j][i]
            output.append(activation)

        probs = self._softmax(output)
        return hidden, probs

    def _init_weights(self):
        """Initialize network weights."""
        vocab_size = len(self.vocab)
        num_intents = len(self.intent_to_idx)

        # Xavier initialization
        scale_hidden = math.sqrt(2.0 / (vocab_size + self.hidden_size))
        scale_output = math.sqrt(2.0 / (self.hidden_size + num_intents))

        self.weights_hidden = [
            [random.gauss(0, scale_hidden) for _ in range(self.hidden_size)]
            for _ in range(vocab_size)
        ]
        self.weights_output = [
            [random.gauss(0, scale_output) for _ in range(num_intents)]
            for _ in range(self.hidden_size)
        ]
        self.bias_hidden = [0.0] * self.hidden_size
        self.bias_output = [0.0] * num_intents

    def train(self, samples: List[ConversationSample], epochs: int = 100, lr: float = 0.1) -> ConversationTrainingMetrics:
        """Train the classifier."""
        # Build vocabulary
        self._build_vocab(samples)
        self._init_weights()

        metrics = ConversationTrainingMetrics()
        metrics.intents_learned = len(self.intent_to_idx)

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0

            random.shuffle(samples)

            for sample in samples:
                bow = self._text_to_bow(sample.user_input)
                target = self._intent_to_onehot(sample.intent)
                target_idx = self.intent_to_idx[sample.intent]

                # Forward pass
                hidden, probs = self._forward(bow)

                # Compute loss (cross-entropy)
                loss = -math.log(max(probs[target_idx], 1e-10))
                total_loss += loss

                # Check accuracy
                pred_idx = max(range(len(probs)), key=lambda i: probs[i])
                if pred_idx == target_idx:
                    correct += 1

                # Backward pass (simplified gradient descent)
                # Output gradients
                output_grads = [probs[i] - target[i] for i in range(len(probs))]

                # Update output weights
                for j in range(self.hidden_size):
                    for i in range(len(self.intent_to_idx)):
                        self.weights_output[j][i] -= lr * output_grads[i] * hidden[j]

                # Update output bias
                for i in range(len(self.intent_to_idx)):
                    self.bias_output[i] -= lr * output_grads[i]

                # Hidden gradients
                hidden_grads = []
                for j in range(self.hidden_size):
                    grad = sum(output_grads[i] * self.weights_output[j][i] for i in range(len(self.intent_to_idx)))
                    grad *= hidden[j] * (1 - hidden[j])  # Sigmoid derivative
                    hidden_grads.append(grad)

                # Update hidden weights
                for k in range(len(bow)):
                    if bow[k] > 0:
                        for j in range(self.hidden_size):
                            self.weights_hidden[k][j] -= lr * hidden_grads[j] * bow[k]

                # Update hidden bias
                for j in range(self.hidden_size):
                    self.bias_hidden[j] -= lr * hidden_grads[j]

            metrics.loss = total_loss / len(samples)
            metrics.accuracy = correct / len(samples)
            metrics.epochs = epoch + 1
            metrics.samples_trained = len(samples)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: loss={metrics.loss:.4f}, accuracy={metrics.accuracy:.2%}")

        self.is_trained = True
        return metrics

    def predict(self, text: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Predict intent for text."""
        if not self.is_trained:
            return "unknown", 0.0, []

        bow = self._text_to_bow(text)
        _, probs = self._forward(bow)

        # Get best prediction
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best_intent = self.idx_to_intent[best_idx]
        best_prob = probs[best_idx]

        # Get alternatives
        alternatives = sorted(
            [(self.idx_to_intent[i], probs[i]) for i in range(len(probs))],
            key=lambda x: -x[1]
        )[1:4]

        return best_intent, best_prob, alternatives

    def save(self, filepath: Path):
        """Save model to file."""
        data = {
            "vocab": self.vocab,
            "intent_to_idx": self.intent_to_idx,
            "idx_to_intent": {str(k): v for k, v in self.idx_to_intent.items()},
            "weights_hidden": self.weights_hidden,
            "weights_output": self.weights_output,
            "bias_hidden": self.bias_hidden,
            "bias_output": self.bias_output,
            "hidden_size": self.hidden_size,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: Path) -> bool:
        """Load model from file."""
        try:
            with open(filepath) as f:
                data = json.load(f)

            self.vocab = data["vocab"]
            self.intent_to_idx = data["intent_to_idx"]
            self.idx_to_intent = {int(k): v for k, v in data["idx_to_intent"].items()}
            self.weights_hidden = data["weights_hidden"]
            self.weights_output = data["weights_output"]
            self.bias_hidden = data["bias_hidden"]
            self.bias_output = data["bias_output"]
            self.hidden_size = data["hidden_size"]
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False


class ConversationTrainer:
    """Main conversation training service."""

    def __init__(self, data_dir: str = "brain_b_data"):
        self.data_dir = Path(data_dir)
        self.model_dir = self.data_dir / "conversation_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = ConversationDataset(self.data_dir)
        self.classifier = ConversationIntentClassifier()

    def generate_training_data(self, count: int = 2000, use_llm: bool = True) -> int:
        """
        Generate training data using multiple sources.

        Args:
            count: Number of samples to generate
            use_llm: Whether to use LLM for additional diverse samples
        """
        print(f"Generating {count} conversation samples...")

        # Load any existing data
        loaded = self.dataset.load()
        print(f"  Loaded {loaded} existing samples")

        # Generate synthetic samples (rule-based)
        synthetic_count = count // 2 if use_llm else count
        generated = self.dataset.generate_synthetic(synthetic_count)
        print(f"  Generated {generated} rule-based samples")

        # Use LLM to generate diverse samples
        llm_backend = get_llm_backend()
        if use_llm and llm_backend.is_available:
            print(f"  Using {llm_backend.backend} backend for diverse samples...")

            # Collect base phrases for LLM to expand
            base_phrases = []
            for intent, phrases in CONVERSATION_INTENTS.items():
                base_phrases.extend(phrases[:3])

            llm_samples = llm_backend.generate_training_samples(base_phrases, count // 2)

            for sample_data in llm_samples:
                try:
                    sample = ConversationSample(
                        user_input=sample_data.get("user_input", ""),
                        intent=sample_data.get("intent", "unknown"),
                        response=sample_data.get("response", ""),
                        action=sample_data.get("action"),
                        source="llm",
                    )
                    if sample.user_input and sample.intent:
                        self.dataset.samples.append(sample)
                except Exception:
                    continue

            print(f"  Generated {len(llm_samples)} LLM-augmented samples")
        elif use_llm:
            print("  LLM not available, using rule-based generation only")
            # Generate more synthetic to compensate
            extra = self.dataset.generate_synthetic(count // 2)
            print(f"  Generated {extra} additional rule-based samples")

        # Save
        filepath = self.dataset.save()
        print(f"  Saved to {filepath}")
        print(f"  Total samples: {len(self.dataset.samples)}")

        return len(self.dataset.samples)

    def train(self, epochs: int = 100) -> ConversationTrainingMetrics:
        """Train the conversation classifier."""
        print("Training conversation classifier...")

        # Ensure we have data
        if len(self.dataset.samples) == 0:
            self.generate_training_data()

        # Split data
        train_data, test_data = self.dataset.get_train_test_split()
        print(f"  Train samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")

        # Train
        metrics = self.classifier.train(train_data, epochs=epochs)

        # Evaluate on test set
        correct = 0
        for sample in test_data:
            intent, prob, _ = self.classifier.predict(sample.user_input)
            if intent == sample.intent:
                correct += 1

        test_accuracy = correct / len(test_data) if test_data else 0
        print(f"  Test accuracy: {test_accuracy:.2%}")

        # Save model
        model_path = self.model_dir / f"conv_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.classifier.save(model_path)
        print(f"  Model saved to {model_path}")

        # Also save as 'latest'
        latest_path = self.model_dir / "conv_model_latest.json"
        self.classifier.save(latest_path)

        return metrics

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load a trained model."""
        if model_path:
            return self.classifier.load(Path(model_path))

        # Try to load latest
        latest_path = self.model_dir / "conv_model_latest.json"
        if latest_path.exists():
            return self.classifier.load(latest_path)

        return False

    def predict(self, text: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Predict intent and response for text.

        Args:
            text: User input text
            use_llm: Whether to use LLM backend for low-confidence predictions

        Strategy:
        1. Try local trained model first
        2. If confidence < 0.6 and LLM available, use LLM
        3. Fall back to templates for response
        """
        local_result = None

        # Try local model first
        if self.classifier.is_trained or self.load_model():
            intent, confidence, alternatives = self.classifier.predict(text)
            local_result = {
                "intent": intent,
                "confidence": confidence,
                "alternatives": alternatives,
                "backend": "local",
            }

        # If low confidence and LLM available, try LLM
        llm_backend = get_llm_backend()
        if use_llm and llm_backend.is_available:
            if local_result is None or local_result["confidence"] < 0.6:
                llm_result = llm_backend.process(text)

                # LLM gave a good result
                if llm_result.get("confidence", 0) > 0.5 or llm_result.get("intent") != "unknown":
                    # Use LLM result but merge with local alternatives
                    return {
                        "intent": llm_result.get("intent", "unknown"),
                        "confidence": llm_result.get("confidence", 0.8),
                        "response": llm_result.get("response") or self._get_response(llm_result.get("intent")),
                        "action": llm_result.get("action"),
                        "alternatives": local_result["alternatives"] if local_result else [],
                        "backend": llm_backend.backend,
                    }

        # Use local result or fallback
        if local_result:
            return {
                "intent": local_result["intent"],
                "confidence": local_result["confidence"],
                "response": self._get_response(local_result["intent"]),
                "action": INTENT_TO_ACTION.get(local_result["intent"]),
                "alternatives": local_result["alternatives"],
                "backend": "local",
            }

        # No model available
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "response": "I haven't been trained yet. Say 'help' for commands.",
            "action": None,
            "backend": "none",
        }

    def _get_response(self, intent: str) -> str:
        """Get response template for intent."""
        templates = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["unknown"])
        return random.choice(templates)

    def interactive_test(self):
        """Interactive test mode."""
        print("\n" + "="*50)
        print("Conversation Trainer - Interactive Test")
        print("="*50)
        print("Type messages to test intent classification.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ('quit', 'exit', 'q'):
                    break

                result = self.predict(user_input)
                print(f"Intent: {result['intent']} ({result['confidence']:.1%})")
                print(f"Response: {result['response']}")
                if result['action']:
                    print(f"Action: {result['action']}")
                print()

            except KeyboardInterrupt:
                break
            except EOFError:
                break

        print("Goodbye!")


def run_conversation_training(data_dir: str = "brain_b_data", epochs: int = 100) -> Dict:
    """Run conversation training (called by learning partner)."""
    trainer = ConversationTrainer(data_dir)

    # Generate data and train
    total_samples = trainer.generate_training_data(count=2000)
    metrics = trainer.train(epochs=epochs)

    return {
        "status": "success",
        "samples_trained": metrics.samples_trained,
        "intents_learned": metrics.intents_learned,
        "accuracy": metrics.accuracy,
        "epochs": metrics.epochs,
    }


# Singleton for use in conversation handler
_conversation_trainer: Optional[ConversationTrainer] = None


def get_conversation_trainer(data_dir: str = "brain_b_data") -> ConversationTrainer:
    """Get or create conversation trainer singleton."""
    global _conversation_trainer
    if _conversation_trainer is None:
        _conversation_trainer = ConversationTrainer(data_dir)
        _conversation_trainer.load_model()
    return _conversation_trainer


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Conversation Trainer for Brain B")
    parser.add_argument("--generate", type=int, help="Generate N synthetic samples")
    parser.add_argument("--train", action="store_true", help="Train the classifier")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--test", action="store_true", help="Interactive test mode")
    parser.add_argument("--data-dir", default="brain_b_data", help="Data directory")

    args = parser.parse_args()

    trainer = ConversationTrainer(args.data_dir)

    if args.generate:
        trainer.generate_training_data(args.generate)

    if args.train:
        trainer.train(epochs=args.epochs)

    if args.test:
        if not trainer.classifier.is_trained:
            trainer.load_model()
        trainer.interactive_test()

    # If no args, run full training
    if not (args.generate or args.train or args.test):
        result = run_conversation_training(args.data_dir, args.epochs)
        print(f"\nTraining complete: {result}")


if __name__ == "__main__":
    main()
