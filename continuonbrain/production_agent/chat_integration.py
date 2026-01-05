"""
Chat Integration Layer for Seed Model

Bridges the gap between:
1. Text input (user messages)
2. Seed model (HOPE architecture)
3. Text output (robot responses)
4. Action output (robot commands)

Uses a text encoder to convert language to obs_dim embeddings,
and the seed model for reasoning/action generation.
"""

import json
import time
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    action_output: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from the chat integration layer."""
    text: str
    action: Optional[np.ndarray]
    confidence: float
    intent: str
    latency_ms: float
    model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Intent templates for action-grounded responses
INTENT_TEMPLATES = {
    "move": {
        "patterns": ["move", "go", "drive", "navigate", "forward", "backward", "turn"],
        "response_templates": [
            "Moving {direction} now.",
            "Navigating to {target}.",
            "Adjusting position.",
        ],
        "action_channels": [0, 1, 2, 3],  # Mobility channels
    },
    "look": {
        "patterns": ["look", "see", "watch", "observe", "camera", "focus"],
        "response_templates": [
            "Looking {direction}.",
            "Adjusting camera view.",
            "Focusing on {target}.",
        ],
        "action_channels": [14, 15, 16, 17],  # Head channels
    },
    "grab": {
        "patterns": ["grab", "pick", "grasp", "hold", "take", "gripper"],
        "response_templates": [
            "Reaching for {target}.",
            "Grasping object.",
            "Adjusting grip.",
        ],
        "action_channels": [6, 7, 8, 9, 10, 11],  # Arm channels
    },
    "stop": {
        "patterns": ["stop", "halt", "freeze", "wait", "pause"],
        "response_templates": [
            "Stopping all motion.",
            "Holding position.",
            "Paused.",
        ],
        "action_channels": [24, 25],  # Safety channels
    },
    "greet": {
        "patterns": ["hello", "hi", "hey", "greetings"],
        "response_templates": [
            "Hello! How can I help?",
            "Hi there! Ready to assist.",
            "Greetings! What would you like me to do?",
        ],
        "action_channels": [18, 21],  # LED and emotion
    },
    "status": {
        "patterns": ["status", "how are you", "state", "report"],
        "response_templates": [
            "All systems operational.",
            "Ready for commands.",
            "Status: active and ready.",
        ],
        "action_channels": [],
    },
    "unknown": {
        "patterns": [],
        "response_templates": [
            "I'm not sure how to help with that.",
            "Could you rephrase that?",
            "Let me think about that...",
        ],
        "action_channels": [],
    },
}


class TextEncoder:
    """
    Encodes text to fixed-dimension embeddings.

    Uses sentence-transformers if available, otherwise falls back
    to a simple hash-based encoding.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        output_dim: int = 128,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.output_dim = output_dim
        self.normalize = normalize
        self._model = None
        self._projection = None

    def _load_model(self) -> None:
        """Lazy load the encoder model."""
        if self._model is not None:
            return

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.model_name)
                model_dim = self._model.get_sentence_embedding_dimension()

                # Create projection matrix if dimensions don't match
                if model_dim != self.output_dim:
                    np.random.seed(42)
                    self._projection = np.random.randn(model_dim, self.output_dim) * 0.1
                    self._projection = self._projection / np.linalg.norm(self._projection, axis=0)

                print(f"Loaded text encoder: {self.model_name} ({model_dim}d -> {self.output_dim}d)")
                return
            except Exception as e:
                print(f"Failed to load sentence-transformers: {e}")

        # Fallback to hash-based encoding
        print("Using hash-based text encoding (install sentence-transformers for better quality)")
        self._model = "hash"

    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding."""
        self._load_model()

        if self._model == "hash":
            return self._hash_encode(text)

        # Use sentence-transformers
        embedding = self._model.encode([text], convert_to_numpy=True)[0]

        # Project to output_dim if needed
        if self._projection is not None:
            embedding = embedding @ self._projection

        if self.normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.astype(np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts."""
        self._load_model()

        if self._model == "hash":
            return np.stack([self._hash_encode(t) for t in texts])

        embeddings = self._model.encode(texts, convert_to_numpy=True)

        if self._projection is not None:
            embeddings = embeddings @ self._projection

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def _hash_encode(self, text: str) -> np.ndarray:
        """Simple hash-based encoding as fallback."""
        import hashlib

        # Create deterministic embedding from text
        text_bytes = text.lower().encode('utf-8')
        hash_bytes = hashlib.sha256(text_bytes).digest()

        # Expand hash to output_dim
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        embedding = np.random.randn(self.output_dim).astype(np.float32)

        if self.normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding


class ResponseGenerator:
    """
    Generates text responses from action outputs.

    Maps seed model output patterns to appropriate responses.
    """

    def __init__(self):
        self.intent_templates = INTENT_TEMPLATES

    def classify_intent(self, text: str) -> str:
        """Classify the intent of a message."""
        text_lower = text.lower()

        for intent, config in self.intent_templates.items():
            if intent == "unknown":
                continue
            for pattern in config["patterns"]:
                if pattern in text_lower:
                    return intent

        return "unknown"

    def generate_response(
        self,
        intent: str,
        action_output: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a text response based on intent and action."""
        config = self.intent_templates.get(intent, self.intent_templates["unknown"])
        templates = config["response_templates"]

        # Select template based on action characteristics
        template_idx = 0
        if action_output is not None and len(config["action_channels"]) > 0:
            # Use action magnitude to select response
            relevant_actions = [action_output[i] for i in config["action_channels"] if i < len(action_output)]
            if relevant_actions:
                magnitude = np.mean(np.abs(relevant_actions))
                template_idx = min(int(magnitude * len(templates)), len(templates) - 1)

        template = templates[template_idx]

        # Fill in placeholders
        context = context or {}
        try:
            response = template.format(
                direction=context.get("direction", "as requested"),
                target=context.get("target", "the target"),
            )
        except KeyError:
            response = template

        return response


class ChatIntegrationLayer:
    """
    Full chat integration for the seed model.

    Flow:
    1. User text -> Text encoder -> obs embedding
    2. obs embedding -> Seed model -> action output
    3. action output -> Response generator -> text response
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        encoder_model: str = "all-MiniLM-L6-v2",
        obs_dim: int = 128,
        action_dim: int = 32,
        output_dim: int = 32,
        max_history: int = 10,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.max_history = max_history

        # Components
        self.text_encoder = TextEncoder(
            model_name=encoder_model,
            output_dim=obs_dim,
        )
        self.response_generator = ResponseGenerator()

        # Model
        self.model = None
        self.params = None
        self.config = None
        self._model_loaded = False

        if model_path:
            self.load_model(model_path)

        # State
        self.history: List[ChatMessage] = []
        self._model_state = None

    def load_model(self, model_path: Path) -> bool:
        """Load the seed model."""
        if not JAX_AVAILABLE:
            print("JAX not available, running in text-only mode")
            return False

        model_path = Path(model_path)

        try:
            # Try loading manifest
            manifest_path = model_path / "model_manifest.json" if model_path.is_dir() else model_path.parent / "model_manifest.json"
            params_path = model_path / "params_step_16.pkl" if model_path.is_dir() else model_path

            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                self.obs_dim = manifest.get("input_dims", {}).get("obs_dim", self.obs_dim)
                self.action_dim = manifest.get("input_dims", {}).get("action_dim", self.action_dim)
                self.output_dim = manifest.get("output_dim", self.output_dim)

                from continuonbrain.jax_models.config import CoreModelConfig
                self.config = CoreModelConfig(**manifest["config"])

            # Load params
            if params_path.exists():
                with open(params_path, 'rb') as f:
                    data = pickle.load(f)
                    self.params = data if 'params' in data else {'params': data}

            # Create model
            from continuonbrain.jax_models.core_model import CoreModel
            self.model = CoreModel(
                config=self.config,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                output_dim=self.output_dim,
            )

            self._model_loaded = True
            print(f"Loaded seed model from {model_path}")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def _init_model_state(self) -> Dict[str, Any]:
        """Initialize model state."""
        if not self._model_loaded:
            return {}

        return {
            's': jnp.zeros((1, self.config.d_s)),
            'w': jnp.zeros((1, self.config.d_w)),
            'p': jnp.zeros((1, self.config.d_p)),
            'cms_memories': [
                jnp.zeros((1, sz, dim))
                for sz, dim in zip(self.config.cms_sizes, self.config.cms_dims)
            ],
            'cms_keys': [
                jnp.zeros((1, sz, self.config.d_k))
                for sz in self.config.cms_sizes
            ],
        }

    def _run_model_inference(
        self,
        obs: np.ndarray,
        state: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run seed model inference."""
        if not self._model_loaded:
            # Return zero action if model not loaded
            return np.zeros(self.output_dim), state

        obs_jax = jnp.array(obs.reshape(1, -1))
        action = jnp.zeros((1, self.action_dim))
        reward = jnp.zeros((1, 1))

        output, info = self.model.apply(
            self.params,
            obs_jax, action, reward,
            state['s'], state['w'], state['p'],
            state['cms_memories'], state['cms_keys'],
        )

        new_state = {
            's': info['fast_state'],
            'w': info['wave_state'],
            'p': info['particle_state'],
            'cms_memories': info['cms_memories'],
            'cms_keys': info['cms_keys'],
        }

        return np.array(output[0]), new_state

    def chat(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        reset_state: bool = False,
    ) -> ChatResponse:
        """
        Process a chat message and generate response.

        Args:
            user_message: The user's input text
            context: Optional context (e.g., detected objects, location)
            reset_state: Whether to reset model state

        Returns:
            ChatResponse with text and optional action
        """
        start_time = time.time()

        # Initialize state if needed
        if self._model_state is None or reset_state:
            self._model_state = self._init_model_state()

        # Encode text
        embedding = self.text_encoder.encode(user_message)

        # Classify intent
        intent = self.response_generator.classify_intent(user_message)

        # Run model inference
        action_output, self._model_state = self._run_model_inference(
            embedding, self._model_state
        )

        # Generate response
        response_text = self.response_generator.generate_response(
            intent, action_output, context
        )

        # Calculate confidence based on action magnitude
        confidence = float(np.mean(np.abs(action_output)))

        # Store in history
        user_msg = ChatMessage(
            role="user",
            content=user_message,
            embedding=embedding,
        )
        assistant_msg = ChatMessage(
            role="assistant",
            content=response_text,
            action_output=action_output,
            metadata={"intent": intent, "confidence": confidence},
        )
        self.history.append(user_msg)
        self.history.append(assistant_msg)

        # Trim history
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        latency_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            text=response_text,
            action=action_output if self._model_loaded else None,
            confidence=confidence,
            intent=intent,
            latency_ms=latency_ms,
            model_used="seed_model" if self._model_loaded else "template",
            metadata={
                "embedding_norm": float(np.linalg.norm(embedding)),
                "action_norm": float(np.linalg.norm(action_output)) if action_output is not None else 0,
            },
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Get chat history."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "intent": msg.metadata.get("intent"),
            }
            for msg in self.history
        ]

    def reset(self) -> None:
        """Reset chat state."""
        self.history.clear()
        self._model_state = None


class ChatServer:
    """
    WebSocket/HTTP server for chat integration.

    Provides real-time chat interface for ContinuonAI and web UI.
    """

    def __init__(self, chat_layer: ChatIntegrationLayer, port: int = 8081):
        self.chat_layer = chat_layer
        self.port = port

    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming chat message."""
        user_text = message.get("text", "")
        context = message.get("context", {})
        reset = message.get("reset", False)

        response = self.chat_layer.chat(user_text, context, reset)

        return {
            "type": "chat_response",
            "text": response.text,
            "action": response.action.tolist() if response.action is not None else None,
            "confidence": response.confidence,
            "intent": response.intent,
            "latency_ms": response.latency_ms,
            "model": response.model_used,
        }


if __name__ == "__main__":
    # Test the chat integration
    print("Testing Chat Integration Layer")
    print("=" * 60)

    # Create chat layer (without model for demo)
    chat = ChatIntegrationLayer()

    # Test messages
    test_messages = [
        "Hello!",
        "Move forward",
        "Look to the left",
        "Pick up the red cube",
        "Stop!",
        "What's your status?",
        "Do something random",
    ]

    for msg in test_messages:
        print(f"\nUser: {msg}")
        response = chat.chat(msg)
        print(f"Bot: {response.text}")
        print(f"    Intent: {response.intent}, Confidence: {response.confidence:.2f}")
        print(f"    Latency: {response.latency_ms:.2f}ms, Model: {response.model_used}")
