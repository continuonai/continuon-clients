"""
HOPE Chat - Conversational interface for the JAX CoreModel brain.

This is THE chat interface for ContinuonBrain. It enables the brain to:
- Understand human messages (text → embeddings)
- Process through the CoreModel (thinking)
- Generate responses (embeddings → text)
- Maintain conversation memory via CMS

Used by both web server (Brain Studio) and Flutter app (ContinuonAI).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """A conversation session with history."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, **metadata) -> ChatMessage:
        """Add a message to the session."""
        msg = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self.last_activity = time.time()
        return msg

    def get_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent history as list of dicts."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages[-limit:]
        ]


class HopeChat:
    """
    HOPE Chat Manager - The conversational brain.

    Architecture:
    1. User message → TextEncoder → embedding
    2. Embedding → CoreModel → response embedding
    3. Response embedding → TextDecoder → text response

    The CoreModel's CMS (Continuous Memory System) provides:
    - Short-term context (recent conversation)
    - Long-term memory (learned patterns)
    - Personality consistency
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        config_dir: Optional[Path] = None,
    ):
        self.model_dir = Path(model_dir or "/opt/continuonos/brain/model/seed_stable")
        self.config_dir = Path(config_dir or "/opt/continuonos/brain")

        # Model components (lazy loaded)
        self._encoder = None
        self._decoder = None
        self._core_model = None
        self._core_params = None
        self._core_state = None

        # Sessions
        self._sessions: Dict[str, ChatSession] = {}
        self._default_session_id = "default"

        # System personality
        self.system_prompt = (
            "You are HOPE, a helpful robot assistant. "
            "You help humans with tasks, answer questions, and control robot systems. "
            "You are friendly, knowledgeable, and safety-conscious."
        )

        logger.info("HopeChat initialized")

    def _ensure_loaded(self):
        """Lazy load model components."""
        if self._encoder is not None:
            return

        logger.info("Loading HOPE Chat components...")

        # Load text encoder
        try:
            from continuonbrain.jax_models.text_encoder import SelfContainedEncoder, TextEncoderConfig

            encoder_path = self.model_dir / "text_encoder"
            if encoder_path.exists():
                self._encoder = SelfContainedEncoder()
                self._encoder.load(encoder_path)
                logger.info(f"Loaded text encoder from {encoder_path}")
            else:
                # Create new encoder
                config = TextEncoderConfig(
                    vocab_size=8000,
                    max_length=128,
                    embed_dim=128,  # Match CoreModel obs_dim
                    hidden_dim=256,
                    num_layers=2,
                )
                self._encoder = SelfContainedEncoder(config)
                self._encoder.init()
                logger.info("Created new text encoder")
        except Exception as e:
            logger.error(f"Failed to load encoder: {e}")
            self._encoder = None

        # Load/create text decoder
        try:
            self._decoder = TextDecoder(
                vocab_size=8000,
                embed_dim=128,
                hidden_dim=256,
                max_length=128,
            )
            self._decoder.init()
            logger.info("Text decoder initialized")
        except Exception as e:
            logger.error(f"Failed to create decoder: {e}")
            self._decoder = None

        # Load CoreModel
        try:
            from continuonbrain.jax_models.core_model import make_core_model
            from continuonbrain.jax_models.config_presets import get_config_for_preset

            config = get_config_for_preset("pi5")
            rng_key = jax.random.PRNGKey(42)

            model, params = make_core_model(
                rng_key,
                obs_dim=128,
                action_dim=32,
                output_dim=128,  # Match embed_dim for text generation
                config=config,
            )
            self._core_model = model
            self._core_params = params

            # Initialize CMS state
            self._core_state = {
                "cms_memory": jnp.zeros((1, config.cms_levels, config.cms_slots, config.d_s)),
                "hidden_state": jnp.zeros((1, config.d_s)),
                "prev_action": jnp.zeros((1, 32)),
            }

            logger.info("CoreModel loaded")
        except Exception as e:
            logger.error(f"Failed to load CoreModel: {e}")
            self._core_model = None

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate a response.

        Args:
            message: User's message text
            session_id: Session ID for conversation tracking
            history: Optional conversation history

        Returns:
            Dict with response, session_id, and metadata
        """
        start_time = time.time()

        # Ensure models are loaded
        self._ensure_loaded()

        # Get or create session
        session_id = session_id or self._default_session_id
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id=session_id)
        session = self._sessions[session_id]

        # Add user message to session
        session.add_message("user", message)

        # Generate response
        if self._core_model is None or self._encoder is None:
            # Fallback response if model not loaded
            response_text = self._generate_fallback_response(message)
            response_agent = "fallback"
        else:
            response_text, response_embedding = self._generate_response(message, session)
            response_agent = "hope_brain"

        # Add response to session
        session.add_message("assistant", response_text)

        duration = time.time() - start_time

        return {
            "success": True,
            "response": response_text,
            "session_id": session_id,
            "agent": response_agent,
            "duration_ms": duration * 1000,
            "turn_count": len(session.messages),
        }

    def _generate_response(
        self,
        message: str,
        session: ChatSession,
    ) -> Tuple[str, np.ndarray]:
        """Generate response using the CoreModel brain."""

        # 1. Encode user message to embedding
        message_embedding = self._encoder.encode([message])[0]  # [embed_dim]

        # 2. Build context from recent history
        context_texts = []
        for msg in session.messages[-5:]:  # Last 5 messages
            context_texts.append(f"{msg.role}: {msg.content}")
        context = " ".join(context_texts)
        context_embedding = self._encoder.encode([context])[0]

        # 3. Combine message and context as observation
        obs = jnp.array(message_embedding).reshape(1, -1)  # [1, obs_dim]
        prev_action = self._core_state["prev_action"]

        # 4. Run through CoreModel
        from continuonbrain.jax_models.core_model import CoreModel

        # Forward pass
        output, new_state = self._core_model.apply(
            self._core_params,
            obs,
            prev_action,
            self._core_state["cms_memory"],
            self._core_state["hidden_state"],
            training=False,
        )

        # Update state
        self._core_state["cms_memory"] = new_state["cms_memory"]
        self._core_state["hidden_state"] = new_state["hidden_state"]
        self._core_state["prev_action"] = output["action"]

        # 5. Get response embedding from output
        response_embedding = np.array(output["predicted_next_obs"][0])

        # 6. Decode to text
        if self._decoder is not None:
            response_text = self._decoder.decode(response_embedding)
        else:
            response_text = self._embedding_to_template_response(message, response_embedding)

        return response_text, response_embedding

    def _embedding_to_template_response(
        self,
        message: str,
        embedding: np.ndarray,
    ) -> str:
        """Convert response embedding to text using templates (fallback)."""
        # Use embedding magnitude and direction to select response type
        magnitude = np.linalg.norm(embedding)
        direction = embedding[:4]  # First 4 dims as "intent"

        # Simple template selection based on embedding
        if "hello" in message.lower() or "hi" in message.lower():
            return "Hello! I'm HOPE, your robot assistant. How can I help you today?"
        elif "help" in message.lower():
            return "I'm here to help! I can assist with robot control, answer questions, and help with tasks."
        elif "move" in message.lower() or "go" in message.lower():
            return "I understand you want me to move. Please specify the direction or destination."
        elif "stop" in message.lower():
            return "Stopping all motion. Safety first!"
        elif "?" in message:
            return f"That's a good question. Based on my understanding, {self._infer_answer(message)}"
        else:
            return f"I understand. {self._context_aware_response(message, magnitude)}"

    def _infer_answer(self, question: str) -> str:
        """Generate an answer to a question."""
        question_lower = question.lower()
        if "what" in question_lower and "name" in question_lower:
            return "my name is HOPE - Hierarchical Orchestrated Predictive Engine."
        elif "how" in question_lower and "are" in question_lower:
            return "I'm functioning well and ready to assist!"
        elif "can" in question_lower and "you" in question_lower:
            return "I can help with robot control, conversations, and learning tasks."
        else:
            return "let me think about that based on my training."

    def _context_aware_response(self, message: str, magnitude: float) -> str:
        """Generate context-aware response."""
        if magnitude > 1.0:
            return "I'm actively processing your request."
        else:
            return "How can I assist you further?"

    def _generate_fallback_response(self, message: str) -> str:
        """Generate response when model is not available."""
        message_lower = message.lower()

        if "hello" in message_lower or "hi" in message_lower:
            return "Hello! I'm HOPE. My neural core is still initializing, but I'm here to help."
        elif "help" in message_lower:
            return "I'm here to assist! What do you need help with?"
        elif "?" in message:
            return "That's a great question. I'm still loading my full capabilities, but I'll do my best to help."
        else:
            return "I hear you. How can I assist?"

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        return self._sessions.get(session_id)

    def clear_session(self, session_id: str):
        """Clear a chat session."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_status(self) -> Dict[str, Any]:
        """Get chat service status."""
        return {
            "ready": self._core_model is not None,
            "encoder_loaded": self._encoder is not None,
            "decoder_loaded": self._decoder is not None,
            "active_sessions": len(self._sessions),
            "model_dir": str(self.model_dir),
        }


class TextDecoder:
    """
    Simple text decoder for generating responses from embeddings.

    Uses a small transformer-like architecture to decode
    embedding vectors back to text tokens.
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        max_length: int = 128,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self._model = None
        self._params = None
        self._tokenizer = None

    def init(self):
        """Initialize decoder model."""
        from continuonbrain.jax_models.text_encoder import SimpleTokenizer

        self._tokenizer = SimpleTokenizer(
            vocab_size=self.vocab_size,
            max_length=self.max_length,
        )

        # Simple MLP decoder
        rng_key = jax.random.PRNGKey(123)

        # Initialize weights
        k1, k2, k3 = jax.random.split(rng_key, 3)
        self._params = {
            "embed_to_hidden": jax.random.normal(k1, (self.embed_dim, self.hidden_dim)) * 0.02,
            "hidden_to_vocab": jax.random.normal(k2, (self.hidden_dim, self.vocab_size)) * 0.02,
            "output_bias": jnp.zeros(self.vocab_size),
        }

        logger.info(f"TextDecoder initialized: {self.vocab_size} vocab, {self.embed_dim} embed")

    def decode(self, embedding: np.ndarray, max_tokens: int = 50) -> str:
        """Decode embedding to text."""
        if self._params is None:
            self.init()

        embedding = jnp.array(embedding)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        tokens = []
        hidden = jnp.tanh(embedding @ self._params["embed_to_hidden"])

        for _ in range(max_tokens):
            # Predict next token
            logits = hidden @ self._params["hidden_to_vocab"] + self._params["output_bias"]
            probs = jax.nn.softmax(logits, axis=-1)

            # Sample token (or take argmax for deterministic)
            token_id = int(jnp.argmax(probs[0]))

            if token_id == self._tokenizer.token_to_id.get('<EOS>', 3):
                break

            if token_id > 3:  # Skip special tokens
                tokens.append(token_id)

            # Update hidden (simple recurrence)
            token_embed = jax.nn.one_hot(token_id, self.vocab_size)
            hidden = jnp.tanh(hidden * 0.9 + token_embed @ self._params["hidden_to_vocab"].T * 0.1)

        # Decode tokens to text
        return self._tokenizer.decode(tokens)


# Convenience function
def create_hope_chat(
    model_dir: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> HopeChat:
    """Create a HopeChat instance."""
    chat = HopeChat(model_dir=model_dir, config_dir=config_dir)
    return chat
