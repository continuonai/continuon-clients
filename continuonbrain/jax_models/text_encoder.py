"""
Self-Contained Text Encoder

A lightweight text embedding layer that:
- Works without external transformers
- Uses trainable character/subword embeddings
- Fits in <50MB memory
- Can be trained as part of WaveCore

This replaces EmbeddingGemma-300m for fully self-contained operation.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn


# ============= Tokenizer (BPE-like) =============

class SimpleTokenizer:
    """
    Simple BPE-style tokenizer.
    Can be trained on corpus or loaded from pretrained vocab.
    """
    
    def __init__(self, vocab_size: int = 8000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        # Initialize with character-level vocab
        self._init_vocab()
    
    def _init_vocab(self):
        """Initialize vocabulary with basic characters."""
        self.token_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Add printable ASCII
        for i, char in enumerate(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789 .,!?;:\'"()-_/\\@#$%&*+=<>[]{}|~`'
        ):
            idx = len(self.token_to_id)
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
        
        # Common subwords (to be expanded via training)
        common_subwords = [
            'the', 'ing', 'and', 'tion', 'er', 'ed', 'es', 's',
            'move', 'pick', 'up', 'down', 'left', 'right', 'turn',
            'stop', 'go', 'start', 'end', 'open', 'close', 'grab',
            'robot', 'arm', 'gripper', 'object', 'target', 'position',
            'forward', 'backward', 'rotate', 'place', 'hold', 'release',
            'emergency', 'danger', 'safety', 'halt', 'abort',
            'navigate', 'kitchen', 'room', 'door', 'table',
            'hello', 'help', 'please', 'thank', 'you', 'what', 'where',
        ]
        
        for word in common_subwords:
            if word not in self.token_to_id:
                idx = len(self.token_to_id)
                if idx < self.vocab_size:
                    self.token_to_id[word] = idx
                    self.id_to_token[idx] = word
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        text = text.lower().strip()
        tokens = [self.token_to_id[self.bos_token]]
        
        i = 0
        while i < len(text) and len(tokens) < self.max_length - 1:
            # Try to match longest subword first
            matched = False
            for length in range(min(10, len(text) - i), 0, -1):
                subword = text[i:i+length]
                if subword in self.token_to_id:
                    tokens.append(self.token_to_id[subword])
                    i += length
                    matched = True
                    break
            
            if not matched:
                # Fall back to character
                char = text[i]
                if char in self.token_to_id:
                    tokens.append(self.token_to_id[char])
                else:
                    tokens.append(self.token_to_id[self.unk_token])
                i += 1
        
        tokens.append(self.token_to_id[self.eos_token])
        
        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.token_to_id[self.pad_token])
        
        return tokens[:self.max_length]
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts."""
        return np.array([self.encode(t) for t in texts])
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if token not in [self.pad_token, self.bos_token, self.eos_token]:
                    tokens.append(token)
        return ''.join(tokens)
    
    def save(self, path: Path):
        """Save tokenizer vocabulary."""
        with open(path, 'w') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'token_to_id': self.token_to_id,
            }, f)
    
    def load(self, path: Path):
        """Load tokenizer vocabulary."""
        with open(path) as f:
            data = json.load(f)
            self.vocab_size = data['vocab_size']
            self.max_length = data['max_length']
            self.token_to_id = data['token_to_id']
            self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}


# ============= Text Encoder Model =============

class TextEncoderConfig:
    """Configuration for text encoder."""
    
    def __init__(
        self,
        vocab_size: int = 8000,
        max_length: int = 128,
        embed_dim: int = 768,  # Match EmbeddingGemma output
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,  # For optional attention
        dropout_rate: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate


class TextEncoder(nn.Module):
    """
    Lightweight text encoder.
    
    Architecture:
    - Token embedding (vocab_size x hidden_dim)
    - Positional embedding (max_length x hidden_dim)
    - Stack of Dense layers with GELU
    - Mean pooling over sequence
    - Final projection to embed_dim
    
    Total params: ~4-8M (vs 300M for EmbeddingGemma)
    """
    
    config: TextEncoderConfig
    
    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Args:
            input_ids: [batch, seq_len] token IDs
            training: Whether in training mode
        
        Returns:
            embeddings: [batch, embed_dim] sentence embeddings
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_dim,
            name='token_embed'
        )(input_ids)  # [B, L, H]
        
        # Positional embeddings
        positions = jnp.arange(seq_len)
        pos_embed = nn.Embed(
            num_embeddings=self.config.max_length,
            features=self.config.hidden_dim,
            name='pos_embed'
        )(positions)  # [L, H]
        
        # Combine
        x = token_embed + pos_embed  # [B, L, H]
        
        # Layer norm
        x = nn.LayerNorm(name='ln_input')(x)
        
        # Process with dense layers
        for i in range(self.config.num_layers):
            # Dense block with residual
            residual = x
            x = nn.Dense(self.config.hidden_dim * 2, name=f'dense_{i}_up')(x)
            x = nn.gelu(x)
            x = nn.Dense(self.config.hidden_dim, name=f'dense_{i}_down')(x)
            
            if training:
                x = nn.Dropout(self.config.dropout_rate)(x, deterministic=False)
            
            x = x + residual
            x = nn.LayerNorm(name=f'ln_{i}')(x)
        
        # Create attention mask (ignore PAD tokens = id 0)
        mask = (input_ids != 0).astype(jnp.float32)  # [B, L]
        mask = mask[:, :, None]  # [B, L, 1]
        
        # Masked mean pooling
        x = x * mask  # Zero out padding
        x = jnp.sum(x, axis=1) / (jnp.sum(mask, axis=1) + 1e-8)  # [B, H]
        
        # Project to output dimension
        x = nn.Dense(self.config.embed_dim, name='output_proj')(x)
        
        # L2 normalize
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        
        return x


class SelfContainedEncoder:
    """
    Complete self-contained text encoder.
    Combines tokenizer and neural encoder.
    """
    
    def __init__(self, config: Optional[TextEncoderConfig] = None):
        self.config = config or TextEncoderConfig()
        self.tokenizer = SimpleTokenizer(
            vocab_size=self.config.vocab_size,
            max_length=self.config.max_length
        )
        self.model = TextEncoder(config=self.config)
        self.params = None
        self._initialized = False
    
    def init(self, rng_key: Optional[jnp.ndarray] = None):
        """Initialize model parameters."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        
        dummy_input = jnp.zeros((1, self.config.max_length), dtype=jnp.int32)
        self.params = self.model.init(rng_key, dummy_input)
        self._initialized = True
        
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"TextEncoder initialized: {param_count:,} params ({param_count * 4 / 1e6:.1f} MB)")
    
    def encode(self, texts: List[str], convert_to_numpy: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        if not self._initialized:
            self.init()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        input_ids = self.tokenizer.encode_batch(texts)
        input_ids = jnp.array(input_ids)
        
        # Encode
        embeddings = self.model.apply(self.params, input_ids, training=False)
        
        if convert_to_numpy:
            return np.array(embeddings)
        return embeddings
    
    def save(self, path: Path):
        """Save encoder (tokenizer + weights)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer.save(path / 'tokenizer.json')
        
        with open(path / 'config.json', 'w') as f:
            json.dump(self.config.__dict__, f)
        
        with open(path / 'params.pkl', 'wb') as f:
            pickle.dump(self.params, f)
    
    def load(self, path: Path):
        """Load encoder from saved files."""
        path = Path(path)
        
        self.tokenizer.load(path / 'tokenizer.json')
        
        with open(path / 'config.json') as f:
            config_dict = json.load(f)
            self.config = TextEncoderConfig(**config_dict)
        
        with open(path / 'params.pkl', 'rb') as f:
            self.params = pickle.load(f)
        
        self._initialized = True
    
    @property
    def param_count(self) -> int:
        if not self._initialized:
            return 0
        return sum(x.size for x in jax.tree_util.tree_leaves(self.params))


def create_encoder(embed_dim: int = 768) -> SelfContainedEncoder:
    """Create a self-contained encoder with default config."""
    config = TextEncoderConfig(
        vocab_size=8000,
        max_length=128,
        embed_dim=embed_dim,
        hidden_dim=512,
        num_layers=2,
    )
    encoder = SelfContainedEncoder(config)
    encoder.init()
    return encoder

