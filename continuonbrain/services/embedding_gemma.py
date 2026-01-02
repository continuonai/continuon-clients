"""
EmbeddingGemma-300m Integration for Semantic Search

State-of-the-art 300M parameter embedding model from Google.
Produces 768-dimensional embeddings (or smaller via MRL truncation).

Reference: https://huggingface.co/google/embeddinggemma-300m
Paper: https://arxiv.org/abs/2509.20354

Usage:
    from continuonbrain.services.embedding_gemma import get_embedding_model, EmbeddingGemmaEncoder
    
    encoder = get_embedding_model()
    embedding = encoder.encode_query("What is the Red Planet?")
    doc_embeddings = encoder.encode_document(["Mars is...", "Venus is..."])
"""
import os
import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_embedding_model = None
_model_load_attempted = False

# Model configuration
EMBEDDING_GEMMA_MODEL_ID = "google/embeddinggemma-300m"
DEFAULT_EMBEDDING_DIM = 768  # Full precision; can truncate to 512, 256, 128 via MRL


class EmbeddingGemmaEncoder:
    """
    Wrapper for EmbeddingGemma-300m using sentence-transformers.
    
    Features:
    - Separate encode_query() and encode_document() methods (as recommended)
    - MRL support for dimension reduction (768 → 512 → 256 → 128)
    - Task-specific prompts for optimal performance
    """
    
    # Task prompts (from HuggingFace model card)
    TASK_PROMPTS = {
        "retrieval": "task: search result | query: ",
        "question_answering": "task: question answering | query: ",
        "fact_verification": "task: fact checking | query: ",
        "classification": "task: classification | query: ",
        "clustering": "task: clustering | query: ",
        "semantic_similarity": "task: sentence similarity | query: ",
        "code_retrieval": "task: code retrieval | query: ",
    }
    
    DOCUMENT_PROMPT = "title: none | text: "
    
    def __init__(
        self,
        model_id: str = EMBEDDING_GEMMA_MODEL_ID,
        output_dim: int = DEFAULT_EMBEDDING_DIM,
        device: str = "cpu",
        trust_remote_code: bool = True,
    ):
        """
        Initialize EmbeddingGemma encoder.
        
        Args:
            model_id: HuggingFace model identifier
            output_dim: Output embedding dimension (768, 512, 256, or 128)
            device: Device to run on ("cpu", "cuda", etc.)
            trust_remote_code: Whether to trust remote code from HuggingFace
        """
        self.model_id = model_id
        self.output_dim = output_dim
        self.device = device
        self.model = None
        self._loaded = False
        
        # Validate output dimension (MRL supported sizes)
        if output_dim not in (768, 512, 256, 128):
            logger.warning(f"Non-standard output_dim={output_dim}. Recommend 768, 512, 256, or 128.")
        
        # Check for HuggingFace token (required for gated model)
        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            logger.warning(
                "HUGGINGFACE_TOKEN not set. EmbeddingGemma requires accepting license at "
                "https://huggingface.co/google/embeddinggemma-300m"
            )
    
    def load(self) -> bool:
        """Load the model. Returns True on success."""
        if self._loaded:
            return True
            
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading EmbeddingGemma from {self.model_id}...")
            
            # Load model with trust_remote_code for Gemma architecture
            self.model = SentenceTransformer(
                self.model_id,
                device=self.device,
                trust_remote_code=True,
            )
            
            self._loaded = True
            logger.info(f"✅ EmbeddingGemma loaded successfully (dim={self.output_dim})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load EmbeddingGemma: {e}")
            return False
    
    def _ensure_loaded(self) -> bool:
        """Ensure model is loaded."""
        if not self._loaded:
            return self.load()
        return True
    
    def _truncate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Truncate embedding using Matryoshka Representation Learning (MRL).
        
        MRL allows truncating the 768-dim embedding to smaller sizes
        while maintaining quality.
        """
        if self.output_dim >= 768:
            return embedding
        
        # Truncate to desired dimension
        truncated = embedding[..., :self.output_dim]
        
        # Re-normalize (important for MRL)
        if truncated.ndim == 1:
            norm = np.linalg.norm(truncated)
            if norm > 0:
                truncated = truncated / norm
        else:
            norms = np.linalg.norm(truncated, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            truncated = truncated / norms
        
        return truncated
    
    def encode_query(
        self,
        query: Union[str, List[str]],
        task: str = "retrieval",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode query/queries for retrieval.
        
        Args:
            query: Single query or list of queries
            task: Task type for prompt selection
            normalize: Whether to L2-normalize the output
            
        Returns:
            Embedding array of shape (dim,) or (n_queries, dim)
        """
        if not self._ensure_loaded():
            # Return zero vector on failure
            dim = self.output_dim
            if isinstance(query, str):
                return np.zeros(dim, dtype=np.float32)
            return np.zeros((len(query), dim), dtype=np.float32)
        
        # Apply task-specific prompt
        prompt_prefix = self.TASK_PROMPTS.get(task, self.TASK_PROMPTS["retrieval"])
        
        if isinstance(query, str):
            prompted_query = f"{prompt_prefix}{query}"
        else:
            prompted_query = [f"{prompt_prefix}{q}" for q in query]
        
        # Use sentence-transformers encode_query if available (EmbeddingGemma has it)
        try:
            embedding = self.model.encode_query(prompted_query)
        except AttributeError:
            # Fallback to regular encode
            embedding = self.model.encode(
                prompted_query,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
        
        return self._truncate_embedding(np.asarray(embedding))
    
    def encode_document(
        self,
        documents: Union[str, List[str]],
        titles: Optional[Union[str, List[str]]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode document(s) for retrieval.
        
        Args:
            documents: Single document or list of documents
            titles: Optional titles for documents (improves quality)
            normalize: Whether to L2-normalize the output
            
        Returns:
            Embedding array of shape (dim,) or (n_docs, dim)
        """
        if not self._ensure_loaded():
            dim = self.output_dim
            if isinstance(documents, str):
                return np.zeros(dim, dtype=np.float32)
            return np.zeros((len(documents), dim), dtype=np.float32)
        
        # Apply document prompt
        if isinstance(documents, str):
            if titles:
                title = titles if isinstance(titles, str) else titles[0]
                prompted_doc = f"title: {title} | text: {documents}"
            else:
                prompted_doc = f"{self.DOCUMENT_PROMPT}{documents}"
        else:
            if titles:
                title_list = [titles] * len(documents) if isinstance(titles, str) else titles
                prompted_doc = [
                    f"title: {t} | text: {d}" for t, d in zip(title_list, documents)
                ]
            else:
                prompted_doc = [f"{self.DOCUMENT_PROMPT}{d}" for d in documents]
        
        # Use sentence-transformers encode_document if available
        try:
            embedding = self.model.encode_document(prompted_doc)
        except AttributeError:
            # Fallback to regular encode
            embedding = self.model.encode(
                prompted_doc,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
        
        return self._truncate_embedding(np.asarray(embedding))
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        convert_to_numpy: bool = True,  # Accepted for backward compat (always returns numpy)
        normalize_embeddings: bool = True,  # Alias for normalize
    ) -> np.ndarray:
        """
        Generic encode (for backward compatibility with sentence-transformers API).
        
        For best results, use encode_query() for queries and encode_document() for documents.
        
        Args:
            texts: Text or list of texts to encode
            normalize: Whether to L2-normalize (default True)
            convert_to_numpy: Accepted for compatibility (always returns numpy)
            normalize_embeddings: Alias for normalize (for sentence-transformers compat)
        """
        if not self._ensure_loaded():
            dim = self.output_dim
            if isinstance(texts, str):
                return np.zeros(dim, dtype=np.float32)
            return np.zeros((len(texts), dim), dtype=np.float32)
        
        # Use normalize_embeddings if explicitly set to False
        do_normalize = normalize and normalize_embeddings
        
        embedding = self.model.encode(
            texts,
            normalize_embeddings=do_normalize,
            convert_to_numpy=True,  # Always return numpy
        )
        
        return self._truncate_embedding(np.asarray(embedding))
    
    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Shape (dim,) or (n_queries, dim)
            document_embeddings: Shape (n_docs, dim)
            
        Returns:
            Similarity scores of shape (n_docs,) or (n_queries, n_docs)
        """
        # Ensure 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if document_embeddings.ndim == 1:
            document_embeddings = document_embeddings.reshape(1, -1)
        
        # Cosine similarity (assuming normalized)
        return np.dot(query_embedding, document_embeddings.T)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_id": self.model_id,
            "output_dim": self.output_dim,
            "device": self.device,
            "loaded": self._loaded,
            "has_hf_token": bool(self.hf_token),
        }


def get_embedding_model(
    output_dim: int = DEFAULT_EMBEDDING_DIM,
    device: str = "cpu",
    force_reload: bool = False,
) -> Optional[EmbeddingGemmaEncoder]:
    """
    Get the singleton EmbeddingGemma encoder.
    
    Args:
        output_dim: Embedding dimension (768, 512, 256, or 128)
        device: Device to run on
        force_reload: Force reload even if already loaded
        
    Returns:
        EmbeddingGemmaEncoder instance or None on failure
    """
    global _embedding_model, _model_load_attempted
    
    if _embedding_model is not None and not force_reload:
        return _embedding_model
    
    if _model_load_attempted and not force_reload:
        # Already tried and failed
        return None
    
    _model_load_attempted = True
    
    try:
        encoder = EmbeddingGemmaEncoder(
            output_dim=output_dim,
            device=device,
        )
        
        if encoder.load():
            _embedding_model = encoder
            return encoder
        else:
            return None
            
    except Exception as e:
        logger.error(f"Failed to create EmbeddingGemma encoder: {e}")
        return None


def get_encoder():
    """
    Backward-compatible function matching experience_logger.get_encoder() signature.
    
    Falls back to MiniLM if EmbeddingGemma fails.
    """
    # Try EmbeddingGemma first
    encoder = get_embedding_model()
    if encoder is not None:
        return encoder
    
    # Fallback to MiniLM (existing implementation)
    logger.info("EmbeddingGemma unavailable, falling back to MiniLM")
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.warning(f"MiniLM fallback failed: {e}")
        return None

