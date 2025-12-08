"""
Experience Logger Service

Logs conversation exchanges as learned experiences that HOPE can recall.
Uses semantic embeddings for intelligent memory retrieval and deduplication.
"""
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Lazy load sentence-transformers to avoid slow startup
_encoder = None

def get_encoder():
    """Lazy load the sentence transformer model."""
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, fast
            logger.info("Loaded sentence-transformers model for semantic search")
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformers: {e}")
            _encoder = False  # Mark as failed, don't retry
    return _encoder if _encoder is not False else None


class ExperienceLogger:
    """Logs and retrieves learned conversation experiences."""
    
    def __init__(self, storage_dir: Path):
        """
        Initialize experience logger.
        
        Args:
            storage_dir: Directory to store conversation experiences
        """
        self.storage_dir = Path(storage_dir)
        self.conversations_file = self.storage_dir / "learned_conversations.jsonl"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Deduplication threshold (0.95 = 95% similar)
        self.dedup_threshold = 0.95
        
    def log_conversation(
        self, 
        question: str, 
        answer: str, 
        agent: str, 
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a conversation exchange as a learned experience.
        
        Args:
            question: User's question
            answer: Agent's answer
            agent: Which agent provided the answer (hope_brain, llm_fallback, etc)
            confidence: Confidence score
            metadata: Optional additional metadata
        """
        try:
            # Check for duplicates before storing
            similar = self.get_similar_conversations(question, max_results=1)
            if similar and similar[0].get('relevance', 0) > self.dedup_threshold:
                logger.info(f"Skipping duplicate conversation: '{question[:50]}...'")
                return
            
            experience = {
                "type": "conversation",
                "question": question,
                "answer": answer,
                "agent": agent,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "validated": False,  # Can be updated via user feedback
                "metadata": metadata or {}
            }
            
            # Append to JSONL file
            with open(self.conversations_file, 'a') as f:
                f.write(json.dumps(experience) + '\n')
                
            logger.info(f"Logged conversation: '{question[:50]}...' -> {agent}")
            
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
    
    def get_similar_conversations(
        self, 
        query: str, 
        max_results: int = 5,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversations similar to a query using semantic search.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of conversation experiences, ranked by relevance
        """
        try:
            if not self.conversations_file.exists():
                return []
            
            encoder = get_encoder()
            conversations = []
            
            # Encode query once
            if encoder:
                query_embedding = encoder.encode(query, convert_to_numpy=True)
            else:
                query_lower = query.lower()
            
            # Read all conversations
            with open(self.conversations_file, 'r') as f:
                for line in f:
                    try:
                        conv = json.loads(line.strip())
                        
                        # Filter by confidence
                        if conv.get("confidence", 0.0) < min_confidence:
                            continue
                        
                        question = conv.get("question", "")
                        
                        # Calculate relevance using embeddings or fallback to keywords
                        if encoder:
                            question_embedding = encoder.encode(question, convert_to_numpy=True)
                            relevance = self._cosine_similarity(query_embedding, question_embedding)
                        else:
                            # Fallback to keyword matching
                            relevance = self._keyword_similarity(query_lower, question.lower())
                        
                        if relevance > 0:
                            conv["relevance"] = float(relevance)
                            conversations.append(conv)
                            
                    except json.JSONDecodeError:
                        continue
            
            # Sort by relevance (descending) and return top results
            conversations.sort(key=lambda x: x.get("relevance", 0), reverse=True)
            return conversations[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversations: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product > 0 else 0.0
    
    def _keyword_similarity(self, query: str, question: str) -> float:
        """
        Fallback keyword similarity (Jaccard).
        
        Returns:
            Similarity score 0.0-1.0
        """
        query_words = set(query.split())
        question_words = set(question.split())
        
        if not query_words or not question_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words & question_words
        union = query_words | question_words
        
        return len(intersection) / len(union) if union else 0.0
    
    
    def validate_conversation(
        self, 
        timestamp: str, 
        validated: bool, 
        correction: Optional[str] = None
    ) -> bool:
        """
        Update validation status of a conversation.
        
        Args:
            timestamp: ISO timestamp of the conversation to validate
            validated: True if correct, False if incorrect
            correction: Optional correction text if validated=False
            
        Returns:
            True if conversation was found and updated
        """
        try:
            if not self.conversations_file.exists():
                return False
            
            temp_file = self.conversations_file.with_suffix('.tmp')
            updated = False
            
            with open(self.conversations_file, 'r') as f_in, open(temp_file, 'w') as f_out:
                for line in f_in:
                    try:
                        conv = json.loads(line.strip())
                        
                        # Update if timestamp matches
                        if conv.get('timestamp') == timestamp:
                            conv['validated'] = validated
                            if correction:
                                conv['correction'] = correction
                            updated = True
                            logger.info(f"Validated conversation: {conv.get('question', '')[:50]}... -> {validated}")
                        
                        f_out.write(json.dumps(conv) + '\n')
                        
                    except json.JSONDecodeError:
                        f_out.write(line)  # Keep malformed lines as-is
            
            # Replace original with updated file
            if updated:
                temp_file.replace(self.conversations_file)
            else:
                temp_file.unlink()  # Remove temp file if no updates
            
            return updated
            
        except Exception as e:
            logger.error(f"Failed to validate conversation: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        try:
            if not self.conversations_file.exists():
                return {"total_conversations": 0}
            
            total = 0
            by_agent = {}
            
            with open(self.conversations_file, 'r') as f:
                for line in f:
                    try:
                        conv = json.loads(line.strip())
                        total += 1
                        agent = conv.get("agent", "unknown")
                        by_agent[agent] = by_agent.get(agent, 0) + 1
                    except json.JSONDecodeError:
                        continue
            
            return {
                "total_conversations": total,
                "by_agent": by_agent
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"total_conversations": 0}
