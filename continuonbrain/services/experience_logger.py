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
    
    def consolidate_memories(self, similarity_threshold: float = 0.90, min_confidence: float = 0.3) -> dict:
        """
        Consolidate similar conversations and remove low-quality entries.
        
        Args:
            similarity_threshold: Merge conversations above this similarity
            min_confidence: Remove conversations below this confidence
            
        Returns:
            Stats about consolidation
        """
        try:
            conversations = self.get_similar_conversations("", max_results=1000) # Changed from get_relevant_conversations
            
            if len(conversations) < 2:
                return {"merged": 0, "removed": 0, "total": len(conversations)}
            
            merged_count = 0
            to_remove = set()
            
            # Find and merge similar conversations
            for i, conv1 in enumerate(conversations):
                if i in to_remove:
                    continue
                    
                for j, conv2 in enumerate(conversations[i+1:], start=i+1):
                    if j in to_remove:
                        continue
                    
                    # Check similarity
                    # Assuming _calculate_similarity is a helper that uses embeddings or keyword
                    # For now, using get_similar_conversations's relevance logic
                    # This part needs an actual _calculate_similarity or direct embedding comparison
                    # For simplicity, let's assume we can get relevance between two questions
                    # This would ideally involve re-encoding or having embeddings stored.
                    # For this exercise, I'll use a placeholder or adapt existing logic.
                    
                    # Re-using the logic from get_similar_conversations for similarity check
                    encoder = get_encoder()
                    if encoder:
                        q1_emb = encoder.encode(conv1['question'], convert_to_numpy=True)
                        q2_emb = encoder.encode(conv2['question'], convert_to_numpy=True)
                        sim = self._cosine_similarity(q1_emb, q2_emb)
                    else:
                        sim = self._keyword_similarity(conv1['question'].lower(), conv2['question'].lower())
                    
                    if sim >= similarity_threshold:
                        # Merge: keep higher confidence, mark other for removal
                        if conv1.get('confidence', 0.5) >= conv2.get('confidence', 0.5):
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break
                        merged_count += 1
            
            # Remove low confidence and duplicates
            filtered = []
            removed_count = 0
            
            for i, conv in enumerate(conversations):
                if i in to_remove:
                    removed_count += 1
                    continue
                
                # Remove low confidence entries
                if conv.get('confidence', 0.5) < min_confidence and not conv.get('validated', False):
                    removed_count += 1
                    continue
                
                filtered.append(conv)
            
            # Write back consolidated conversations
            # Changed from self.storage_path to self.conversations_file
            with open(self.conversations_file, 'w') as f:
                for c in filtered:
                    f.write(json.dumps(c) + '\n')
            
            logger.info(f"Consolidated memories: {merged_count} merged, {removed_count} removed, {len(filtered)} remaining")
            
            return {
                "merged": merged_count,
                "removed": removed_count,
                "total_before": len(conversations),
                "total_after": len(filtered)
            }
            
        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            return {"error": str(e), "merged": 0, "removed": 0}
    
    def apply_confidence_decay(self, decay_factor: float = 0.95, max_age_days: int = 30):
        """
        Apply time-based confidence decay to old conversations.
        
        Args:
            decay_factor: Confidence multiplier per day
            max_age_days: Only decay entries older than this
        """
        try:
            conversations = []
            if self.conversations_file.exists():
                with open(self.conversations_file, 'r') as f:
                    for line in f:
                        try:
                            conversations.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue

            updated_count = 0
            
            now = datetime.now()
            
            for conv in conversations:
                # Skip validated conversations
                if conv.get('validated', False):
                    continue
                
                timestamp_str = conv.get('timestamp', '')
                if not timestamp_str:
                    continue
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    age_days = (now - timestamp).days
                    
                    if age_days > max_age_days:
                        # Apply decay
                        original_confidence = conv.get('confidence', 0.7)
                        days_to_decay = age_days - max_age_days
                        new_confidence = original_confidence * (decay_factor ** days_to_decay)
                        
                        conv['confidence'] = max(0.1, new_confidence)  # Floor at 0.1
                        updated_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp: {e}")
                    continue
            
            # Write back
            if updated_count > 0:
                with open(self.conversations_file, 'w') as f:
                    for c in conversations:
                        f.write(json.dumps(c) + '\n')
                logger.info(f"Applied confidence decay to {updated_count} conversations")
            
            return {"updated": updated_count, "total": len(conversations)}
            
        except Exception as e:
            logger.error(f"Failed to apply confidence decay: {e}")
            return {"error": str(e), "updated": 0}

    
    def search_conversations(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search through conversations with full-text and semantic search.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of matching conversations with relevance scores
        """
        try:
            conversations = []
            if self.conversations_file.exists():
                with open(self.conversations_file, 'r') as f:
                    for line in f:
                        try:
                            conversations.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            
            if not conversations:
                return []
            
            query_lower = query.lower()
            results = []
            
            # Try semantic search first
            encoder = get_encoder()
            if encoder:
                query_embedding = encoder.encode(query, convert_to_numpy=True)
                
                for conv in conversations:
                    # Search in question and answer
                    q_embedding = encoder.encode(conv['question'], convert_to_numpy=True)
                    a_embedding = encoder.encode(conv['answer'], convert_to_numpy=True)
                    
                    q_sim = self._cosine_similarity(query_embedding, q_embedding)
                    a_sim = self._cosine_similarity(query_embedding, a_embedding)
                    
                    relevance = max(q_sim, a_sim)
                    
                    if relevance > 0.3:  # Minimum relevance threshold
                        results.append({
                            **conv,
                            'relevance': float(relevance),
                            'match_type': 'semantic'
                        })
            else:
                # Fallback to keyword search
                for conv in conversations:
                    # Check if query appears in question or answer
                    if (query_lower in conv['question'].lower() or 
                        query_lower in conv['answer'].lower()):
                        # Calculate simple relevance
                        q_words = set(conv['question'].lower().split())
                        a_words = set(conv['answer'].lower().split())
                        query_words = set(query_lower.split())
                        
                        q_overlap = len(q_words & query_words) / max(len(query_words), 1)
                        a_overlap = len(a_words & query_words) / max(len(query_words), 1)
                        
                        relevance = max(q_overlap, a_overlap)
                        
                        results.append({
                            **conv,
                            'relevance': float(relevance),
                            'match_type': 'keyword'
                        })
            
            # Sort by relevance
            results.sort(key=lambda x: x['relevance'], reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []
    
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
