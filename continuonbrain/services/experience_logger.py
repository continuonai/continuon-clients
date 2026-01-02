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
from typing import Dict, Any, List, Optional, Callable
from continuonbrain.core.feedback_store import SQLiteFeedbackStore

logger = logging.getLogger(__name__)

# Lazy load embedding model to avoid slow startup
_encoder = None
_encoder_name = None

def get_encoder():
    """
    Lazy load the embedding model for semantic search.
    
    Priority:
    1. EmbeddingGemma-300m (state-of-the-art, 768d)
    2. all-MiniLM-L6-v2 fallback (fast, 384d)
    """
    global _encoder, _encoder_name
    if _encoder is None:
        # Try EmbeddingGemma first (best quality)
        try:
            from continuonbrain.services.embedding_gemma import get_embedding_model
            _encoder = get_embedding_model()
            if _encoder is not None:
                _encoder_name = "google/embeddinggemma-300m"
                logger.info("✅ Loaded EmbeddingGemma-300m for semantic search (768d)")
                return _encoder
        except Exception as e:
            logger.debug(f"EmbeddingGemma unavailable: {e}")
        
        # Fallback to MiniLM
        try:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, fast
            _encoder_name = "all-MiniLM-L6-v2"
            logger.info("Loaded MiniLM for semantic search (fallback)")
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformers: {e}")
            _encoder = False  # Mark as failed, don't retry
    return _encoder if _encoder is not False else None


def get_encoder_name() -> str:
    """Get the name of the loaded encoder model."""
    global _encoder_name
    get_encoder()  # Ensure encoder is loaded
    return _encoder_name or "none"


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
        
        # Initialize feedback store
        self.feedback_store = SQLiteFeedbackStore(str(self.storage_dir / "feedback.db"))
        self.feedback_store.initialize_db()
        
        # Deduplication threshold (0.95 = 95% similar)
        self.dedup_threshold = 0.95

        # Define category centroids for automatic tagging
        self.category_hints = {
            "Safety": "safety protocols, rules, emergency stop, collision, halt, protocol 66",
            "Motion": "move, drive, joint, arm, forward, backward, steering, throttle",
            "Vision": "camera, see, depth, image, rgb, vision, look",
            "Identity": "who are you, name, creator, owner, system name",
            "Knowledge": "what is, explain, facts, wikipedia, calculate"
        }
        self._centroids = None

    def _get_centroids(self):
        if self._centroids is None:
            encoder = get_encoder()
            if encoder:
                self._centroids = {
                    cat: encoder.encode(hint, convert_to_numpy=True)
                    for cat, hint in self.category_hints.items()
                }
        return self._centroids

    def _assign_topic_tags(self, embedding: np.ndarray) -> List[str]:
        centroids = self._get_centroids()
        if not centroids:
            return []
        
        tags = []
        for cat, centroid in centroids.items():
            sim = self._cosine_similarity(embedding, centroid)
            if sim > 0.4: # Topic threshold
                tags.append(cat)
        return tags
        
    def log_conversation(
        self, 
        question: str, 
        answer: str, 
        agent: str, 
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a conversation exchange as a learned experience.
        
        Args:
            question: User's question
            answer: Agent's answer
            agent: Which agent provided the answer (hope_brain, llm_fallback, etc)
            confidence: Confidence score
            metadata: Optional additional metadata
            
        Returns:
            conversation_id: Unique ID of the logged experience
        """
        try:
            # Check for duplicates before storing (Active Gating)
            similar = self.get_similar_conversations(question, max_results=1)
            if similar and similar[0].get('relevance', 0) > self.dedup_threshold:
                # Update existing memory instead of creating new
                conversation_id = similar[0].get('conversation_id', similar[0].get('timestamp'))
                self._increment_hit_count(conversation_id)
                logger.info(f"Incremented hit_count for existing memory: {conversation_id}")
                return conversation_id
            
            encoder = get_encoder()
            embedding = None
            tags = []
            if encoder:
                emb_np = encoder.encode(question, convert_to_numpy=True)
                embedding = emb_np.tolist()
                tags = self._assign_topic_tags(emb_np)

            # Enrich metadata
            final_metadata = metadata or {}
            if "novelty" not in final_metadata:
                # Infer novelty from confidence if missing
                final_metadata["novelty"] = 1.0 - confidence

            timestamp = datetime.now().isoformat()
            conversation_id = f"conv_{int(datetime.now().timestamp())}_{np.random.randint(1000, 9999)}"

            experience = {
                "type": "conversation",
                "conversation_id": conversation_id,
                "question": question,
                "answer": answer,
                "agent": agent,
                "confidence": confidence,
                "timestamp": timestamp,
                "last_accessed": timestamp,
                "hit_count": 1,
                "validated": False,
                "embedding": embedding,
                "tags": tags,
                "metadata": final_metadata
            }
            
            # Append to JSONL file
            with open(self.conversations_file, 'a') as f:
                f.write(json.dumps(experience) + '\n')
                
            logger.info(f"Logged conversation: {conversation_id} ('{question[:50]}...') -> {agent}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
            return ""

    def _increment_hit_count(self, timestamp: str) -> bool:
        """Update hit_count and last_accessed for a conversation."""
        try:
            if not self.conversations_file.exists():
                return False
            
            temp_file = self.conversations_file.with_suffix('.tmp')
            updated = False
            
            with open(self.conversations_file, 'r') as f_in, open(temp_file, 'w') as f_out:
                for line in f_in:
                    try:
                        conv = json.loads(line.strip())
                        if conv.get('timestamp') == timestamp:
                            conv['hit_count'] = conv.get('hit_count', 1) + 1
                            conv['last_accessed'] = datetime.now().isoformat()
                            updated = True
                        f_out.write(json.dumps(conv) + '\n')
                    except json.JSONDecodeError:
                        f_out.write(line)
            
            if updated:
                temp_file.replace(self.conversations_file)
            else:
                temp_file.unlink()
            return updated
        except Exception as e:
            logger.error(f"Failed to increment hit_count: {e}")
            return False
    
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
                            # Use cached embedding if available
                            if "embedding" in conv:
                                question_embedding = np.array(conv["embedding"])
                            else:
                                # One-time migration: encode and store back (or just encode for now)
                                question_embedding = encoder.encode(question, convert_to_numpy=True)
                            
                            relevance = self._cosine_similarity(query_embedding, question_embedding)
                        else:
                            # Fallback to keyword matching
                            relevance = self._keyword_similarity(query_lower, question.lower())
                        
                        if relevance > 0:
                            # 2.5 Priority Boosting (+0.2 for validated memories)
                            # Check feedback store for validation status
                            fb = self.feedback_store.get_feedback(conv.get("conversation_id", conv.get("timestamp")))
                            is_validated = fb["is_validated"] if fb else conv.get("validated", False)
                            
                            conv["validated"] = is_validated
                            if is_validated:
                                relevance += 0.2
                                relevance = min(1.0, relevance)
                            
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
        conversation_id: str, 
        validated: bool, 
        correction: Optional[str] = None
    ) -> bool:
        """
        Update validation status of a conversation in both JSONL and feedback store.
        
        Args:
            conversation_id: Unique ID of the conversation to validate
            validated: True if correct, False if incorrect
            correction: Optional correction text if validated=False
            
        Returns:
            True if conversation was found and updated
        """
        try:
            # 1. Update Feedback Store (SQLite)
            self.feedback_store.add_feedback(conversation_id, validated, correction)

            # 2. Update primary JSONL log
            if not self.conversations_file.exists():
                return False
            
            temp_file = self.conversations_file.with_suffix('.tmp')
            updated = False
            
            with open(self.conversations_file, 'r') as f_in, open(temp_file, 'w') as f_out:
                for line in f_in:
                    try:
                        conv = json.loads(line.strip())
                        
                        # Update if ID matches (or timestamp for back-compat)
                        if conv.get('conversation_id') == conversation_id or conv.get('timestamp') == conversation_id:
                            conv['validated'] = validated
                            if correction:
                                conv['correction'] = correction
                            updated = True
                            logger.info(f"Validated conversation in JSONL: {conversation_id} -> {validated}")
                        
                        f_out.write(json.dumps(conv) + '\n')
                        
                    except json.JSONDecodeError:
                        f_out.write(line)
            
            if updated:
                temp_file.replace(self.conversations_file)
            else:
                temp_file.unlink()
            
            return updated
            
        except Exception as e:
            logger.error(f"Failed to validate conversation: {e}")
            return False
    
    def consolidate_memories(self, similarity_threshold: float = 0.90, min_confidence: float = 0.3, synthesizer_cb: Optional[Callable] = None) -> dict:
        """
        Consolidate similar conversations and remove low-quality entries.
        
        Args:
            similarity_threshold: Merge conversations above this similarity
            min_confidence: Remove conversations below this confidence
            synthesizer_cb: Callback to synthesize a single answer from a cluster
            
        Returns:
            Stats about consolidation
        """
        try:
            clusters = self.cluster_similar_memories(threshold=similarity_threshold)
            
            new_memories = []
            merged_count = 0
            removed_count = 0
            
            for cluster in clusters:
                if len(cluster) > 1:
                    # Perform synthesis if callback provided
                    if synthesizer_cb:
                        anchors = synthesizer_cb([cluster])
                        if anchors:
                            new_memories.extend(anchors)
                            merged_count += len(cluster)
                        else:
                            # Fallback: keep highest confidence
                            best = max(cluster, key=lambda x: x.get("confidence", 0.5))
                            new_memories.append(best)
                            merged_count += len(cluster) - 1
                    else:
                        # Simple merge: keep highest confidence
                        best = max(cluster, key=lambda x: x.get("confidence", 0.5))
                        new_memories.append(best)
                        merged_count += len(cluster) - 1
                else:
                    # Single item cluster
                    conv = cluster[0]
                    # Filter by confidence
                    fb = self.feedback_store.get_feedback(conv.get("conversation_id", conv.get("timestamp")))
                    is_validated = fb["is_validated"] if fb else conv.get("validated", False)
                    
                    if is_validated or conv.get("confidence", 0.5) >= min_confidence:
                        new_memories.append(conv)
                    else:
                        removed_count += 1
            
            # Write back to file
            with open(self.conversations_file, 'w') as f:
                for m in new_memories:
                    # Re-log ensuring ID and embedding exist
                    if "conversation_id" not in m:
                        m["conversation_id"] = f"conv_{int(datetime.now().timestamp())}_{np.random.randint(1000, 9999)}"
                    f.write(json.dumps(m) + '\n')
            
            logger.info(f"Consolidation complete: {merged_count} merged, {removed_count} removed, {len(new_memories)} remaining")
            
            return {
                "merged": merged_count,
                "removed": removed_count,
                "total_after": len(new_memories),
                "status": "complete"
            }
            
        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            return {"error": str(e), "merged": 0, "removed": 0}

    def cluster_similar_memories(self, threshold: float = 0.90) -> List[List[Dict[str, Any]]]:
        """
        Group memories into clusters based on semantic similarity.
        
        Returns:
            List of clusters, where each cluster is a list of conversation dicts.
        """
        try:
            if not self.conversations_file.exists():
                return []
            
            all_convs = []
            with open(self.conversations_file, 'r') as f:
                for line in f:
                    try:
                        all_convs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            if not all_convs:
                return []

            # 1. Ensure all have embeddings
            encoder = get_encoder()
            for conv in all_convs:
                if "embedding" not in conv or conv["embedding"] is None:
                    if encoder:
                        conv["embedding"] = encoder.encode(conv["question"], convert_to_numpy=True).tolist()
                    else:
                        # Fallback: cannot cluster without embeddings
                        return [[c] for c in all_convs]

            # 2. Greedy Clustering
            clusters = []
            assigned = set()

            for i, conv1 in enumerate(all_convs):
                if i in assigned:
                    continue
                
                current_cluster = [conv1]
                assigned.add(i)
                vec1 = np.array(conv1["embedding"])

                for j, conv2 in enumerate(all_convs[i+1:], start=i+1):
                    if j in assigned:
                        continue
                    
                    vec2 = np.array(conv2["embedding"])
                    sim = self._cosine_similarity(vec1, vec2)
                    
                    if sim >= threshold:
                        current_cluster.append(conv2)
                        assigned.add(j)
                
                clusters.append(current_cluster)
            
            return clusters

        except Exception as e:
            logger.error(f"Failed to cluster memories: {e}")
            return []
    
    def apply_confidence_decay(self, decay_factor: float = 0.95, max_age_days: int = 30):
        """
        Apply time-based confidence decay to old conversations based on last_accessed.
        
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
                # 1. Status-based Immunity: Gold Data (validated) is immune
                fb = self.feedback_store.get_feedback(conv.get("conversation_id", conv.get("timestamp")))
                is_validated = fb["is_validated"] if fb else conv.get("validated", False)
                if is_validated:
                    continue
                
                # 2. Use last_accessed for recency decay
                access_str = conv.get('last_accessed', conv.get('timestamp', ''))
                if not access_str:
                    continue
                
                try:
                    last_access = datetime.fromisoformat(access_str)
                    age_days = (now - last_access).days
                    
                    if age_days > max_age_days:
                        # Apply decay
                        original_confidence = conv.get('confidence', 0.7)
                        days_to_decay = age_days - max_age_days
                        new_confidence = original_confidence * (decay_factor ** days_to_decay)
                        
                        conv['confidence'] = max(0.1, round(new_confidence, 4))  # Floor at 0.1
                        updated_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to parse access timestamp: {e}")
                    continue
            
            # Write back
            if updated_count > 0:
                with open(self.conversations_file, 'w') as f:
                    for c in conversations:
                        f.write(json.dumps(c) + '\n')
                logger.info(f"Applied recency decay to {updated_count} conversations")
            
            return {"updated": updated_count, "total": len(conversations)}
            
        except Exception as e:
            logger.error(f"Failed to apply confidence decay: {e}")
            return {"error": str(e), "updated": 0}

    def apply_model_evolution_penalty(self, penalty: float = 0.10):
        """
        Apply a one-time penalty to all unvalidated memories (e.g. after model update).
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
            for conv in conversations:
                # Immunity check
                fb = self.feedback_store.get_feedback(conv.get("conversation_id", conv.get("timestamp")))
                is_validated = fb["is_validated"] if fb else conv.get("validated", False)
                if is_validated:
                    continue
                
                original_confidence = conv.get('confidence', 0.7)
                conv['confidence'] = max(0.1, round(original_confidence * (1.0 - penalty), 4))
                updated_count += 1
            
            if updated_count > 0:
                with open(self.conversations_file, 'w') as f:
                    for c in conversations:
                        f.write(json.dumps(c) + '\n')
                logger.info(f"Applied {penalty*100}% model evolution penalty to {updated_count} memories")
                
            return {"updated": updated_count, "total": len(conversations)}
        except Exception as e:
            logger.error(f"Failed to apply model evolution penalty: {e}")
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
                        # Check validation status
                        fb = self.feedback_store.get_feedback(conv.get("conversation_id", conv.get("timestamp")))
                        is_validated = fb["is_validated"] if fb else conv.get("validated", False)
                        
                        results.append({
                            **conv,
                            'relevance': float(relevance),
                            'validated': is_validated,
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
                        
                        # Check validation status
                        fb = self.feedback_store.get_feedback(conv.get("conversation_id", conv.get("timestamp")))
                        is_validated = fb["is_validated"] if fb else conv.get("validated", False)

                        results.append({
                            **conv,
                            'relevance': float(relevance),
                            'validated': is_validated,
                            'match_type': 'keyword'
                        })
            
            # Sort by relevance
            results.sort(key=lambda x: x['relevance'], reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics including user validation data."""
        try:
            # 1. Base counts from JSONL
            total = 0
            by_agent = {}
            if self.conversations_file.exists():
                with open(self.conversations_file, 'r') as f:
                    for line in f:
                        try:
                            conv = json.loads(line.strip())
                            total += 1
                            agent = conv.get("agent", "unknown")
                            by_agent[agent] = by_agent.get(agent, 0) + 1
                        except json.JSONDecodeError:
                            continue
            
            # 2. Validation summary from SQLite
            fb_summary = self.feedback_store.get_summary()
            
            return {
                "total_conversations": total,
                "feedback_stats": fb_summary,
                "by_agent": by_agent
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"total_conversations": 0}
