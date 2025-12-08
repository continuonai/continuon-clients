"""
HOPE Agent Service

Wraps the HOPE brain to provide agent-like capabilities:
- Confidence/novelty assessment for queries
- Memory-based response generation
- Natural language interface to HOPE's learned knowledge
"""
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class HOPEAgent:
    """Agent interface for HOPE brain to respond from learned knowledge."""
    
    def __init__(self, hope_brain, confidence_threshold: float = 0.6):
        """
        Initialize HOPE agent.
        
        Args:
            hope_brain: The HOPE brain instance
            confidence_threshold: Minimum confidence to use HOPE response (0.0-1.0)
        """
        self.brain = hope_brain
        self.confidence_threshold = confidence_threshold
        
    def can_answer(self, message: str) -> Tuple[bool, float]:
        """
        Determine if HOPE can confidently answer a query.
        
        Uses HOPE's stability metrics and training state to estimate confidence.
        
        Args:
            message: User's question/message
            
        Returns:
            (can_answer, confidence_score) tuple
        """
        try:
            # Check if HOPE is initialized
            if not self.brain or not hasattr(self.brain, 'columns'):
                return (False, 0.0)
            
            # Get active column's stability metrics
            try:
                col = self.brain.columns[self.brain.active_column_idx]
                
                # If no state yet (untrained), very low confidence
                if col._state is None:
                    return (False, 0.1)
                
                # Use stability metrics as confidence proxy
                # Lower Lyapunov energy = higher confidence (better prediction match)
                metrics = col.stability_monitor.get_metrics()
                lyapunov = metrics.get('lyapunov', float('inf'))
                
                # Normalize Lyapunov to 0-1 confidence score
                # Lower is better, so invert: confidence = 1 / (1 + lyapunov)
                # Cap at reasonable values
                if lyapunov == float('inf') or lyapunov > 100:
                    confidence = 0.1
                else:
                    confidence = 1.0 / (1.0 + lyapunov)
                
                # Boost confidence for capability queries (HOPE knows itself)
                capability_keywords = ['see', 'camera', 'move', 'arm', 'sensor', 'can you', 'what do you']
                if any(kw in message.lower() for kw in capability_keywords):
                    confidence = min(0.9, confidence + 0.3)  # Boost but cap at 0.9
                
                # Check stability - unstable brain = lower confidence
                if metrics.get('is_stable', True) == False:
                    confidence *= 0.5  # Halve confidence if unstable
                
                return (confidence >= self.confidence_threshold, confidence)
                
            except Exception as e:
                logger.warning(f"Failed to get HOPE metrics, using fallback: {e}")
                # Fallback to simple heuristic
                capability_keywords = ['see', 'camera', 'move', 'arm', 'sensor', 'can you', 'what do you']
                if any(kw in message.lower() for kw in capability_keywords):
                    return (True, 0.7)
                return (False, 0.3)
            
        except Exception as e:
            logger.error(f"Error assessing HOPE confidence: {e}")
            return (False, 0.0)
    
    def generate_response(self, message: str) -> str:
        """
        Generate response from HOPE's learned knowledge.
        
        Args:
            message: User's question/message
            
        Returns:
            Natural language response based on HOPE's knowledge
        """
        try:
            # Simple capability responses for now
            # Future: integrate with HOPE's actual state and memories
            
            message_lower = message.lower()
            
            if 'see' in message_lower or 'camera' in message_lower:
                return self._describe_vision_capability()
            elif 'move' in message_lower or 'drive' in message_lower:
                return self._describe_motion_capability()
            elif 'arm' in message_lower:
                return self._describe_arm_capability()
            elif 'can you' in message_lower or 'what do you' in message_lower:
                return self._describe_general_capabilities()
            else:
                # Fallback for queries we can't handle yet
                return None
                
        except Exception as e:
            logger.error(f"Error generating HOPE response: {e}")
            return None
    
    def get_relevant_memories(self, query: str, max_memories: int = 5, experience_logger=None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories from HOPE's CMS and learned conversations.
        
        Args:
            query: Search query
            max_memories: Maximum number of memories to return
            experience_logger: Optional ExperienceLogger for learned conversations
            
        Returns:
            List of memory dictionaries with keys: description, confidence, timestamp
        """
        try:
            memories = []
            
            # 1. Get system state memories (existing)
            if hasattr(self.brain, 'columns'):
                memories.append({
                    "description": f"I have {len(self.brain.columns)} HOPE columns for processing",
                    "confidence": 0.9,
                    "timestamp": "system",
                    "source": "hope_state"
                })
            
            # 2. Get learned conversations (NEW)
            if experience_logger:
                learned = experience_logger.get_similar_conversations(query, max_results=max_memories)
                for conv in learned:
                    memories.append({
                        "description": f"Previously learned: Q: '{conv['question']}' A: '{conv['answer']}'",
                        "confidence": conv.get("confidence", 0.5),
                        "timestamp": conv.get("timestamp", "unknown"),
                        "source": "learned_conversation"
                    })
            
            # Sort by confidence and return top results
            memories.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            return memories[:max_memories]
            
        except Exception as e:
            logger.error(f"Error retrieving HOPE memories: {e}")
            return []
    
    def _describe_vision_capability(self) -> str:
        """Describe vision capabilities from HOPE's perspective."""
        return ("I can see through my OAK-D camera. I process depth and RGB images "
                "to understand my environment. My vision system helps me navigate "
                "and identify objects.")
    
    def _describe_motion_capability(self) -> str:
        """Describe motion capabilities."""
        return ("I can move using my drivetrain with steering and throttle control. "
                "I'm configured with PCA9685 PWM channels for precise motor control. "
                "Motion is currently supervised for safety.")
    
    def _describe_arm_capability(self) -> str:
        """Describe arm capabilities."""
        return ("I have a robotic arm controlled via PCA9685 servos. "
                "The arm can reach, grasp, and manipulate objects in my workspace. "
                "I'm learning manipulation skills through practice.")
    
    def _describe_general_capabilities(self) -> str:
        """Describe general capabilities."""
        return ("I'm a learning robot with vision (OAK-D camera), mobility (drivetrain), "
                "and manipulation (robotic arm). I use the HOPE brain architecture to learn "
                "from experience and improve over time. I'm currently operating in autonomous "
                "mode with safety protocols active.")
