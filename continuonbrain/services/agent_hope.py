"""
HOPE Agent Service

Wraps the HOPE brain to provide agent-like capabilities:
- Confidence/novelty assessment for queries
- Memory-based response generation (semantic search)
- World model integration for physics prediction
- Natural language interface to HOPE's learned knowledge

This is the primary interface for human-robot interaction via text/voice/video.
"""
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class HOPEAgent:
    """
    Agent interface for HOPE brain - the primary human-robot interaction layer.
    
    Integrates:
    - HOPE Brain: Neural state tracking and prediction
    - World Model: Physics simulation and action planning
    - Semantic Search: Memory retrieval and knowledge lookup
    - Vision: SAM3 segmentation and object recognition
    """
    
    def __init__(
        self, 
        hope_brain, 
        confidence_threshold: float = 0.6,
        world_model=None,
        semantic_search=None,
        vision_service=None,
    ):
        """
        Initialize HOPE agent with integrated cognitive systems.
        
        Args:
            hope_brain: The HOPE brain instance (state tracking, prediction)
            confidence_threshold: Minimum confidence to use HOPE response (0.0-1.0)
            world_model: Optional world model for physics prediction (Mamba/JAX)
            semantic_search: Optional semantic search for memory retrieval
            vision_service: Optional vision service (SAM3, Hailo)
        """
        self.brain = hope_brain
        self.confidence_threshold = confidence_threshold
        
        # Integrated cognitive systems
        self.world_model = world_model
        self.semantic_search = semantic_search
        self.vision_service = vision_service
        
        # State tracking
        self._last_prediction = None
        self._last_plan = None
        
    def can_answer(self, message: str, context_subgraph: Optional[Dict[str, Any]] = None) -> Tuple[bool, float]:
        """
        Determine if HOPE can confidently answer a query.
        
        Uses prediction error (novelty) from the HOPE brain to determine confidence.
        
        Args:
            message: User's question/message
            
        Returns:
            (can_answer, confidence_score) tuple
        """
        try:
            # Check if HOPE is initialized
            if not self.brain or not hasattr(self.brain, 'columns'):
                return (False, 0.0)
            
            # Use current novelty/confidence if available from last step
            # Note: For a query, we should ideally run a "mental simulation" (forward pass)
            # but for now we look at the last observed confidence of the column.
            try:
                col = self.brain.columns[self.brain.active_column_idx]
                
                # If no state yet (untrained), very low confidence
                if col._state is None:
                    return (False, 0.1)
                
                # Retrieve last confidence from stability monitor or step info
                # Assuming BrainService or Column tracks this.
                # In this track, we ensure HOPEBrain info contains 'confidence'
                
                # Heuristic: if we don't have a fresh step yet, use 0.5
                confidence = getattr(col, 'last_confidence', 0.5)
                
                # Check for stability - unstable brain = lower confidence
                metrics = col.stability_monitor.get_metrics()
                if metrics.get('is_stable', True) == False:
                    confidence *= 0.5
                
                if context_subgraph:
                    intent_boost = len([n for n in context_subgraph.get("nodes", []) if getattr(n, "type", "") == "intent"])
                    confidence += min(0.1, 0.02 * intent_boost)
                return (confidence >= self.confidence_threshold, confidence)
                
            except Exception as e:
                logger.warning(f"Failed to get dynamic confidence, using fallback: {e}")
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
                return (
                    "I'm running the HOPE seed model and still learning. "
                    "I can describe my sensors, movement, and safety status. "
                    "Ask me what I can see, how I move, or how to stay safe."
                )
                
        except Exception as e:
            logger.error(f"Error generating HOPE response: {e}")
            return None
    
    def get_relevant_memories(
        self,
        query: str,
        max_memories: int = 5,
        experience_logger=None,
        context_subgraph: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
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
            
            # Add symbolic context from the particle graph if available
            if context_subgraph:
                for node in context_subgraph.get("nodes", []):
                    memories.append(
                        {
                            "description": f"Context node ({node.type}): {node.name}",
                            "confidence": node.belief.get("score", 0.5) if hasattr(node, "belief") else 0.5,
                            "timestamp": getattr(node, "attributes", {}).get("timestamp", "context"),
                            "source": "context_graph",
                        }
                    )

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

    # =========================================================================
    # Vision Integration - SAM3 Segmentation and Object Recognition
    # =========================================================================

    def get_visual_perception(self, segmentation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get current visual perception from SAM segmentation.

        This provides the HOPE agent with grounded visual understanding
        of the environment for reasoning and planning.

        Args:
            segmentation_data: Optional pre-fetched segmentation data.
                               If None, will try to fetch from vision_service.

        Returns:
            Dict with visual perception including detected objects,
            their positions, and scene description.
        """
        perception = {
            "has_vision": False,
            "num_objects": 0,
            "objects": [],
            "scene_description": "No visual data available",
            "timestamp": None,
        }

        try:
            # Use provided data or fetch from vision service
            data = segmentation_data

            if data is None and self.vision_service:
                # Try to get latest segmentation from vision service
                if hasattr(self.vision_service, 'last_segmentation'):
                    data = self.vision_service.last_segmentation

            if data and data.get("num_objects", 0) > 0:
                perception["has_vision"] = True
                perception["num_objects"] = data["num_objects"]
                perception["timestamp"] = data.get("timestamp")
                perception["frame_size"] = data.get("frame_size", [640, 400])

                # Process detected objects into semantic descriptions
                objects = []
                for obj in data.get("objects", []):
                    obj_desc = {
                        "id": obj["id"],
                        "score": obj["score"],
                        "position": {
                            "center": obj["center"],
                            "box": obj["box"],
                        },
                        "size": obj.get("area", 0),
                        "description": self._describe_object(obj),
                    }
                    objects.append(obj_desc)

                perception["objects"] = objects
                perception["scene_description"] = self._generate_scene_description(objects)

        except Exception as e:
            logger.error(f"Error getting visual perception: {e}")

        return perception

    def _describe_object(self, obj: Dict[str, Any]) -> str:
        """Generate a natural language description of a detected object."""
        center = obj.get("center", [0, 0])
        score = obj.get("score", 0)
        area = obj.get("area", 0)

        # Determine position in frame
        frame_w, frame_h = 640, 400  # Default frame size
        x_pos = "center" if 0.3 < center[0]/frame_w < 0.7 else ("left" if center[0]/frame_w < 0.3 else "right")
        y_pos = "middle" if 0.3 < center[1]/frame_h < 0.7 else ("top" if center[1]/frame_h < 0.3 else "bottom")

        # Estimate size category
        size_cat = "small" if area < 5000 else ("medium" if area < 20000 else "large")

        return f"{size_cat} object at {x_pos}-{y_pos} (confidence: {score:.0%})"

    def _generate_scene_description(self, objects: List[Dict[str, Any]]) -> str:
        """Generate a natural language description of the scene."""
        if not objects:
            return "The scene appears empty or no objects were detected."

        n = len(objects)
        if n == 1:
            return f"I can see 1 object: {objects[0]['description']}"

        desc_list = [obj['description'] for obj in objects[:3]]  # Top 3
        remaining = n - 3 if n > 3 else 0

        scene = f"I can see {n} objects: " + ", ".join(desc_list)
        if remaining > 0:
            scene += f", and {remaining} more"

        return scene

    def inject_visual_context(self, segmentation_data: Dict[str, Any]) -> None:
        """
        Inject visual perception into the HOPE brain's context.

        This updates the brain's internal state with current visual observations,
        allowing it to reason about what it sees.

        Args:
            segmentation_data: Segmentation data from SAM vision service
        """
        perception = self.get_visual_perception(segmentation_data)

        if not perception["has_vision"]:
            return

        # Create visual context for HOPE brain
        visual_context = {
            "type": "visual_observation",
            "timestamp": perception["timestamp"],
            "scene": perception["scene_description"],
            "objects": perception["objects"],
        }

        # Inject into brain if it supports context injection
        if hasattr(self.brain, 'inject_context'):
            self.brain.inject_context(visual_context)

        logger.debug(f"Injected visual context: {perception['num_objects']} objects")

    # =========================================================================
    # World Model Integration - Physics Prediction and Planning
    # =========================================================================
    
    def predict_action_outcome(
        self, 
        current_state: Dict[str, Any], 
        proposed_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict the outcome of an action using the world model.
        
        This allows HOPE to "think ahead" before taking actions.
        
        Args:
            current_state: Current robot state (joint positions, etc.)
            proposed_action: Proposed action (joint deltas, etc.)
            
        Returns:
            Prediction result with next_state, uncertainty, and safety info
        """
        if not self.world_model:
            return {
                "success": False,
                "error": "World model not available",
                "can_predict": False
            }
        
        try:
            from continuonbrain.mamba_brain import WorldModelState, WorldModelAction
            
            # Convert to world model format
            joint_pos = current_state.get("joint_positions", [0.5] * 6)
            joint_delta = proposed_action.get("joint_delta", [0.0] * 6)
            
            state = WorldModelState(joint_pos=joint_pos)
            action = WorldModelAction(joint_delta=joint_delta)
            
            # Predict outcome
            result = self.world_model.predict(state, action)
            
            self._last_prediction = result
            
            return {
                "success": True,
                "next_state": result.next_state.joint_pos,
                "uncertainty": result.uncertainty,
                "is_safe": result.uncertainty < 0.5,  # Low uncertainty = safe
                "backend": result.debug.get("backend", "unknown"),
            }
            
        except Exception as e:
            logger.error(f"World model prediction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "can_predict": False
            }
    
    def plan_to_goal(
        self,
        current_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        time_budget_ms: int = 150,
    ) -> Dict[str, Any]:
        """
        Plan a sequence of actions to reach a goal state.
        
        Uses beam search with the world model to find safe action sequences.
        
        Args:
            current_state: Current robot state
            goal_state: Target state to reach
            time_budget_ms: Maximum planning time
            
        Returns:
            Plan with action sequence and confidence
        """
        if not self.world_model:
            return {
                "success": False,
                "error": "World model not available",
                "actions": []
            }
        
        try:
            from continuonbrain.reasoning.tree_search import beam_search_plan, ArmGoal
            from continuonbrain.mamba_brain import WorldModelState
            
            joint_pos = current_state.get("joint_positions", [0.5] * 6)
            target_pos = goal_state.get("target_positions", [0.5] * 6)
            
            start = WorldModelState(joint_pos=joint_pos)
            goal = ArmGoal(target_joint_pos=target_pos)
            
            # Run beam search
            actions, info = beam_search_plan(
                world_model=self.world_model,
                start_state=start,
                goal=goal,
                time_budget_ms=time_budget_ms,
            )
            
            self._last_plan = {
                "actions": [a.joint_delta for a in actions],
                "info": info
            }
            
            return {
                "success": True,
                "actions": [a.joint_delta for a in actions],
                "steps": len(actions),
                "total_uncertainty": info.get("total_uncertainty", 0.0),
                "planning_time_ms": info.get("planning_time_ms", 0),
            }
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "actions": []
            }
    
    def describe_world_understanding(self) -> str:
        """
        Describe what the robot understands about the world from its model.
        
        Returns natural language description of world model state.
        """
        description = []
        
        # World model status
        if self.world_model:
            backend = getattr(self.world_model, '_stub', None)
            if backend is None:
                description.append("I have a trained world model for physics prediction.")
            else:
                description.append("I'm using a basic physics model (learning in progress).")
            
            if self._last_prediction:
                uncertainty = self._last_prediction.uncertainty
                if uncertainty < 0.3:
                    description.append("My last prediction was very confident.")
                elif uncertainty < 0.6:
                    description.append("My last prediction had moderate confidence.")
                else:
                    description.append("My predictions are still uncertain - I need more experience.")
        else:
            description.append("I don't have a world model loaded yet.")
        
        # Semantic memory status
        if self.semantic_search:
            description.append("I can search my memories for relevant experiences.")
        
        # Vision status
        if self.vision_service:
            description.append("I can see and segment objects in my environment.")
        
        return " ".join(description)
    
    # =========================================================================
    # Semantic Search Integration - Memory and Knowledge
    # =========================================================================
    
    def search_memories(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search semantic memory for relevant past experiences.
        
        Args:
            query: Natural language query
            max_results: Maximum memories to return
            
        Returns:
            List of relevant memories with scores
        """
        if not self.semantic_search:
            return []
        
        try:
            return self.semantic_search.get_similar_conversations(
                query, 
                max_results=max_results
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def answer_from_memory(self, question: str) -> Optional[str]:
        """
        Try to answer a question from learned memories.
        
        Args:
            question: User's question
            
        Returns:
            Answer from memory if found with high confidence, else None
        """
        memories = self.search_memories(question, max_results=1)
        
        if memories and memories[0].get('relevance', 0) > 0.85:
            if memories[0].get('validated', False):
                return memories[0].get('answer')
        
        return None
    
    # =========================================================================
    # Unified Response Generation
    # =========================================================================
    
    def think_and_respond(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Full cognitive response pipeline:
        1. Check memories for similar questions
        2. Use world model for physics queries
        3. Generate response with confidence
        
        This is the primary method for human-robot interaction.
        
        Args:
            message: User's message
            context: Optional context (images, robot state, etc.)
            
        Returns:
            Response dict with text, confidence, reasoning chain
        """
        context = context or {}
        reasoning_chain = []
        response = None
        confidence = 0.0
        
        # Step 1: Check semantic memory
        memory_response = self.answer_from_memory(message)
        if memory_response:
            reasoning_chain.append("Found answer in validated memory")
            response = memory_response
            confidence = 0.9
        
        # Step 2: Check if it's a physics/planning question
        if not response and self._is_physics_question(message):
            reasoning_chain.append("Detected physics/planning question")
            physics_response = self._handle_physics_question(message, context)
            if physics_response:
                response = physics_response
                confidence = 0.7
                reasoning_chain.append("Used world model for prediction")
        
        # Step 3: Check if it's a vision question
        if not response and self._is_vision_question(message):
            reasoning_chain.append("Detected vision question")
            vision_response = self._handle_vision_question(message, context)
            if vision_response:
                response = vision_response
                confidence = 0.8
                reasoning_chain.append("Used vision service")
        
        # Step 4: Use HOPE brain confidence check
        if not response:
            can_answer, hope_conf = self.can_answer(message)
            if can_answer:
                response = self.generate_response(message)
                confidence = hope_conf
                reasoning_chain.append(f"HOPE brain responded (conf: {hope_conf:.2f})")
        
        # Step 5: If still no response, return None for LLM fallback
        if not response:
            reasoning_chain.append("No confident answer - deferring to LLM")
            confidence = 0.0
        
        return {
            "response": response,
            "confidence": confidence,
            "reasoning_chain": reasoning_chain,
            "source": "hope_agent" if response else "defer_to_llm",
        }
    
    def _is_physics_question(self, message: str) -> bool:
        """Check if message is about physics/planning."""
        keywords = ['predict', 'what will happen', 'if i', 'plan', 'reach', 'move to', 'how to get']
        return any(k in message.lower() for k in keywords)
    
    def _is_vision_question(self, message: str) -> bool:
        """Check if message is about vision/seeing."""
        keywords = ['see', 'look', 'show', 'camera', 'object', 'what is', 'identify', 'segment']
        return any(k in message.lower() for k in keywords)
    
    def _handle_physics_question(self, message: str, context: Dict[str, Any]) -> Optional[str]:
        """Handle physics/planning questions using world model."""
        if not self.world_model:
            return None
        
        # Simple planning response
        current_state = context.get("robot_state", {"joint_positions": [0.5] * 6})
        
        # Try to extract goal from message
        if "reach" in message.lower() or "move to" in message.lower():
            result = self.plan_to_goal(
                current_state,
                {"target_positions": [0.7] * 6},  # Default target
            )
            if result["success"]:
                return (
                    f"I can plan a path with {result['steps']} steps. "
                    f"Total uncertainty: {result['total_uncertainty']:.2f}. "
                    f"Planning took {result['planning_time_ms']:.0f}ms."
                )
        
        return self.describe_world_understanding()
    
    def _handle_vision_question(self, message: str, context: Dict[str, Any]) -> Optional[str]:
        """Handle vision questions using VisionCore or SAM/Hailo."""
        # Try VisionCore first (unified perception)
        try:
            from continuonbrain.services.vision_core import create_vision_core
            vision_core = create_vision_core()
            if vision_core.is_ready():
                return vision_core.describe_scene()
        except Exception:
            pass
        
        # Fallback to direct vision service
        if not self.vision_service:
            return self._describe_vision_capability()
        
        # If we have an image in context, describe it
        image = context.get("image")
        if image:
            try:
                result = self.vision_service.segment_text(image, "object")
                if result and result.get("masks"):
                    return f"I can see {len(result['masks'])} objects in the image."
            except Exception:
                pass
        
        return self._describe_vision_capability()
