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

    # =========================================================================
    # Active Learning - Question Asking and Knowledge Gap Identification
    # =========================================================================

    def identify_knowledge_gaps(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify areas where HOPE lacks knowledge or confidence.

        This enables the agent to proactively ask for help in areas
        where it's uncertain, rather than making poor decisions.

        Args:
            context: Current context (visual, state, task)

        Returns:
            List of knowledge gaps with gap type, description, and priority
        """
        gaps = []
        context = context or {}

        # Check brain stability - unstable state indicates learning gaps
        if self.brain and hasattr(self.brain, 'columns'):
            try:
                col = self.brain.columns[self.brain.active_column_idx]
                metrics = col.stability_monitor.get_metrics()

                # High novelty indicates unfamiliar situations
                novelty = metrics.get('novelty', 0.0)
                if novelty > 0.7:
                    gaps.append({
                        "type": "high_novelty",
                        "description": "I'm seeing something unfamiliar",
                        "priority": "high",
                        "novelty_score": novelty,
                        "suggested_question": "What is this object/situation I'm seeing?",
                    })

                # Low confidence indicates uncertainty
                confidence = getattr(col, 'last_confidence', 0.5)
                if confidence < 0.4:
                    gaps.append({
                        "type": "low_confidence",
                        "description": "I'm not confident about what to do",
                        "priority": "medium",
                        "confidence_score": confidence,
                        "suggested_question": "Can you help me understand what I should do here?",
                    })

                # Unstable brain state
                if not metrics.get('is_stable', True):
                    gaps.append({
                        "type": "unstable_state",
                        "description": "My internal state is uncertain",
                        "priority": "high",
                        "suggested_question": "I'm having trouble - can you guide me?",
                    })

            except Exception as e:
                logger.warning(f"Error checking brain stability: {e}")

        # Check vision understanding gaps
        if context.get("visual_perception"):
            perception = context["visual_perception"]
            objects = perception.get("objects", [])

            # Low-confidence detections
            uncertain_objects = [
                obj for obj in objects
                if obj.get("score", 1.0) < 0.6
            ]
            if uncertain_objects:
                gaps.append({
                    "type": "uncertain_detection",
                    "description": f"I see {len(uncertain_objects)} objects I'm not sure about",
                    "priority": "medium",
                    "objects": uncertain_objects,
                    "suggested_question": "Can you tell me what these objects are?",
                })

            # Unknown object types (no labels)
            unlabeled = [
                obj for obj in objects
                if not obj.get("label") or obj.get("label") == "unknown"
            ]
            if unlabeled:
                gaps.append({
                    "type": "unknown_objects",
                    "description": f"I see {len(unlabeled)} objects I can't identify",
                    "priority": "high",
                    "objects": unlabeled,
                    "suggested_question": "What are these objects?",
                })

        # Check task understanding
        current_task = context.get("current_task")
        if current_task and not self._understands_task(current_task):
            gaps.append({
                "type": "task_unclear",
                "description": f"I don't fully understand the task: {current_task}",
                "priority": "high",
                "suggested_question": f"Can you explain what you mean by '{current_task}'?",
            })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda g: priority_order.get(g.get("priority", "low"), 2))

        return gaps

    def _understands_task(self, task: str) -> bool:
        """Check if HOPE has knowledge about a task."""
        # Simple heuristic - check if task keywords are in known domains
        known_domains = [
            "move", "drive", "turn", "forward", "backward",
            "arm", "reach", "grasp", "pick", "place",
            "look", "see", "find", "identify",
            "stop", "pause", "wait", "safe",
        ]
        task_lower = task.lower()
        return any(domain in task_lower for domain in known_domains)

    def generate_clarifying_questions(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        max_questions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate clarifying questions when HOPE doesn't understand.

        This is the core "ask for help" capability that enables
        HOPE to learn from human guidance.

        Args:
            message: The user's message or command
            context: Optional context (visual, state)
            max_questions: Maximum number of questions to generate

        Returns:
            List of questions with type, text, and options
        """
        questions = []
        context = context or {}
        message_lower = message.lower()

        # Check for ambiguous object references
        ambiguous_refs = ["it", "that", "this", "the thing", "the object"]
        for ref in ambiguous_refs:
            if ref in message_lower:
                # If we have visual context, offer options
                objects = context.get("visual_perception", {}).get("objects", [])
                if objects:
                    options = [
                        {"id": i, "description": obj.get("description", f"Object {i}")}
                        for i, obj in enumerate(objects[:5])
                    ]
                    questions.append({
                        "type": "object_reference",
                        "question": f"When you say '{ref}', which object do you mean?",
                        "options": options,
                        "requires_selection": True,
                    })
                else:
                    questions.append({
                        "type": "object_reference",
                        "question": f"When you say '{ref}', what are you referring to?",
                        "options": [],
                        "requires_text": True,
                    })
                break

        # Check for vague location references
        vague_locations = ["over there", "here", "somewhere", "around"]
        for loc in vague_locations:
            if loc in message_lower:
                questions.append({
                    "type": "location_clarification",
                    "question": f"Where exactly do you mean by '{loc}'? Can you point or describe?",
                    "options": [
                        {"id": "left", "description": "Left side of view"},
                        {"id": "right", "description": "Right side of view"},
                        {"id": "center", "description": "Center of view"},
                        {"id": "near", "description": "Close to me"},
                        {"id": "far", "description": "Far from me"},
                    ],
                    "requires_selection": True,
                })
                break

        # Check for unknown action words
        action_words = self._extract_action_words(message)
        unknown_actions = [
            action for action in action_words
            if not self._knows_action(action)
        ]
        if unknown_actions:
            questions.append({
                "type": "action_clarification",
                "question": f"I don't know how to '{unknown_actions[0]}'. Can you show me or explain?",
                "options": [
                    {"id": "show", "description": "I'll show you (demonstration)"},
                    {"id": "explain", "description": "Let me explain in words"},
                    {"id": "similar", "description": "It's similar to something you know"},
                ],
                "requires_selection": True,
            })

        # Check for quantity/degree ambiguity
        vague_quantities = ["a bit", "some", "a lot", "more", "less", "faster", "slower"]
        for qty in vague_quantities:
            if qty in message_lower:
                questions.append({
                    "type": "quantity_clarification",
                    "question": f"How much do you mean by '{qty}'?",
                    "options": [
                        {"id": "small", "description": "A small amount (10-20%)"},
                        {"id": "medium", "description": "A medium amount (30-50%)"},
                        {"id": "large", "description": "A large amount (70-100%)"},
                    ],
                    "requires_selection": True,
                })
                break

        return questions[:max_questions]

    def _extract_action_words(self, message: str) -> List[str]:
        """Extract action verbs from a message."""
        # Simple extraction - look for words after "please" or at start
        words = message.lower().split()
        actions = []

        # Common action starters
        for i, word in enumerate(words):
            if word in ["please", "can", "could", "would", "will"]:
                if i + 1 < len(words):
                    actions.append(words[i + 1])
            elif i == 0 and word.endswith(("e", "k", "p", "t", "ch", "sh")):
                # Likely imperative verb
                actions.append(word)

        return actions

    def _knows_action(self, action: str) -> bool:
        """Check if HOPE knows how to perform an action."""
        known_actions = {
            "move", "go", "drive", "turn", "stop", "pause", "wait",
            "look", "see", "find", "watch", "show",
            "reach", "grasp", "grab", "pick", "place", "put", "drop",
            "say", "speak", "tell", "explain",
            "learn", "remember", "forget",
        }
        return action in known_actions

    def analyze_scene_for_learning(
        self,
        segmentation_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a scene and generate questions to learn from it.

        This is the proactive learning capability - HOPE looks at
        what it sees and asks the user to help it understand.

        Args:
            segmentation_data: Current segmentation data from vision

        Returns:
            Dict with scene analysis and learning questions
        """
        perception = self.get_visual_perception(segmentation_data)

        result = {
            "scene_description": perception.get("scene_description", "No scene data"),
            "objects_detected": perception.get("num_objects", 0),
            "learning_opportunities": [],
            "questions": [],
        }

        objects = perception.get("objects", [])

        # Identify learning opportunities
        for obj in objects:
            score = obj.get("score", 1.0)
            label = obj.get("label", "unknown")

            # Low confidence detection - good learning opportunity
            if score < 0.7:
                result["learning_opportunities"].append({
                    "type": "uncertain_object",
                    "object_id": obj.get("id"),
                    "current_score": score,
                    "description": obj.get("description", "Unknown object"),
                })
                result["questions"].append({
                    "question": f"I see something at {obj.get('description', 'unknown location')} but I'm only {score:.0%} sure what it is. Can you tell me what it is?",
                    "type": "object_identification",
                    "object_id": obj.get("id"),
                    "priority": "high" if score < 0.5 else "medium",
                })

            # Unknown label
            if label in ["unknown", "object", None, ""]:
                result["learning_opportunities"].append({
                    "type": "unlabeled_object",
                    "object_id": obj.get("id"),
                    "description": obj.get("description"),
                })
                result["questions"].append({
                    "question": f"What is the object I see at the {obj.get('description', 'center')}?",
                    "type": "object_naming",
                    "object_id": obj.get("id"),
                    "priority": "high",
                })

        # Check for spatial relationships we don't understand
        if len(objects) > 1:
            result["questions"].append({
                "question": f"I see {len(objects)} objects. Can you tell me how they relate to each other?",
                "type": "spatial_relationship",
                "priority": "low",
            })

        # Ask about affordances (what can be done with objects)
        if objects:
            result["questions"].append({
                "question": "What can I do with these objects? Can I pick them up, move them, or should I avoid them?",
                "type": "affordance_learning",
                "priority": "medium",
            })

        return result

    def learn_from_correction(
        self,
        original_response: str,
        correction: str,
        context: Optional[Dict[str, Any]] = None,
        experience_logger=None
    ) -> Dict[str, Any]:
        """
        Learn from a user correction to improve future responses.

        This allows HOPE to incorporate human feedback directly
        into its knowledge base.

        Args:
            original_response: What HOPE said/did
            correction: What the user said was correct
            context: Optional context for this correction
            experience_logger: Optional logger for storing learned corrections

        Returns:
            Dict with learning result and updated confidence
        """
        result = {
            "learned": False,
            "memory_stored": False,
            "confidence_impact": 0.0,
            "message": "",
        }

        try:
            # Store the correction as a learned experience
            if experience_logger:
                experience_logger.store_correction(
                    original=original_response,
                    correction=correction,
                    context=context or {},
                    timestamp_ns=int(__import__('time').time_ns()),
                )
                result["memory_stored"] = True

            # Update brain state to reduce confidence in similar situations
            if self.brain and hasattr(self.brain, 'columns'):
                col = self.brain.columns[self.brain.active_column_idx]

                # Lower confidence for similar future inputs
                if hasattr(col, 'last_confidence'):
                    old_conf = col.last_confidence
                    col.last_confidence = max(0.1, old_conf - 0.2)
                    result["confidence_impact"] = col.last_confidence - old_conf

            result["learned"] = True
            result["message"] = (
                f"Thank you for the correction. I've learned that "
                f"'{original_response[:50]}...' should be '{correction[:50]}...'. "
                f"I'll remember this for next time."
            )

        except Exception as e:
            logger.error(f"Error learning from correction: {e}")
            result["message"] = f"I had trouble learning from that correction: {e}"

        return result

    def should_ask_question(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Decide if HOPE should ask a clarifying question before responding.

        This is the main entry point for the question-asking system.

        Args:
            message: User's message
            context: Current context

        Returns:
            (should_ask, question_dict) - True if should ask, with the question
        """
        context = context or {}

        # First, check our confidence
        can_answer, confidence = self.can_answer(message)

        # If confidence is very low, we should definitely ask
        if confidence < 0.3:
            gaps = self.identify_knowledge_gaps(context)
            if gaps:
                return (True, {
                    "reason": "low_confidence",
                    "confidence": confidence,
                    "question": gaps[0].get("suggested_question", "I'm not sure I understand. Can you help me?"),
                    "gap": gaps[0],
                })

        # Check for clarifying questions needed
        clarifying = self.generate_clarifying_questions(message, context)
        if clarifying:
            return (True, {
                "reason": "needs_clarification",
                "confidence": confidence,
                "question": clarifying[0]["question"],
                "options": clarifying[0].get("options", []),
                "type": clarifying[0]["type"],
            })

        # If medium confidence and we have knowledge gaps, might want to ask
        if confidence < 0.6:
            gaps = self.identify_knowledge_gaps(context)
            high_priority_gaps = [g for g in gaps if g.get("priority") == "high"]
            if high_priority_gaps:
                return (True, {
                    "reason": "knowledge_gap",
                    "confidence": confidence,
                    "question": high_priority_gaps[0].get("suggested_question"),
                    "gap": high_priority_gaps[0],
                })

        # High enough confidence, don't need to ask
        return (False, None)
